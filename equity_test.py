"""
Scores the salmon signal against listed Norwegian salmon-farming equities, which
are liquid and free to trade, instead of the SSB spot proxy.

    python equity_test.py                 # cached prices
    python equity_test.py --refresh       # re-download prices
    python equity_test.py --cost-bps 25 --from 2018-01-01

The signal is the walk-forward forecast from main.py, unchanged. Each position is
put on at Wednesday's close, when SSB publishes the previous week, and earns the
Wednesday-to-Wednesday return of the basket. Three P&L versions:
  spot      SSB spot return, as in main.py
  basket    equal-weight Mowi, SalMar, Leroy, Grieg, Bakkafrost (whichever are listed)
  hedged    basket minus rolling beta x an equal-weight basket of non-seafood Oslo stocks
"""
import argparse

import numpy as np
import pandas as pd

from backtest import run_backtest, walk_forward
from config import Config
from features import ENV_FEATURES, MARKET_FEATURES, SUPPLY_FEATURES, build_dataset
from main import SEP, _header, load_data

SALMON = ['MOWI.OL', 'SALM.OL', 'LSG.OL', 'GSF.OL', 'BAKKA.OL']
CONTROLS = ['EQNR.OL', 'DNB.OL', 'TEL.OL', 'ORK.OL', 'YAR.OL', 'NHY.OL']
BETA_WEEKS = 104


def fetch_prices(raw_dir, refresh: bool) -> pd.DataFrame:
    path = raw_dir / 'equities.csv'
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    frames = {}
    for t in SALMON + CONTROLS:
        for attempt in range(3):  # Yahoo occasionally drops the TLS connection
            d = yf.download(t, start='2000-01-01', progress=False, auto_adjust=True)
            if len(d):
                frames[t] = d['Close'].squeeze()
                break
        else:
            raise RuntimeError(f'could not download {t}')
    px = pd.DataFrame(frames)
    px.to_csv(path)
    return px


def wednesday_log_prices(px: pd.DataFrame, week_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Last close on or before the Wednesday of each SSB week (index = that week's Sunday)."""
    daily = px.reindex(pd.date_range(px.index[0], px.index[-1], freq='D')).ffill()
    wed = week_index - pd.Timedelta(days=4)
    wed = wed[wed <= daily.index[-1]]
    out = np.log(daily.loc[wed])
    out.index = wed + pd.Timedelta(days=4)
    return out


def basket_returns(px: pd.DataFrame, week_index: pd.DatetimeIndex) -> pd.DataFrame:
    r = wednesday_log_prices(px, week_index).diff()
    salmon = r[[t for t in SALMON if t in r]].mean(axis=1, skipna=True)
    market = r[[t for t in CONTROLS if t in r]].mean(axis=1, skipna=True)
    # Beta on past weeks only, then applied to this week.
    beta = (salmon.rolling(BETA_WEEKS, min_periods=52).cov(market)
            / market.rolling(BETA_WEEKS, min_periods=52).var()).shift(1).fillna(1.0)
    return pd.DataFrame({'basket': salmon, 'market': market, 'beta': beta,
                         'hedged': salmon - beta * market})


def pnl_stats(position: pd.Series, ret: pd.Series, cost_bps: float, start: str) -> dict:
    d = pd.concat([position, ret], axis=1, keys=['pos', 'ret']).dropna().loc[start:]
    net = d['pos'] * d['ret'] - d['pos'].diff().abs().fillna(0) * cost_bps / 1e4
    eq = net.cumsum()
    ann, vol = net.mean() * 52, net.std() * np.sqrt(52)
    return {'weeks': len(d), 'ann_ret': ann, 'ann_vol': vol, 'sharpe': ann / vol,
            'max_dd': (eq - eq.cummax()).min(), 'week_hit': (np.sign(d['pos']) == np.sign(d['ret']))[d['pos'].abs() > 1e-9].mean(),
            'net': net}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--cost-bps', type=float, default=15.0, help='per unit of position change (equities)')
    p.add_argument('--from', dest='start', default='2010-01-01', help='first week scored')
    args = p.parse_args()

    cfg = Config()
    ssb, oisst, weather = load_data(cfg, refresh=False)
    X, y, mkt = build_dataset(ssb, oisst, weather, cfg)
    s = cfg.strategy
    pred = walk_forward(X[MARKET_FEATURES + SUPPLY_FEATURES + ENV_FEATURES], y, s)
    bt = run_backtest(pred, mkt['adj_ret'], s)

    px = fetch_prices(cfg.raw_dir, args.refresh)
    eq = basket_returns(px, mkt.index)
    print(f'  Equity prices {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}, {len(px.columns)} tickers')

    lag = s.horizon_weeks + s.publication_lag_weeks
    rets = {'spot (SSB adj.)': mkt['adj_ret'], 'basket': eq['basket'], 'hedged': eq['hedged']}
    pos = bt['position']

    _header(f'SALMON SIGNAL vs LISTED EQUITIES  (scored from {args.start}, {args.cost_bps:g}bp costs)')
    print(f'  Correlation of weekly returns with spot adj. return (since {args.start}):')
    c = pd.concat([mkt['adj_ret'], eq['basket'], eq['hedged']], axis=1, keys=['spot', 'basket', 'hedged']).loc[args.start:].dropna()
    print('   ', ', '.join(f'{k} {c["spot"].corr(c[k]):.2f}' for k in ('basket', 'hedged')))
    print(f'  Average beta of basket to non-seafood Oslo stocks: {eq["beta"].loc[args.start:].mean():.2f}\n')

    rows, nets = {}, {}
    for name, r in rets.items():
        cost = 25.0 if name.startswith('spot') else args.cost_bps
        st = pnl_stats(pos, r, cost, args.start)
        nets[name] = st.pop('net')
        # IC against the matching 4-week forward return.
        fwd = r.rolling(s.horizon_weeks).sum().shift(-lag)
        sc = pd.concat([pred, fwd], axis=1, keys=['p', 'f']).dropna().loc[args.start:]
        ic = sc['p'].corr(sc['f'])
        st['IC'] = ic
        st['IC_t_eff'] = ic * np.sqrt(len(sc) / s.horizon_weeks - 2) / np.sqrt(1 - ic ** 2)
        rows[name] = st
    table = pd.DataFrame(rows).T[['weeks', 'IC', 'IC_t_eff', 'sharpe', 'ann_ret', 'ann_vol', 'max_dd', 'week_hit']]
    fmt = {'weeks': '{:.0f}', 'IC': '{:.3f}', 'IC_t_eff': '{:.2f}', 'sharpe': '{:.2f}', 'ann_ret': '{:.1%}',
           'ann_vol': '{:.1%}', 'max_dd': '{:.1%}', 'week_hit': '{:.1%}'}
    print(table.apply(lambda col: col.map(fmt[col.name].format)).to_string())

    print('\n  Net log return by year:')
    print(pd.DataFrame({k: v.groupby(v.index.year).sum() for k, v in nets.items()}).round(3).to_string())
    print('\n  Positions are the main.py book (vol-capped, 1.3x), so ann_ret is not comparable across')
    print('  rows; Sharpe and IC are. Shorting the basket also costs borrow, not charged here.')
    print(f'\n{SEP}')


if __name__ == '__main__':
    main()
