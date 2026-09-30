"""
Salmon futures strategy driven by public ocean and weather data at the farms.

Predicts how Norway's weekly salmon export price will move relative to its
normal seasonal path over the next few weeks, using sea temperature (NOAA
OISST) and weather (ERA5 or NASA POWER) at every major farming region plus SSB
price and volume, then trades the forecast as a long/short front-month position.

Usage:
    python main.py                      # cached data (first run downloads, ~2 min)
    python main.py --refresh            # re-download everything
    python main.py --horizon 8 --model gbm
    python main.py --weather power      # NASA POWER instead of ERA5 (lagged 8w)
    python main.py --months 4-9         # only hold positions Apr-Sep
"""
import argparse
import dataclasses
from datetime import datetime

import pandas as pd

from backtest import (fit_final, no_trade_mask, performance, raw_book, ridge_contributions,
                      run_backtest, signal_tranches, vol_cap_scale, walk_forward)
from config import Config
from data_sources import fetch_era5, fetch_oisst, fetch_power, fetch_ssb_salmon
from features import (ENV_FEATURES, FEATURE_LABELS, MARKET_FEATURES, SUPPLY_FEATURES,
                      build_dataset)
import plots

FEATURE_SETS = {
    'all': MARKET_FEATURES + SUPPLY_FEATURES + ENV_FEATURES,
    'market + supply': MARKET_FEATURES + SUPPLY_FEATURES,
    'env only': ENV_FEATURES,
}
MAIN_SET = 'all'

SEP = '=' * 78


def _header(title: str) -> None:
    print(f'\n{SEP}\n  {title}\n{SEP}')


def _parse_months(text: str) -> tuple[int, ...]:
    """'4-9' -> Apr..Sep, '10-3' wraps -> Oct..Mar, '1,6-8' -> Jan, Jun..Aug."""
    months = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        a, b = int(a), int(b or a)
        months += [(m - 1) % 12 + 1 for m in range(a, b + 1 if b >= a else b + 13)]
    return tuple(sorted(set(months)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--refresh', action='store_true', help='re-download all data')
    p.add_argument('--horizon', type=int, help='forecast / holding horizon in weeks')
    p.add_argument('--model', choices=['ridge', 'gbm'])
    p.add_argument('--weather', choices=['era5', 'power'], help='wind/solar data source')
    p.add_argument('--test-start', help='first out-of-sample week (YYYY-MM-DD)')
    p.add_argument('--cost-bps', type=float, help='cost per unit of position change')
    p.add_argument('--leverage', type=float, help='multiplier on the vol-capped book')
    p.add_argument('--no-vol-cap', action='store_true', help='size without the P&L vol cap')
    p.add_argument('--months', type=_parse_months, help="months to trade, e.g. '4-9' or '10-3'")
    p.add_argument('--no-plots', action='store_true')
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    overrides = {
        'horizon_weeks': args.horizon,
        'model': args.model,
        'test_start': args.test_start,
        'cost_bps': args.cost_bps,
        'leverage': args.leverage,
        'trade_months': args.months,
    }
    cfg.strategy = dataclasses.replace(
        cfg.strategy, **{k: v for k, v in overrides.items() if v is not None})
    if args.no_vol_cap:
        cfg.strategy = dataclasses.replace(cfg.strategy, vol_cap_halflife=None)
    if args.weather:
        cfg.env = dataclasses.replace(cfg.env, weather_source=args.weather)
    return cfg


def load_data(cfg: Config, refresh: bool):
    _header('DATA')
    ssb = fetch_ssb_salmon(cfg.raw_dir, refresh)
    print(f'  SSB salmon export price/volume  {ssb.index[0]:%Y-%m-%d} -> {ssb.index[-1]:%Y-%m-%d}')
    if cfg.env.weather_source == 'power':
        weather = fetch_power(cfg.regions, cfg.env, cfg.raw_dir, refresh)
        source = f'NASA POWER wind/solar, {cfg.env.power_lag_weeks}w lag'
    else:
        weather = fetch_era5(cfg.regions, cfg.env, cfg.raw_dir, refresh)
        source = 'ERA5 wind/solar (Open-Meteo)'
    last_wx = min(df.dropna().index[-1] for df in weather.values())
    print(f'  {source:<32}{len(weather)} regions, through {last_wx:%Y-%m-%d}')
    oisst = fetch_oisst(cfg.regions, cfg.raw_dir, refresh)
    last_sst = min(df['sst'].dropna().index[-1] for df in oisst.values())
    print(f'  NOAA OISST sea temperature      {len(oisst)} regions, week of {last_sst:%Y-%m-%d}')
    return ssb, oisst, weather


def run_feature_sets(X: pd.DataFrame, y: pd.Series, mkt: pd.DataFrame, cfg: Config) -> dict:
    results = {}
    for name, cols in FEATURE_SETS.items():
        pred = walk_forward(X[cols], y, cfg.strategy)
        bt = run_backtest(pred, mkt['adj_ret'], cfg.strategy)
        results[name] = {
            'pred': pred,
            'bt': bt,
            'metrics': performance(bt, pred, y, cfg.strategy.horizon_weeks),
        }
    return results


def print_metrics(results: dict, cfg: Config) -> pd.DataFrame:
    s = cfg.strategy
    bt = results[MAIN_SET]['bt']
    _header(f'WALK-FORWARD RESULTS  {bt.index[0]:%Y-%m-%d} -> {bt.index[-1]:%Y-%m-%d}  '
            f'({s.model}, {s.horizon_weeks}w horizon, {s.cost_bps:.0f}bp costs)')
    vol_cap = f'{s.vol_cap_halflife}w P&L vol cap' if s.vol_cap_halflife else 'no vol cap'
    print(f'  Sizing: {s.leverage:g}x leverage, {vol_cap}.')
    for start, end in s.no_trade_windows:
        print(f'  Flat from {start} to {end}.')
    if len(s.trade_months) < 12:
        print(f'  Trading only in months {", ".join(map(str, s.trade_months))}.')
    print('  Scored on SSB spot return minus its normal seasonal move (a futures proxy).\n')
    table = pd.DataFrame({name: r['metrics'] for name, r in results.items()}).T
    fmt = {
        'IC': '{:.3f}', 'rank_IC': '{:.3f}', 'IC_t_eff': '{:.2f}', 'dir_hit': '{:.1%}',
        'R2_vs_0': '{:.3f}', 'ann_ret': '{:.1%}', 'ann_vol': '{:.1%}', 'sharpe': '{:.2f}',
        'sharpe_act': '{:.2f}', 'in_mkt': '{:.0%}', 'max_dd': '{:.1%}', 'week_hit': '{:.1%}',
        'avg_abs_pos': '{:.2f}', 'turnover_yr': '{:.1f}',
    }
    shown = table.apply(lambda col: col.map(fmt[col.name].format))
    print(shown.to_string())
    print('\n  IC_t_eff: IC t-stat using n/horizon independent obs (labels overlap).'
          '\n  ann_ret, ann_vol, sharpe: per calendar year, flat weeks count as 0.'
          '\n  sharpe_act: only weeks in the market (in_mkt), annualised by sqrt(52).'
          '\n  max_dd is in log-return units.')
    return table


def live_signal(X: pd.DataFrame, y: pd.Series, mkt: pd.DataFrame, results: dict,
                cfg: Config) -> pd.DataFrame | None:
    s = cfg.strategy
    cols = FEATURE_SETS[MAIN_SET]
    model = fit_final(X[cols], y, s)
    latest = X[cols].iloc[[-1]]

    pred = results[MAIN_SET]['pred'].copy()
    pred.iloc[-1] = model.predict(latest)[0]
    tranche = signal_tranches(pred, s)
    unscaled = tranche.iloc[-s.horizon_weeks:].mean()
    scale = vol_cap_scale(raw_book(tranche, s), mkt['adj_ret'], s).iloc[-1]
    book = unscaled * scale * s.leverage
    next_week = X.index[-1] + pd.Timedelta(weeks=s.publication_lag_weeks + 1)
    if no_trade_mask(pd.DatetimeIndex([next_week]), s)[0]:
        book = 0.0

    week = X.index[-1]
    iso = week.isocalendar()
    first = week + pd.Timedelta(weeks=s.publication_lag_weeks + 1)
    last = week + pd.Timedelta(weeks=s.publication_lag_weeks + s.horizon_weeks)
    side = 'LONG' if book > 0.05 else 'SHORT' if book < -0.05 else 'FLAT'

    _header(f'LIVE SIGNAL  (data through {iso.year}-W{iso.week:02d}, week ending {week:%Y-%m-%d})')
    print(f'  Spot price                  {mkt["price"].iloc[-1]:.2f} NOK/kg')
    print(f'  Forecast vs seasonal path   {pred.iloc[-1]:+.1%} over weeks ending '
          f'{first:%Y-%m-%d} -> {last:%Y-%m-%d}')
    print(f'  New tranche                 {tranche.iloc[-1]:+.2f}')
    print(f'  {f"Avg of last {s.horizon_weeks} tranches":<28}{unscaled:+.2f}')
    print(f'  x vol cap x leverage        x{scale:.2f} x{s.leverage:g}')
    print(f'  Book                        {book:+.2f}  ->  {side} {abs(book):.2f}x capital '
          f'in front-month salmon futures')

    if s.model != 'ridge':
        return None
    contrib = ridge_contributions(model, latest)
    print('\n  Biggest drivers of this forecast (contribution, % over horizon):')
    for feat, row in contrib.head(8).iterrows():
        print(f'    {row["contribution"] * 100:+6.2f}  {FEATURE_LABELS[feat]:<52} (z = {row["z_now"]:+.2f})')
    return contrib


def save_results(results: dict, metrics: pd.DataFrame, contrib: pd.DataFrame | None,
                 X: pd.DataFrame, y: pd.Series, mkt: pd.DataFrame, cfg: Config,
                 make_plots: bool) -> None:
    out = cfg.results_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / 'metrics.csv')
    results[MAIN_SET]['bt'].join(mkt[['price', 'volume', 'seas_ret']]).to_csv(out / 'backtest.csv')
    preds = pd.DataFrame({name: r['pred'] for name, r in results.items()}).join(y)
    preds.to_csv(out / 'predictions.csv')
    X.to_csv(out / 'features.csv')
    if contrib is not None:
        contrib.to_csv(out / 'ridge_coefficients.csv')

    if make_plots:
        plots.tearsheet(results, mkt, MAIN_SET, out / 'tearsheet.png')
        if contrib is not None:
            plots.coefficients(contrib['coef'], out / 'ridge_coefficients.png')
    print(f'\n  Results written to {out}')


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    print(f'\n{SEP}\n  SALMON ENVIRONMENTAL SIGNAL STRATEGY\n  {datetime.now():%Y-%m-%d %H:%M}\n{SEP}')

    ssb, oisst, weather = load_data(cfg, args.refresh)
    X, y, mkt = build_dataset(ssb, oisst, weather, cfg)
    results = run_feature_sets(X, y, mkt, cfg)
    metrics = print_metrics(results, cfg)
    contrib = live_signal(X, y, mkt, results, cfg)
    save_results(results, metrics, contrib, X, y, mkt, cfg, not args.no_plots)


if __name__ == '__main__':
    main()
