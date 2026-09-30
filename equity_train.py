"""
Trains the model directly on the return of listed salmon farmers (market-hedged),
the only salmon exposure with free, liquid, daily-tradable prices.

    python equity_train.py

Salmon futures history (Euronext / Fish Pool / Sitagri) is paywalled and the EUR
contract only started in July 2024, so it cannot be tested from free data.

Economic case: farmer equities are a levered claim on the salmon price, but they
price it through slow earnings expectations. Spot deviations from seasonal, supply
pressure and storms reach earnings with a delay, so the equities may lag them.
Longer horizons (earnings drift) may fit shares better than the 4-week spot
horizon, so 4, 8 and 13 weeks are all shown. Six variants are tried in total;
read the best one with that in mind.
"""
import dataclasses

import numpy as np
import pandas as pd

from backtest import performance, run_backtest, walk_forward
from config import Config
from equity_test import basket_returns, fetch_prices
from features import ENV_FEATURES, MARKET_FEATURES, SUPPLY_FEATURES, build_dataset
from main import SEP, _header, load_data

EQ_FEATURES = ['eq_mom_4', 'eq_mom_13', 'eq_dev_26']
COST_BPS = 15.0


def equity_features(ret: pd.Series) -> pd.DataFrame:
    """Hedged-basket momentum, using only returns through the week being scored."""
    logp = ret.fillna(0).cumsum()
    return pd.DataFrame({
        'eq_mom_4': logp.diff(4),
        'eq_mom_13': logp.diff(13),
        'eq_dev_26': logp - logp.rolling(26).mean(),
    })


def main():
    cfg = Config()
    ssb, oisst, weather = load_data(cfg, refresh=False)
    px = fetch_prices(cfg.raw_dir, refresh=False)

    rows = {}
    for h in (4, 8, 13):
        c = dataclasses.replace(cfg, strategy=dataclasses.replace(
            cfg.strategy, horizon_weeks=h, cost_bps=COST_BPS))
        s = c.strategy
        X, _, mkt = build_dataset(ssb, oisst, weather, c)
        r = basket_returns(px, mkt.index)['hedged'].reindex(mkt.index)
        lag = h + s.publication_lag_weeks
        y = r.rolling(h).sum().shift(-lag).reindex(X.index)
        X = X.join(equity_features(r).reindex(X.index))
        base = MARKET_FEATURES + SUPPLY_FEATURES + ENV_FEATURES
        for label, cols in (('spot+env', base), ('spot+env+equity', base + EQ_FEATURES)):
            pred = walk_forward(X[cols], y, s)
            bt = run_backtest(pred, r, s)
            m = performance(bt, pred, y, h)
            m['sharpe_2018'] = bt.loc['2018':, 'net'].mean() / bt.loc['2018':, 'net'].std() * np.sqrt(52)
            rows[f'{h}w, {label}'] = m

    _header('MODEL TRAINED ON HEDGED SALMON-EQUITY BASKET  (2010-26, 15bp costs, no COVID window)')
    t = pd.DataFrame(rows).T[['IC', 'IC_t_eff', 'sharpe', 'sharpe_2018', 'ann_ret', 'ann_vol', 'max_dd']]
    fmt = {'IC': '{:.3f}', 'IC_t_eff': '{:.2f}', 'sharpe': '{:.2f}', 'sharpe_2018': '{:.2f}',
           'ann_ret': '{:.1%}', 'ann_vol': '{:.1%}', 'max_dd': '{:.1%}'}
    print(t.apply(lambda col: col.map(fmt[col.name].format)).to_string())
    print('\n  IC_t_eff uses n/horizon independent observations. Six variants were tried.')
    print(f'\n{SEP}')


if __name__ == '__main__':
    main()
