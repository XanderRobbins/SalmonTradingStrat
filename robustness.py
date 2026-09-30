"""
Robustness checks for the salmon strategy. Does not change the main pipeline.

    python robustness.py                    # everything, 200 placebo reps
    python robustness.py --reps 500 --vol-target 0.12
    python robustness.py --skip-placebo

Sections:
  1. Is the price edge measurement noise? (variance ratios, drop mom_1, enter a week later, costs)
  2. Is the environmental lift real? (small model, wind placebo, bootstrap CI)
  3. Selection bias: ridge alpha chosen inside the walk-forward
  4. Sizing without hindsight: ex-ante vol target, no hand-picked COVID window
"""
import argparse
import dataclasses

import numpy as np
import pandas as pd

from backtest import (make_model, no_trade_mask, performance, raw_book, run_backtest,
                      signal_tranches, walk_forward)
from config import Config
from features import ENV_FEATURES, MARKET_FEATURES, SUPPLY_FEATURES, build_dataset
from main import SEP, _header, load_data

BASE = MARKET_FEATURES + SUPPLY_FEATURES
WIND = [f for f in ENV_FEATURES if 'wind' in f]
COLS = ['IC', 'IC_t_eff', 'sharpe', 'sharpe_act', 'ann_ret', 'max_dd']


def evaluate(X, y, mkt, cols, cfg_s, pred=None):
    if pred is None:
        pred = walk_forward(X[cols], y, cfg_s)
    bt = run_backtest(pred, mkt['adj_ret'], cfg_s)
    return pred, bt, performance(bt, pred, y, cfg_s.horizon_weeks)


def show(rows: dict) -> None:
    table = pd.DataFrame(rows).T[COLS]
    fmt = {'IC': '{:.3f}', 'IC_t_eff': '{:.2f}', 'sharpe': '{:.2f}', 'sharpe_act': '{:.2f}',
           'ann_ret': '{:.1%}', 'max_dd': '{:.1%}'}
    print(table.apply(lambda c: c.map(fmt[c.name].format)).to_string())


# =============================================================================
# 1. Is the price edge measurement noise?
# =============================================================================

def variance_ratio(ret: pd.Series, q: int) -> float:
    ret = ret.dropna()
    return ret.rolling(q).sum().dropna().var() / (q * ret.var())


def section_noise(ssb, oisst, weather, cfg, X, y, mkt):
    _header('1. IS THE PRICE EDGE MEASUREMENT NOISE?')
    print('  Variance ratio of weekly log returns (1.0 = random walk, <1 = mean reversion).')
    print('  Strong reversion at q=2 that fades by q=8 looks like averaging noise, not a tradable move.')
    for q in (2, 4, 8, 13):
        print(f'    VR({q:>2}) = {variance_ratio(mkt["ret"], q):.3f}')
    print()

    s, all_cols = cfg.strategy, BASE + ENV_FEATURES
    rows = {'baseline (all)': evaluate(X, y, mkt, all_cols, s)[2]}
    rows['drop mom_1'] = evaluate(X, y, mkt, [c for c in all_cols if c != 'mom_1'], s)[2]

    late = dataclasses.replace(cfg, strategy=dataclasses.replace(s, publication_lag_weeks=2))
    X2, y2, mkt2 = build_dataset(ssb, oisst, weather, late)
    rows['enter 1 week later'] = evaluate(X2, y2, mkt2, all_cols, late.strategy)[2]

    pred = walk_forward(X[all_cols], y, s)
    for bps in (50, 100):
        cs = dataclasses.replace(s, cost_bps=float(bps))
        rows[f'costs {bps}bp'] = evaluate(X, y, mkt, all_cols, cs, pred)[2]
    show(rows)


# =============================================================================
# 2. Is the environmental lift real?
# =============================================================================

def block_bootstrap_ic(pred: pd.Series, y: pd.Series, block: int = 13, n: int = 2000, seed: int = 0):
    d = pd.concat([pred, y], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(d) / block))
    starts_max = len(d) - block
    ics = np.empty(n)
    for k in range(n):
        idx = (rng.integers(0, starts_max + 1, n_blocks)[:, None] + np.arange(block)).ravel()[:len(d)]
        ics[k] = np.corrcoef(d[idx, 0], d[idx, 1])[0, 1]
    return np.percentile(ics, [2.5, 97.5])


def shuffle_wind(X: pd.DataFrame, groups: list[np.ndarray], wind_pos: list[int], rng) -> pd.DataFrame:
    """Permute wind across years within each ISO week (all groups together, so they stay correlated)."""
    Xs = X.copy()
    vals = X.to_numpy()
    out = Xs.to_numpy().copy()
    for idx in groups:
        out[np.ix_(idx, wind_pos)] = vals[np.ix_(rng.permutation(idx), wind_pos)]
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def section_env(X, y, mkt, cfg, reps: int, skip_placebo: bool):
    _header('2. IS THE ENVIRONMENTAL LIFT REAL?')
    s, all_cols = cfg.strategy, BASE + ENV_FEATURES
    sets = {
        'market + supply': BASE,
        'market + supply + wind': BASE + WIND,
        'all (25 features)': all_cols,
    }
    preds, rows = {}, {}
    for name, cols in sets.items():
        preds[name], _, rows[name] = evaluate(X, y, mkt, cols, s)
    show(rows)

    print('\n  Block-bootstrap 95% CI for IC (13-week blocks):')
    for name, p in preds.items():
        lo, hi = block_bootstrap_ic(p, y)
        print(f'    {name:<26} [{lo:+.3f}, {hi:+.3f}]')

    if skip_placebo:
        return
    print(f'\n  Wind placebo: shuffle wind across years within each ISO week, refit, {reps} reps.')
    woy = X.index.isocalendar().week.clip(upper=52).to_numpy()
    groups = [np.flatnonzero(woy == w) for w in np.unique(woy)]
    cols = BASE + ENV_FEATURES
    wind_pos = [cols.index(c) for c in WIND]
    Xc = X[cols]
    rng = np.random.default_rng(0)
    ics, sharpes = [], []
    for _ in range(reps):
        _, _, m = evaluate(shuffle_wind(Xc, groups, wind_pos, rng), y, mkt, cols, s)
        ics.append(m['IC'])
        sharpes.append(m['sharpe'])
    ics, sharpes = np.array(ics), np.array(sharpes)
    real = rows['all (25 features)']
    print(f'    real IC {real["IC"]:.3f}   placebo IC mean {ics.mean():.3f}, '
          f'95th pct {np.percentile(ics, 95):.3f}   p = {(ics >= real["IC"]).mean():.3f}')
    print(f'    real Sharpe {real["sharpe"]:.2f}   placebo Sharpe mean {sharpes.mean():.2f}, '
          f'95th pct {np.percentile(sharpes, 95):.2f}   p = {(sharpes >= real["sharpe"]).mean():.3f}')
    print('    p < 0.05: wind carries information beyond what the other features and the calendar give.')


# =============================================================================
# 3. Nested hyperparameter selection
# =============================================================================

def walk_forward_nested(X, y, cfg_s, alphas=(50, 200, 800), val_weeks=260) -> pd.Series:
    """Like walk_forward, but ridge alpha is picked at each refit from the last `val_weeks` of known labels."""
    lag = cfg_s.horizon_weeks + cfg_s.publication_lag_weeks
    start = X.index.searchsorted(pd.Timestamp(cfg_s.test_start))
    pred = pd.Series(np.nan, index=X.index, name='pred')
    chosen = {}
    for i in range(start, len(X), cfg_s.retrain_every):
        usable = y.iloc[: i - lag + 1].dropna()
        alpha = cfg_s.ridge_alpha
        if len(usable) > val_weeks + lag + 200:
            val = usable.iloc[-val_weeks:]
            inner = usable.iloc[: -val_weeks - lag]  # gap so no inner label overlaps the validation window
            errs = {}
            for a in alphas:
                m = make_model(dataclasses.replace(cfg_s, ridge_alpha=a)).fit(X.loc[inner.index], inner)
                errs[a] = ((m.predict(X.loc[val.index]) - val) ** 2).mean()
            alpha = min(errs, key=errs.get)
        chosen[X.index[i]] = alpha
        model = make_model(dataclasses.replace(cfg_s, ridge_alpha=alpha)).fit(X.loc[usable.index], usable)
        block = slice(i, i + cfg_s.retrain_every)
        pred.iloc[block] = model.predict(X.iloc[block])
    pred.attrs['alphas'] = pd.Series(chosen)
    return pred


def section_nested(X, y, mkt, cfg):
    _header('3. RIDGE ALPHA CHOSEN INSIDE THE WALK-FORWARD')
    s, cols = cfg.strategy, BASE + ENV_FEATURES
    pred = walk_forward_nested(X[cols], y, s)
    rows = {
        f'fixed alpha = {s.ridge_alpha:g}': evaluate(X, y, mkt, cols, s)[2],
        'alpha picked in-sample-of-past': evaluate(X, y, mkt, cols, s, pred)[2],
    }
    show(rows)
    print('\n  Alpha chosen, share of refits:')
    print('   ', pred.attrs['alphas'].value_counts(normalize=True).sort_index().round(2).to_dict())
    print('  Not nested: climatology window, dropped cold-snap features, vol-cap settings.')


# =============================================================================
# 4. Sizing without hindsight
# =============================================================================

def vol_target_backtest(pred, adj_ret, cfg_s, target, max_lev=3.0, spike_flatten=False,
                        halflife=26) -> pd.DataFrame:
    """Position = raw book x (target vol / ex-ante book vol). Vol at week k uses P&L through k-2."""
    tranche = signal_tranches(pred, cfg_s)
    raw = raw_book(tranche, cfg_s)
    lag = cfg_s.publication_lag_weeks + 1
    pnl = (raw * adj_ret).where(raw.abs().cummax() > 0.05)
    vol = (pnl.ewm(halflife=halflife, min_periods=26).std() * np.sqrt(52)).shift(lag)
    scale = (target / vol).clip(upper=max_lev).fillna(1.0)
    if spike_flatten:
        # Flat when short-term book vol is 3x its own past median: a rule that needs no hindsight.
        fast = pnl.ewm(halflife=4, min_periods=8).std().shift(lag)
        typical = fast.expanding(min_periods=52).median()
        scale = scale.where(~(fast > 3 * typical), 0.0)
    position = (raw * scale).mask(no_trade_mask(raw.index, cfg_s), 0.0)
    gross = position * adj_ret
    cost = position.diff().abs() * cfg_s.cost_bps / 1e4
    bt = pd.DataFrame({'pred': pred, 'position': position, 'adj_ret': adj_ret,
                       'gross': gross, 'cost': cost, 'net': gross - cost})
    bt = bt.loc[pred.first_valid_index():].dropna(subset=['position', 'adj_ret'])
    bt['equity'] = bt['net'].cumsum()
    return bt


def section_sizing(X, y, mkt, cfg, target):
    _header(f'4. SIZING WITHOUT HINDSIGHT  (vol target {target:.0%}, max 3x)')
    s, cols = cfg.strategy, BASE + ENV_FEATURES
    pred = walk_forward(X[cols], y, s)
    no_windows = dataclasses.replace(s, no_trade_windows=())
    rows = {
        'default (cap, 1.3x, COVID flat)': performance(run_backtest(pred, mkt['adj_ret'], s), pred, y, s.horizon_weeks),
        'default, no COVID window': performance(run_backtest(pred, mkt['adj_ret'], no_windows), pred, y, s.horizon_weeks),
    }
    for label, spike in (('vol target', False), ('vol target + spike flatten', True)):
        bt = vol_target_backtest(pred, mkt['adj_ret'], no_windows, target, spike_flatten=spike)
        rows[label] = performance(bt, pred, y, s.horizon_weeks)
    show(rows)
    print('\n  Vol-target rows use no hand-picked window and no backtest-sized leverage.')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--reps', type=int, default=200, help='wind placebo repetitions')
    p.add_argument('--vol-target', type=float, default=0.10, help='annualised vol target')
    p.add_argument('--skip-placebo', action='store_true')
    args = p.parse_args()

    cfg = Config()
    ssb, oisst, weather = load_data(cfg, refresh=False)
    X, y, mkt = build_dataset(ssb, oisst, weather, cfg)

    section_noise(ssb, oisst, weather, cfg, X, y, mkt)
    section_env(X, y, mkt, cfg, args.reps, args.skip_placebo)
    section_nested(X, y, mkt, cfg)
    section_sizing(X, y, mkt, cfg, args.vol_target)
    print(f'\n{SEP}')


if __name__ == '__main__':
    main()
