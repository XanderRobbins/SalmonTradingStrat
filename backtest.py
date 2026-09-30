"""
Walk-forward model fitting and the futures-proxy backtest.

Salmon futures already price the normal seasonal cycle (spring highs, autumn
lows), so predicting the cycle earns nothing. The strategy is scored only on
the seasonally adjusted spot return: the actual weekly log change minus the
change the last few years' seasonality said to expect.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from config import StrategyConfig


def make_model(cfg: StrategyConfig):
    if cfg.model == 'ridge':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                             Ridge(alpha=cfg.ridge_alpha))
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300,
                                         min_samples_leaf=40, l2_regularization=1.0)


def walk_forward(X: pd.DataFrame, y: pd.Series, cfg: StrategyConfig) -> pd.Series:
    """Out-of-sample predictions, refit every `retrain_every` weeks on labels already known."""
    label_lag = cfg.horizon_weeks + cfg.publication_lag_weeks
    start = X.index.searchsorted(pd.Timestamp(cfg.test_start))
    pred = pd.Series(np.nan, index=X.index, name='pred')
    for i in range(start, len(X), cfg.retrain_every):
        # Row t's label is realised at row t + label_lag, so only t <= i - label_lag is usable.
        train = y.iloc[: i - label_lag + 1].dropna()
        model = make_model(cfg).fit(X.loc[train.index], train)
        block = slice(i, i + cfg.retrain_every)
        pred.iloc[block] = model.predict(X.iloc[block])
    return pred


def fit_final(X: pd.DataFrame, y: pd.Series, cfg: StrategyConfig):
    train = y.dropna()
    return make_model(cfg).fit(X.loc[train.index], train)


def signal_tranches(pred: pd.Series, cfg: StrategyConfig) -> pd.Series:
    """Each week opens a tranche sized by the prediction's z-score against past predictions."""
    scale = pred.shift(1).rolling(cfg.signal_vol_window, min_periods=26).std()
    return (pred / scale / cfg.z_cap).clip(-1, 1).fillna(0.0)


def raw_book(tranche: pd.Series, cfg: StrategyConfig) -> pd.Series:
    """
    Overlapping-tranche book: the signal from week t is held for `horizon_weeks`
    starting after publication, so the position earning week k's return is the
    average of the tranches opened at weeks k-lag-h+1 .. k-lag.
    """
    return (tranche.rolling(cfg.horizon_weeks, min_periods=1).mean()
            .shift(cfg.publication_lag_weeks + 1))


def vol_cap_scale(raw_position: pd.Series, adj_ret: pd.Series, cfg: StrategyConfig) -> pd.Series:
    """
    Shrinks the book when its recent P&L volatility runs above its own typical
    level; never scales up. The strategy's rough patches (2012, 2022) came with
    P&L vol spikes, while market vol alone is too stable to flag them.

    The result extends `publication_lag + 1` weeks past the data so its last
    value is the scale for the position about to be put on.
    """
    if cfg.vol_cap_halflife is None:
        return pd.Series(1.0, index=raw_position.index)

    # Leave out the signal warm-up: near-zero positions there would drag the
    # "typical" level down and make the cap far too tight for years.
    pnl = (raw_position * adj_ret).where(raw_position.abs().cummax() > 0.05)
    vol = pnl.ewm(halflife=cfg.vol_cap_halflife, min_periods=13).std()

    decision_lag = cfg.publication_lag_weeks + 1  # P&L through week k-2 sets week k
    ahead = pd.date_range(vol.index[-1], periods=decision_lag + 1, freq='W-SUN')[1:]
    vol = vol.reindex(vol.index.append(ahead)).shift(decision_lag)
    typical = vol.expanding(min_periods=52).median()
    return (typical / vol).clip(upper=1).fillna(1.0)


def no_trade_mask(index: pd.DatetimeIndex, cfg: StrategyConfig) -> np.ndarray:
    """True for weeks the book must be flat: blackout windows and off-season months."""
    mask = ~index.month.isin(cfg.trade_months)
    for start, end in cfg.no_trade_windows:
        mask |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return mask


def run_backtest(pred: pd.Series, adj_ret: pd.Series, cfg: StrategyConfig) -> pd.DataFrame:
    tranche = signal_tranches(pred, cfg)
    raw_position = raw_book(tranche, cfg)
    scale = vol_cap_scale(raw_position, adj_ret, cfg).reindex(raw_position.index)
    position = raw_position * scale * cfg.leverage
    position = position.mask(no_trade_mask(position.index, cfg), 0.0)
    gross = position * adj_ret
    cost = position.diff().abs() * cfg.cost_bps / 1e4
    bt = pd.DataFrame({
        'pred': pred,
        'tranche': tranche,
        'raw_position': raw_position,
        'vol_scale': scale,
        'position': position,
        'adj_ret': adj_ret,
        'gross': gross,
        'cost': cost,
        'net': gross - cost,
    })
    bt = bt.loc[pred.first_valid_index():].dropna(subset=['position', 'adj_ret'])
    bt['equity'] = bt['net'].cumsum()
    return bt


def performance(bt: pd.DataFrame, pred: pd.Series, y: pd.Series, horizon: int) -> dict:
    scored = pd.concat([pred, y], axis=1, keys=['pred', 'y']).dropna()
    ic = scored['pred'].corr(scored['y'])
    rank_ic = scored['pred'].rank().corr(scored['y'].rank())
    # Overlapping h-week labels: roughly n/h independent observations.
    n_eff = len(scored) / horizon
    ic_t = ic * np.sqrt(n_eff - 2) / np.sqrt(1 - ic ** 2)
    r2_vs_zero = 1 - ((scored['y'] - scored['pred']) ** 2).sum() / (scored['y'] ** 2).sum()

    net = bt['net']
    eq = bt['equity']
    holding = bt['position'].abs() > 1e-9
    # Weeks spent flat (warm-up, blackouts, off-season) count as zero in the
    # calendar Sharpe; the active Sharpe drops them. Exit weeks stay in because
    # they carry the closing cost.
    active_net = net[holding | (bt['cost'] > 0)]
    return {
        'IC': ic,
        'rank_IC': rank_ic,
        'IC_t_eff': ic_t,
        'dir_hit': (np.sign(scored['pred']) == np.sign(scored['y'])).mean(),
        'R2_vs_0': r2_vs_zero,
        'ann_ret': net.mean() * 52,
        'ann_vol': net.std() * np.sqrt(52),
        'sharpe': net.mean() / net.std() * np.sqrt(52),
        'sharpe_act': active_net.mean() / active_net.std() * np.sqrt(52),
        'in_mkt': len(active_net) / len(net),
        'max_dd': (eq - eq.cummax()).min(),
        'week_hit': (np.sign(bt.loc[holding, 'position']) == np.sign(bt.loc[holding, 'adj_ret'])).mean(),
        'avg_abs_pos': bt['position'].abs().mean(),
        'turnover_yr': bt['position'].diff().abs().sum() / len(bt) * 52,
    }


def ridge_contributions(model: Pipeline, x_row: pd.DataFrame) -> pd.DataFrame:
    """Standardised coefficients and each feature's contribution to the latest prediction."""
    imputer, scaler, ridge = model.named_steps.values()
    z = scaler.transform(imputer.transform(x_row))[0]
    return pd.DataFrame({
        'coef': ridge.coef_,
        'z_now': z,
        'contribution': ridge.coef_ * z,
    }, index=x_row.columns).sort_values('contribution', key=np.abs, ascending=False)
