"""
Weekly feature panel and prediction target.

Every row is an ISO week (indexed by its Sunday). A feature at week t only uses
data available when SSB publishes week t's price, on Wednesday of week t+1.
"""
import numpy as np
import pandas as pd

from config import Config, EnvConfig, FarmRegion, StrategyConfig

MARKET_FEATURES = ['mom_1', 'mom_4', 'mom_13', 'adj_mom_4', 'dev_52']
SUPPLY_FEATURES = ['vol_yoy_4', 'vol_yoy_13']
GROUPS = ('NO', 'CL', 'OTH')
ENV_SUFFIXES = ('sst_a4', 'sst_a26', 'sst_a52', 'hot_8', 'wind_a2', 'rad_a4')
ENV_FEATURES = [f'{g}_{s}' for g in GROUPS for s in ENV_SUFFIXES]

_GROUP_NAMES = {'NO': 'Norway', 'CL': 'Chile', 'OTH': 'UK/Faroes/Iceland/Canada'}
_SUFFIX_NAMES = {
    'sst_a4': 'sea temp anomaly, 4w',
    'sst_a26': 'sea temp anomaly, 26w',
    'sst_a52': 'sea temp anomaly, 52w',
    'hot_8': 'heat-stress weeks (>16C) vs normal, 8w',
    'wind_a2': 'max wind anomaly, 2w',
    'rad_a4': 'solar radiation anomaly, 4w',
}
FEATURE_LABELS = {
    'mom_1': 'Price change, 1w',
    'mom_4': 'Price change, 4w',
    'mom_13': 'Price change, 13w',
    'adj_mom_4': 'Seasonally adjusted price change, 4w',
    'dev_52': 'Price vs 52w average',
    'vol_yoy_4': 'Export volume YoY, 4w',
    'vol_yoy_13': 'Export volume YoY, 13w',
    **{f'{g}_{s}': f'{_GROUP_NAMES[g]} {_SUFFIX_NAMES[s]}' for g in GROUPS for s in ENV_SUFFIXES},
}


def _week_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    # Week 53 happens every 5-6 years; fold it into 52 so it has a history.
    return index.isocalendar().week.clip(upper=52).to_numpy()


def seasonal_norm(s: pd.Series, min_years: int, window_years: int | None = None) -> pd.Series:
    """Mean of the same ISO week over prior years only, so there is no lookahead."""
    def prior_mean(g: pd.Series) -> pd.Series:
        prior = g.shift(1)
        if window_years is None:
            return prior.expanding(min_periods=min_years).mean()
        return prior.rolling(window_years, min_periods=min_years).mean()

    return s.groupby(_week_of_year(s.index)).transform(prior_mean)


def _roll_mean(s: pd.Series, weeks: int) -> pd.Series:
    # Half-window minimum so one late NASA/NOAA week doesn't blank the feature.
    return s.rolling(weeks, min_periods=max(1, weeks // 2)).mean()


# =============================================================================
# ENVIRONMENT
# =============================================================================

def _weighted_mean(frame: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights[frame.columns]
    covered = frame.notna().mul(w, axis=1).sum(axis=1)
    return frame.fillna(0).mul(w, axis=1).sum(axis=1) / covered.replace(0, np.nan)


def weekly_env_panel(oisst: dict[str, pd.DataFrame], weather: dict[str, pd.DataFrame],
                     regions: tuple[FarmRegion, ...], cfg: EnvConfig) -> pd.DataFrame:
    """Production-weighted weekly SST / heat stress / wind / radiation per region group."""
    weather_lag = cfg.power_lag_weeks if cfg.weather_source == 'power' else 0
    per_var: dict[str, dict[str, pd.Series]] = {v: {} for v in ('sst', 'hot', 'wind', 'rad')}
    for r in regions:
        # OISST stamps the Sunday a 7-day mean starts; it is complete a week later.
        sst = oisst[r.key]['sst'].copy()
        sst.index = sst.index + pd.Timedelta(days=7)

        weekly = weather[r.key].resample('W-SUN')
        n_days = weekly.count()
        per_var['sst'][r.key] = sst
        per_var['hot'][r.key] = (sst >= cfg.hot_sst_c).astype(float).where(sst.notna())
        # Farm points sit in sheltered fjords where absolute wind is damped, so
        # storminess is measured as an anomaly rather than against a gale threshold.
        wind = weekly['wind_max_ms'].max().where(n_days['wind_max_ms'] >= 5)
        rad = weekly['solar_kwh_m2'].mean().where(n_days['solar_kwh_m2'] >= 5)
        per_var['wind'][r.key] = wind.shift(weather_lag)
        per_var['rad'][r.key] = rad.shift(weather_lag)

    weights = pd.Series({r.key: r.weight for r in regions})
    panel = {}
    for g in GROUPS:
        keys = [r.key for r in regions if r.group == g]
        for var, series in per_var.items():
            frame = pd.DataFrame({k: series[k] for k in keys})
            panel[f'{g}_{var}'] = _weighted_mean(frame, weights)
    panel = pd.DataFrame(panel)
    return panel.asfreq('W-SUN')


def env_features(panel: pd.DataFrame, cfg: EnvConfig) -> pd.DataFrame:
    def anom(col: str) -> pd.Series:
        s = panel[col]
        return s - seasonal_norm(s, cfg.climatology_min_years, cfg.climatology_years)

    out = {}
    for g in GROUPS:
        sst_a = anom(f'{g}_sst')
        # 4w = current conditions; 26w/52w = growth over the cohort now being
        # harvested (warm year -> fish reach harvest weight early -> more supply).
        out[f'{g}_sst_a4'] = _roll_mean(sst_a, 4)
        out[f'{g}_sst_a26'] = _roll_mean(sst_a, 26)
        out[f'{g}_sst_a52'] = _roll_mean(sst_a, 52)
        out[f'{g}_hot_8'] = _roll_mean(anom(f'{g}_hot'), 8)
        out[f'{g}_wind_a2'] = _roll_mean(anom(f'{g}_wind'), 2)
        # Sunny, warm spells are the precondition for harmful algal blooms.
        out[f'{g}_rad_a4'] = _roll_mean(anom(f'{g}_rad'), 4)
    return pd.DataFrame(out)


# =============================================================================
# MARKET
# =============================================================================

def market_frame(ssb: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Price, weekly log return, the seasonal return the market expects, and the surprise."""
    logp = np.log(ssb['price_nok_kg'])
    ret = logp.diff()
    # Smooth across neighbouring weeks first: Easter and Christmas move the
    # seasonal peaks around by a week from year to year.
    seas = seasonal_norm(ret.rolling(3, center=True).mean(), min_years=3,
                         window_years=cfg.seasonal_years)
    return pd.DataFrame({
        'price': ssb['price_nok_kg'],
        'volume': ssb['volume_t'],
        'ret': ret,
        'seas_ret': seas,
        'adj_ret': ret - seas,
    })


def market_features(mkt: pd.DataFrame) -> pd.DataFrame:
    logp = np.log(mkt['price'])
    vol4 = np.log(mkt['volume'].rolling(4).sum())
    vol13 = np.log(mkt['volume'].rolling(13).sum())
    return pd.DataFrame({
        'mom_1': mkt['ret'],
        'mom_4': logp.diff(4),
        'mom_13': logp.diff(13),
        'adj_mom_4': mkt['adj_ret'].rolling(4).sum(),
        'dev_52': logp - logp.rolling(52).mean(),
        'vol_yoy_4': vol4 - vol4.shift(52),
        'vol_yoy_13': vol13 - vol13.shift(52),
    })


# =============================================================================
# DATASET
# =============================================================================

def build_dataset(ssb: pd.DataFrame, oisst: dict[str, pd.DataFrame],
                  weather: dict[str, pd.DataFrame], cfg: Config
                  ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns (X, y, mkt).

    y[t] is the sum of seasonally adjusted weekly returns a trade placed after
    week t's publication would earn over the next `horizon_weeks` weeks.
    """
    s = cfg.strategy
    mkt = market_frame(ssb, s)
    env = env_features(weekly_env_panel(oisst, weather, cfg.regions, cfg.env), cfg.env)

    X = market_features(mkt).join(env.reindex(mkt.index))
    X = X.loc[X.dropna().index[0]:]

    lag = s.horizon_weeks + s.publication_lag_weeks
    y = mkt['adj_ret'].rolling(s.horizon_weeks).sum().shift(-lag).reindex(X.index)
    y.name = 'target'
    return X, y, mkt.loc[X.index[0]:]
