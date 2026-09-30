"""
Configuration for the salmon environmental-signal strategy.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FarmRegion:
    """A salmon farming area, sampled at one near-shore ocean point."""
    key: str
    name: str
    group: Literal['NO', 'CL', 'OTH']
    lat: float
    lon: float
    weight: float  # approx. % of global farmed Atlantic salmon harvest


# Approximate harvest shares (~2023-24). Only relative weights inside a group
# matter, since each group becomes its own set of features. Norway is split
# into the 13 official production areas (produksjonsområder).
FARM_REGIONS: tuple[FarmRegion, ...] = (
    FarmRegion('no_po01', 'PO1 Svenskegrensen-Jæren', 'NO', 58.30, 6.40, 0.5),
    FarmRegion('no_po02', 'PO2 Ryfylke', 'NO', 59.15, 5.55, 3.0),
    FarmRegion('no_po03', 'PO3 Karmøy-Sotra', 'NO', 59.90, 5.05, 6.0),
    FarmRegion('no_po04', 'PO4 Nordhordland-Stadt', 'NO', 61.20, 4.80, 7.0),
    FarmRegion('no_po05', 'PO5 Stadt-Hustadvika', 'NO', 62.60, 6.00, 2.5),
    FarmRegion('no_po06', 'PO6 Nordmøre-Sør-Trøndelag', 'NO', 63.70, 8.50, 8.0),
    FarmRegion('no_po07', 'PO7 Nord-Trøndelag', 'NO', 64.80, 10.80, 3.5),
    FarmRegion('no_po08', 'PO8 Helgeland-Bodø', 'NO', 66.30, 12.60, 6.5),
    FarmRegion('no_po09', 'PO9 Vestfjorden-Vesterålen', 'NO', 68.30, 14.50, 5.0),
    FarmRegion('no_po10', 'PO10 Andøya-Senja', 'NO', 69.30, 17.00, 4.5),
    FarmRegion('no_po11', 'PO11 Kvaløya-Loppa', 'NO', 70.10, 19.50, 3.0),
    FarmRegion('no_po12', 'PO12 Vest-Finnmark', 'NO', 70.70, 23.00, 2.0),
    FarmRegion('no_po13', 'PO13 Øst-Finnmark', 'NO', 70.80, 28.00, 0.5),
    FarmRegion('cl_loslagos', 'Chile Los Lagos', 'CL', -42.00, -73.00, 11.0),
    FarmRegion('cl_aysen', 'Chile Aysén', 'CL', -45.30, -73.80, 12.0),
    FarmRegion('cl_magallanes', 'Chile Magallanes', 'CL', -52.80, -72.00, 4.0),
    FarmRegion('uk_westscot', 'Scotland West Coast', 'OTH', 57.30, -6.00, 5.0),
    FarmRegion('uk_shetland', 'Shetland/Orkney', 'OTH', 60.30, -1.20, 1.5),
    FarmRegion('fo_faroes', 'Faroe Islands', 'OTH', 62.00, -6.80, 3.5),
    FarmRegion('is_westfjords', 'Iceland Westfjords', 'OTH', 65.70, -23.50, 2.0),
    FarmRegion('ca_fundy', 'Canada Bay of Fundy', 'OTH', 44.90, -66.80, 1.5),
    FarmRegion('ca_bc', 'Canada British Columbia', 'OTH', 50.20, -125.50, 2.5),
)


@dataclass
class EnvConfig:
    """Environmental data and stress thresholds."""
    # 'era5' (ECMWF reanalysis via Open-Meteo, ~6 day latency) or 'power' (NASA
    # POWER). POWER splices its near-real-time GEOS-IT stream onto MERRA-2 for
    # roughly the last 1-2 months, and the spliced wind is visibly biased (Sep
    # 2026 Chile max wind drops ~2-7 m/s overnight), so POWER features are
    # lagged until they're MERRA-2 only - which also removes most of their value.
    weather_source: Literal['era5', 'power'] = 'era5'
    power_lag_weeks: int = 8
    weather_start: str = '1990-01-01'

    # Atlantic salmon grow best around 12-14 C; above ~16 C appetite drops and
    # lice/algae/low-oxygen mortality rises.
    hot_sst_c: float = 16.0

    # Rolling (not expanding) baseline: coastal SST has warmed ~0.5 C since the
    # 1980s, and an expanding mean turns that trend into a fake "anomaly".
    climatology_years: int = 10
    climatology_min_years: int = 3


@dataclass
class StrategyConfig:
    """Model, signal and backtest parameters."""
    horizon_weeks: int = 4          # roughly the front-month futures tenor
    publication_lag_weeks: int = 1  # SSB publishes week t on Wednesday of week t+1
    seasonal_years: int = 8         # years of history the market uses to price seasonality

    model: Literal['ridge', 'gbm'] = 'ridge'
    ridge_alpha: float = 200.0
    test_start: str = '2010-01-01'
    retrain_every: int = 13         # weeks

    signal_vol_window: int = 104    # weeks of past predictions used to scale the signal
    z_cap: float = 2.0              # |z| at which a tranche is fully sized
    vol_cap_halflife: int | None = 13  # weeks; None turns the P&L vol cap off
    # The vol cap cut 2010-26 max drawdown from 29% to 22%; 1.3x brings it back
    # to ~28%. Set from the backtest's own max DD, so future ones can be deeper.
    leverage: float = 1.3
    cost_bps: float = 25.0          # per unit of position change; salmon futures are thin

    # Forced flat over these inclusive (start, end) date ranges, e.g.
    # (('2020-03-02', '2020-06-28'),). Empty: a hand-picked window uses hindsight.
    no_trade_windows: tuple[tuple[str, str], ...] = ()
    # Months (of the week-ending date) the book may hold positions. The model is
    # sharpest Apr-Sep, but Oct-Mar still adds return, so the default is all year.
    trade_months: tuple[int, ...] = tuple(range(1, 13))


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    regions: tuple[FarmRegion, ...] = FARM_REGIONS
    raw_dir: Path = ROOT / 'data' / 'raw'
    results_dir: Path = ROOT / 'results'
