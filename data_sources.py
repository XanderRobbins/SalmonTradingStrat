"""
Fetchers for the public datasets the strategy uses. No API keys needed.

  - SSB table 03024: weekly Norwegian export price (NOK/kg) and volume of fresh
    farmed salmon, 2000-present. This is the spot series salmon futures
    settle toward.
  - Daily max wind and solar radiation at each farming region, from ERA5
    (via Open-Meteo, free for non-commercial use) or NASA POWER.
  - NOAA OISST v2.1: weekly 0.25 deg sea surface temperature at each region.

Everything is cached under data/raw; pass refresh=True to re-download.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import EnvConfig, FarmRegion

SSB_URL = 'https://data.ssb.no/api/pxwebapi/v2/tables/03024/data'
POWER_URL = 'https://power.larc.nasa.gov/api/temporal/daily/point'
ERA5_URL = 'https://archive-api.open-meteo.com/v1/archive'
OISST_URL = 'https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.week.mean.nc'
OISST_EPOCH = pd.Timestamp('1800-01-01')
OISST_MISSING_ABOVE = 1e30  # land cells are flagged with -9.97e36


def _get(url: str, params: dict | None = None, timeout: int = 300, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:  # Open-Meteo's per-minute limit
                time.sleep(61)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'Still rate limited by {url} after {retries} attempts')


def _iso_week_end(code: str) -> pd.Timestamp:
    """SSB period code like '2026U38' -> Sunday ending that ISO week."""
    return pd.Timestamp(date.fromisocalendar(int(code[:4]), int(code[5:]), 7))


# =============================================================================
# SSB - salmon export price and volume
# =============================================================================

def fetch_ssb_salmon(raw_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Weekly price (NOK/kg) and volume (tonnes), indexed by ISO week-ending Sunday."""
    path = raw_dir / 'ssb_03024.csv'
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col='week_end', parse_dates=['week_end'])

    params = {
        'lang': 'en',
        'outputFormat': 'json-stat2',
        'valueCodes[VareGrupper2]': '01',  # fresh or chilled farmed salmon
        'valueCodes[ContentsCode]': 'Kilopris,Vekt',
        'valueCodes[Tid]': '*',
    }
    js = _get(SSB_URL, params).json()
    if js['id'] != ['VareGrupper2', 'ContentsCode', 'Tid']:
        raise ValueError(f"Unexpected SSB dimension order: {js['id']}")

    values = np.array([np.nan if v is None else v for v in js['value']], dtype=float)
    values = values.reshape(js['size'])[0]
    contents = js['dimension']['ContentsCode']['category']['index']
    weeks = js['dimension']['Tid']['category']['index']
    week_codes = sorted(weeks, key=weeks.get)

    df = pd.DataFrame({
        'week_end': [_iso_week_end(c) for c in week_codes],
        'price_nok_kg': values[contents['Kilopris']],
        'volume_t': values[contents['Vekt']],
    }).set_index('week_end')

    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


# =============================================================================
# Weather - daily max wind (m/s) and solar radiation (kWh/m2/day) per region
# =============================================================================

POWER_VARS = {'WS10M_MAX': 'wind_max_ms', 'ALLSKY_SFC_SW_DWN': 'solar_kwh_m2'}
ERA5_VARS = {'wind_speed_10m_max': 'wind_max_ms', 'shortwave_radiation_sum': 'solar_kwh_m2'}


def _fetch_power_region(region: FarmRegion, cfg: EnvConfig, raw_dir: Path, refresh: bool) -> pd.DataFrame:
    path = raw_dir / f'power_{region.key}.csv'
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col='date', parse_dates=['date'])

    params = {
        'parameters': ','.join(POWER_VARS),
        'community': 'AG',
        'latitude': region.lat,
        'longitude': region.lon,
        'start': pd.Timestamp(cfg.weather_start).strftime('%Y%m%d'),
        'end': date.today().strftime('%Y%m%d'),
        'format': 'JSON',
    }
    js = _get(POWER_URL, params).json()
    fill = js['header']['fill_value']
    df = pd.DataFrame(js['properties']['parameter']).rename(columns=POWER_VARS)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df.index.name = 'date'
    df = df.replace(fill, np.nan)

    df.to_csv(path)
    return df


def fetch_power(regions: tuple[FarmRegion, ...], cfg: EnvConfig, raw_dir: Path,
                refresh: bool = False) -> dict[str, pd.DataFrame]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = pool.map(lambda r: _fetch_power_region(r, cfg, raw_dir, refresh), regions)
    return {r.key: df for r, df in zip(regions, frames)}


def _fetch_era5_region(region: FarmRegion, cfg: EnvConfig, raw_dir: Path, refresh: bool) -> pd.DataFrame:
    path = raw_dir / f'era5_{region.key}.csv'
    cached = pd.read_csv(path, index_col='date', parse_dates=['date']) if path.exists() else None
    if cached is not None and not refresh:
        return cached

    # A refresh only re-pulls the last 90 days: preliminary ERA5T values are
    # finalised within ~3 months, and multi-decade pulls exhaust Open-Meteo's
    # hourly quota after roughly one full download of all regions.
    start = pd.Timestamp(cfg.weather_start)
    if cached is not None:
        start = cached.dropna().index[-1] - pd.Timedelta(days=90)

    params = {
        'latitude': region.lat,
        'longitude': region.lon,
        'start_date': start.date().isoformat(),
        'end_date': date.today().isoformat(),
        'daily': ','.join(ERA5_VARS),
        'wind_speed_unit': 'ms',
        'models': 'era5',  # pure reanalysis; 'best_match' blends in forecast-model analyses
        'timezone': 'GMT',
    }
    daily = _get(ERA5_URL, params).json()['daily']
    df = pd.DataFrame(daily).rename(columns=ERA5_VARS)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop('time')), name='date')
    df['solar_kwh_m2'] /= 3.6  # MJ/m2/day -> kWh/m2/day, same unit as POWER
    if cached is not None:
        df = pd.concat([cached.loc[:start - pd.Timedelta(days=1)], df])

    df.to_csv(path)
    return df


def fetch_era5(regions: tuple[FarmRegion, ...], cfg: EnvConfig, raw_dir: Path,
               refresh: bool = False) -> dict[str, pd.DataFrame]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Sequential: Open-Meteo weights multi-decade requests heavily against its per-minute limit.
    return {r.key: _fetch_era5_region(r, cfg, raw_dir, refresh) for r in regions}


# =============================================================================
# NOAA OISST - weekly sea surface temperature at each farming region
# =============================================================================

def _parse_opendap_ascii(text: str) -> list[tuple[tuple[int, ...], list[float]]]:
    """Data rows of an OPeNDAP .ascii grid response as (index, values) pairs."""
    body = text.split('-' * 45, 1)[1]
    rows = []
    for line in body.splitlines():
        m = re.match(r'^((?:\[\d+\])+), (.+)$', line)
        if m:
            idx = tuple(int(i) for i in re.findall(r'\d+', m.group(1)))
            rows.append((idx, [float(v) for v in m.group(2).split(',')]))
    return rows


def _oisst_time_axis() -> pd.DatetimeIndex:
    text = _get(OISST_URL + '.ascii?time').text
    body = text.split('-' * 45, 1)[1].strip().splitlines()
    days = [float(v) for v in body[1].split(',')]
    return pd.DatetimeIndex(OISST_EPOCH + pd.to_timedelta(days, unit='D'))


def _nearest_ocean_cell(region: FarmRegion, search: int = 4) -> tuple[int, int, float, float]:
    """Grid indices of the closest non-land OISST cell to the region's point."""
    ilat = int(round((region.lat + 89.875) / 0.25))
    ilon = int(round((region.lon % 360 - 0.125) / 0.25)) % 1440
    lat0, lat1 = max(ilat - search, 0), min(ilat + search, 719)
    lon0, lon1 = max(ilon - search, 0), min(ilon + search, 1439)

    query = f'sst[0][{lat0}:1:{lat1}][{lon0}:1:{lon1}]'
    rows = _parse_opendap_ascii(_get(OISST_URL + '.ascii?' + query).text)

    best, best_dist = None, np.inf
    coslat = np.cos(np.radians(region.lat))
    for (_, i), vals in rows:
        for j, v in enumerate(vals):
            if abs(v) > OISST_MISSING_ABOVE:
                continue
            gi, gj = lat0 + i, lon0 + j
            dist = (gi - ilat) ** 2 + ((gj - ilon) * coslat) ** 2
            if dist < best_dist:
                best, best_dist = (gi, gj), dist
    if best is None:
        raise ValueError(f'No ocean cell within {search} cells of {region.name}')

    gi, gj = best
    grid_lat = -89.875 + 0.25 * gi
    grid_lon = 0.125 + 0.25 * gj
    return gi, gj, grid_lat, (grid_lon - 360 if grid_lon > 180 else grid_lon)


def _fetch_oisst_region(region: FarmRegion, times: pd.DatetimeIndex, raw_dir: Path,
                        refresh: bool) -> pd.DataFrame:
    path = raw_dir / f'oisst_{region.key}.csv'
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col='week_start', parse_dates=['week_start'])

    gi, gj, grid_lat, grid_lon = _nearest_ocean_cell(region)
    query = f'sst[0:1:{len(times) - 1}][{gi}][{gj}]'
    rows = _parse_opendap_ascii(_get(OISST_URL + '.ascii?' + query).text)
    sst = np.full(len(times), np.nan)
    for (t, _), vals in rows:
        if abs(vals[0]) < OISST_MISSING_ABOVE:
            sst[t] = vals[0]

    df = pd.DataFrame({'sst': sst, 'grid_lat': grid_lat, 'grid_lon': grid_lon},
                      index=pd.Index(times, name='week_start'))
    df.to_csv(path)
    return df


def fetch_oisst(regions: tuple[FarmRegion, ...], raw_dir: Path,
                refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Weekly mean SST per region. Index is the Sunday the 7-day mean starts."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [r for r in regions if refresh or not (raw_dir / f'oisst_{r.key}.csv').exists()]
    times = _oisst_time_axis() if missing else None
    with ThreadPoolExecutor(max_workers=6) as pool:
        frames = pool.map(lambda r: _fetch_oisst_region(r, times, raw_dir, refresh), regions)
    return {r.key: df for r, df in zip(regions, frames)}
