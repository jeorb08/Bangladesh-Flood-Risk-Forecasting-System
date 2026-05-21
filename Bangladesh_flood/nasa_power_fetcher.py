"""
NASA POWER API — Rainfall & Weather Data Fetcher
Bangladesh Flood Forecasting Project

Fetches daily rainfall, temperature, humidity, and wind speed
for any location in Bangladesh using the free NASA POWER API.
No API key or account required.

Parameters fetched:
    PRECTOTCORR  — Precipitation corrected (mm/day)
    T2M          — Temperature at 2m height (°C)
    RH2M         — Relative humidity at 2m (%)
    WS10M        — Wind speed at 10m height (m/s)

Usage:
    df = fetch_nasa_power(start="2010-01-01", end="2023-12-31")
    df = fetch_nasa_power(lat=22.3475, lon=91.8123, start="2015-01-01", end="2023-12-31")  # Chittagong
"""

import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Bangladesh division coordinates ────────────────────────────────────────────
BANGLADESH_STATIONS = {
    "dhaka":       {"lat": 23.8103, "lon": 90.4125},
    "chittagong":  {"lat": 22.3475, "lon": 91.8123},
    "sylhet":      {"lat": 24.8949, "lon": 91.8687},
    "rajshahi":    {"lat": 24.3745, "lon": 88.6042},
    "khulna":      {"lat": 22.8456, "lon": 89.5403},
    "barisal":     {"lat": 22.7010, "lon": 90.3535},
    "rangpur":     {"lat": 25.7439, "lon": 89.2752},
    "mymensingh":  {"lat": 24.7471, "lon": 90.4203},
}

# NASA POWER API config
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMETERS     = "PRECTOTCORR,T2M,RH2M,WS10M"
FILL_VALUE          = -999.0   # NASA uses this for missing data


# ── Core fetch function ─────────────────────────────────────────────────────────
def fetch_nasa_power(
    lat: float = 23.8103,
    lon: float = 90.4125,
    start: str = "2000-01-01",
    end: str   = "2023-12-31",
    parameters: str = NASA_PARAMETERS,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> pd.DataFrame:
    """
    Fetch daily weather data from NASA POWER API for a single location.

    Args:
        lat         : Latitude  (default: Dhaka)
        lon         : Longitude (default: Dhaka)
        start       : Start date as 'YYYY-MM-DD'
        end         : End date   as 'YYYY-MM-DD'
        parameters  : Comma-separated NASA POWER parameter codes
        max_retries : Number of retry attempts on network error
        retry_delay : Seconds to wait between retries

    Returns:
        pd.DataFrame with a DatetimeIndex and one column per parameter.
        Missing values (-999) are replaced with NaN.

    Raises:
        ValueError  : If date format is wrong or start > end
        RuntimeError: If API returns an error after all retries
    """
    # ── Validate dates ──────────────────────────────────────────────────────
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Dates must be 'YYYY-MM-DD'. Got start='{start}', end='{end}'")

    if start_dt > end_dt:
        raise ValueError(f"start ({start}) must be before end ({end})")

    if end_dt > datetime.today():
        logger.warning("end date is in the future — NASA POWER only has data up to ~5 days ago")
        end_dt = datetime.today() - timedelta(days=5)
        end    = end_dt.strftime("%Y-%m-%d")
        logger.info(f"Adjusted end date to: {end}")

    logger.info(f"Fetching NASA POWER data | lat={lat}, lon={lon} | {start} → {end}")

    # ── Build request ───────────────────────────────────────────────────────
    params = {
        "parameters": parameters,
        "community":  "RE",                         # Renewable Energy community (best dataset)
        "longitude":  lon,
        "latitude":   lat,
        "start":      start.replace("-", ""),       # API expects YYYYMMDD
        "end":        end.replace("-", ""),
        "format":     "JSON",
    }

    # ── Fetch with retries ──────────────────────────────────────────────────
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                NASA_POWER_BASE_URL,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            break

        except requests.exceptions.Timeout:
            logger.warning(f"Attempt {attempt}/{max_retries} timed out")
            if attempt == max_retries:
                raise RuntimeError("NASA POWER API timed out after all retries")
            time.sleep(retry_delay)

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e} | Status: {response.status_code}")
            if response.status_code == 422:
                raise ValueError(
                    "NASA POWER rejected the request — check lat/lon bounds and parameter codes"
                )
            raise RuntimeError(f"NASA POWER API HTTP error: {e}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(f"NASA POWER API unreachable after {max_retries} attempts: {e}")
            time.sleep(retry_delay)

    # ── Parse response ──────────────────────────────────────────────────────
    data = response.json()

    # NASA POWER nests results under: header → properties → parameter
    try:
        param_data = data["properties"]["parameter"]
    except KeyError:
        raise RuntimeError(
            f"Unexpected NASA POWER response structure. Keys found: {list(data.keys())}"
        )

    # ── Build DataFrame ─────────────────────────────────────────────────────
    # Each parameter is a dict of {"YYYYMMDD": value}
    records = {}
    for param_code, daily_values in param_data.items():
        records[param_code] = {
            pd.Timestamp(date_str): val
            for date_str, val in daily_values.items()
        }

    df = pd.DataFrame(records)
    df.index.name = "date"
    df = df.sort_index()

    # ── Clean missing values ────────────────────────────────────────────────
    n_before = len(df)
    df.replace(FILL_VALUE, pd.NA, inplace=True)

    # ── Rename columns to human-readable names ──────────────────────────────
    rename_map = {
        "PRECTOTCORR": "rainfall_mm",
        "T2M":         "temp_c",
        "RH2M":        "humidity_pct",
        "WS10M":       "wind_speed_ms",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ── Clip negative rainfall (rare API artifact) ──────────────────────────
    if "rainfall_mm" in df.columns:
        neg_count = (df["rainfall_mm"] < 0).sum()
        if neg_count > 0:
            logger.warning(f"Clipped {neg_count} negative rainfall values to 0")
        df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0)

    logger.info(f"✓ Fetched {len(df)} days | {df.isna().sum().sum()} missing values filled with NaN")
    return df


# ── Convenience: fetch a named station ─────────────────────────────────────────
def fetch_station(
    station: str = "dhaka",
    start: str = "2000-01-01",
    end: str   = "2023-12-31",
) -> pd.DataFrame:
    """
    Fetch data for a named Bangladesh division station.

    Args:
        station: One of dhaka, chittagong, sylhet, rajshahi,
                 khulna, barisal, rangpur, mymensingh
        start:   Start date 'YYYY-MM-DD'
        end:     End date   'YYYY-MM-DD'

    Returns:
        pd.DataFrame — same as fetch_nasa_power()
    """
    station = station.lower().strip()
    if station not in BANGLADESH_STATIONS:
        raise ValueError(
            f"Unknown station '{station}'. "
            f"Choose from: {', '.join(BANGLADESH_STATIONS.keys())}"
        )
    coords = BANGLADESH_STATIONS[station]
    logger.info(f"Station: {station.title()} | lat={coords['lat']}, lon={coords['lon']}")
    df = fetch_nasa_power(lat=coords["lat"], lon=coords["lon"], start=start, end=end)
    df["station"] = station
    return df


# ── Fetch all Bangladesh divisions at once ──────────────────────────────────────
def fetch_all_stations(
    start: str = "2010-01-01",
    end: str   = "2023-12-31",
    pause_seconds: int = 2,
) -> pd.DataFrame:
    """
    Fetch data for all 8 Bangladesh division stations and return
    a combined long-format DataFrame with a 'station' column.

    Args:
        start          : Start date 'YYYY-MM-DD'
        end            : End date   'YYYY-MM-DD'
        pause_seconds  : Polite delay between API calls (avoid rate limiting)

    Returns:
        pd.DataFrame with columns: date (index), rainfall_mm, temp_c,
        humidity_pct, wind_speed_ms, station
    """
    all_dfs = []
    stations = list(BANGLADESH_STATIONS.keys())

    for i, station in enumerate(stations, 1):
        logger.info(f"[{i}/{len(stations)}] Fetching {station.title()}...")
        try:
            df = fetch_station(station, start=start, end=end)
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch {station}: {e} — skipping")

        if i < len(stations):
            time.sleep(pause_seconds)   # be polite to the free API

    if not all_dfs:
        raise RuntimeError("All station fetches failed — check internet connection")

    combined = pd.concat(all_dfs)
    combined = combined.reset_index().set_index(["date", "station"]).sort_index()
    logger.info(f"✓ Combined dataset: {len(combined)} rows across {len(all_dfs)} stations")
    return combined


# ── Feature engineering helpers ─────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame, lags: list[int] = [1, 3, 7, 14]) -> pd.DataFrame:
    """
    Add lag and rolling window features to a single-station DataFrame.
    Call this after fetching data for one station (not the combined multi-station df).

    Features added:
        rain_lag_{n}d    — Rainfall n days ago
        rain_roll_{n}d   — Rolling mean rainfall over n days
        rain_std_{n}d    — Rolling std rainfall over n days (monsoon onset signal)
        is_monsoon       — 1 if month is June–October (main monsoon season)
        day_of_year_sin  — Sine encoding of day-of-year (cyclical)
        day_of_year_cos  — Cosine encoding of day-of-year (cyclical)
    """
    import numpy as np

    df = df.copy()

    if "rainfall_mm" not in df.columns:
        raise ValueError("DataFrame must have a 'rainfall_mm' column")

    rain = df["rainfall_mm"]

    # Lag features
    for lag in lags:
        df[f"rain_lag_{lag}d"] = rain.shift(lag)

    # Rolling statistics
    for window in [3, 7, 30]:
        df[f"rain_roll_{window}d"] = rain.shift(1).rolling(window).mean()
        df[f"rain_std_{window}d"]  = rain.shift(1).rolling(window).std()

    # Days since last heavy rain (>50mm) — soil saturation proxy
    heavy_rain = (rain > 50).astype(int)
    df["days_since_heavy_rain"] = (
        heavy_rain
        .groupby((heavy_rain != heavy_rain.shift()).cumsum())
        .cumcount()
    )

    # Calendar features
    df["month"]           = df.index.month
    df["day_of_year"]     = df.index.dayofyear
    df["is_monsoon"]      = df["month"].isin([6, 7, 8, 9, 10]).astype(int)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    logger.info(f"Added lag + calendar features → {df.shape[1]} total columns")
    return df


# ── Quick summary ────────────────────────────────────────────────────────────────
def rainfall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a monthly rainfall summary. Useful for EDA and README tables.

    Args:
        df: Single-station DataFrame with 'rainfall_mm' column

    Returns:
        pd.DataFrame — monthly stats (mean, max, total, missing)
    """
    df = df.copy()
    df["month"] = df.index.month
    df["year"]  = df.index.year

    summary = (
        df.groupby("month")["rainfall_mm"]
        .agg(
            mean_daily_mm="mean",
            max_daily_mm="max",
            avg_monthly_total=lambda x: x.resample("ME").sum().mean(),
            missing_days=lambda x: x.isna().sum(),
        )
        .round(2)
    )

    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    summary.index = month_names
    return summary


# ── Example usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Fetch Dhaka (default)
    print("\n── Fetching Dhaka rainfall data ──")
    dhaka_df = fetch_station("dhaka", start="2015-01-01", end="2023-12-31")
    print(dhaka_df.head(10).to_string())

    # 2. Monthly summary
    print("\n── Monthly rainfall summary ──")
    print(rainfall_summary(dhaka_df).to_string())

    # 3. Add features for modelling
    print("\n── Adding lag features ──")
    dhaka_features = add_lag_features(dhaka_df)
    print(f"Feature columns: {list(dhaka_features.columns)}")
    print(dhaka_features.head(20).to_string())

    # 4. Save to CSV
    # dhaka_features.to_csv("dhaka_rainfall_features.csv")
    # print("\n✓ Saved to dhaka_rainfall_features.csv")

    # 5. Fetch all stations (takes ~30 seconds, polite delay between calls)
    all_df = fetch_all_stations(start="2015-01-01", end="2023-12-31")
    all_df.to_csv("bangladesh_all_stations.csv")
    print(all_df.groupby("station")["rainfall_mm"].describe())
