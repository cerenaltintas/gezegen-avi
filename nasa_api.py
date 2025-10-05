"""NASA Exoplanet Archive data access utilities."""

from __future__ import annotations

import os
import time
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

import io

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

MISSION_TABLE_MAP: dict[str, str] = {
    "kepler": "cumulative",  # Ana cumulative tablo - tüm fiziksel parametreleri içerir
}

BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CACHE_DIR = Path(os.getenv("EXOPLANET_CACHE", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


class NASAExoplanetAPIError(RuntimeError):
    """Raised when the NASA Exoplanet Archive request fails."""


def _build_query(mission: Literal["kepler"], limit: int) -> str:
    if mission != "kepler":
        raise ValueError("Only the Kepler mission is currently supported.")

    table = MISSION_TABLE_MAP[mission]
    columns = (
        "koi_disposition, koi_period, koi_depth, koi_duration, koi_prad, "
        "koi_srad, koi_steff, koi_slogg, koi_teq, koi_insol, koi_ror, koi_srho, "
        "koi_model_snr, koi_impact, koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec"
    )
    where_clause = "WHERE koi_disposition IS NOT NULL"
    query = (
        f"SELECT TOP {limit} {columns} FROM {table} {where_clause} "
        "ORDER BY koi_score DESC"
    )
    return query


def _get_http_session(max_retries: int = 5, backoff: float = 0.6) -> requests.Session:
    """Create a requests session with retry and backoff for resilient downloads."""
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "ExoplanetAI/1.0 (+https://example.org)",
        "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
    })
    return session


def _load_best_cache(mission: str, limit: int) -> pd.DataFrame | None:
    """Load the freshest available cache for a mission, trimming to limit if needed."""
    # Exact cache
    exact = CACHE_DIR / f"{mission}_latest_{limit}.parquet"
    if exact.exists():
        try:
            df = pd.read_parquet(exact)
            return df
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", exact, e)
    # Fallback to any mission cache (pick newest by mtime)
    candidates = sorted(CACHE_DIR.glob(f"{mission}_latest_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            df = pd.read_parquet(path)
            if limit and len(df) > limit:
                return df.head(limit).copy()
            return df
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", path, e)
    return None


@lru_cache(maxsize=12)
def fetch_latest_catalog(
    mission: Literal["kepler"] = "kepler",
    limit: int = 200,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch latest exoplanet candidate data for a mission."""

    mission = mission.lower()
    if mission not in MISSION_TABLE_MAP:
        raise ValueError(f"Unsupported mission '{mission}'.")

    cache_file = CACHE_DIR / f"{mission}_latest_{limit}.parquet"
    if cache_file.exists() and not force_refresh:
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime < 3600:
            return pd.read_parquet(cache_file)

    params = {
        "query": _build_query(mission, limit),
        "format": "csv",
    }

    session = _get_http_session()
    try:
        response = session.get(BASE_URL, params=params, timeout=30)
        if response.status_code != 200:
            raise NASAExoplanetAPIError(
                f"NASA Exoplanet Archive request failed ({response.status_code}): {response.text[:200]}"
            )
        df = pd.read_csv(io.StringIO(response.text))
        # Persist fresh cache
        try:
            df.to_parquet(cache_file, index=False)
        except Exception as e:
            logger.warning("Failed to write cache %s: %s", cache_file, e)
        return df
    except Exception as e:
        logger.warning("NASA data fetch failed: %s", e)
        # Fallback: stale cache if available
        cached = _load_best_cache(mission, limit)
        if cached is not None:
            logger.info("Falling back to cached dataset for mission '%s' (rows=%d)", mission, len(cached))
            return cached
        # As a last resort, raise a meaningful error
        raise NASAExoplanetAPIError(
            "NASA verisi alınamadı ve geçerli bir önbellek bulunamadı. Lütfen daha sonra tekrar deneyin."
        ) from e


def get_latest_dataframe(
    mission: Literal["kepler"] = "kepler",
    limit: int = 200,
    force_refresh: bool = False,
    add_labels: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper returning cleaned dataframe with optional label column."""

    df = fetch_latest_catalog(mission=mission, limit=limit, force_refresh=force_refresh).copy()
    df.columns = [c.lower() for c in df.columns]

    if add_labels and "koi_disposition" in df.columns:
        df["is_exoplanet"] = (df["koi_disposition"].str.upper() == "CONFIRMED").astype(int)

    return df
