"""NASA Exoplanet Archive data access utilities."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import io

import pandas as pd
import requests

MISSION_TABLE_MAP: dict[str, str] = {
    "kepler": "cumulative",  # Ana cumulative tablo - tüm fiziksel parametreleri içerir
}

BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CACHE_DIR = Path(os.getenv("EXOPLANET_CACHE", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
        "format": "csv"
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    if response.status_code != 200:
        raise NASAExoplanetAPIError(
            f"NASA Exoplanet Archive request failed ({response.status_code}): {response.text[:200]}"
        )

    df = pd.read_csv(io.StringIO(response.text))
    df.to_parquet(cache_file, index=False)
    return df


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
