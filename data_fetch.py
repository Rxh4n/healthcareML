"""
Download REAL UK area-level data from the OHID Fingertips API and cache it.

Each indicator is fetched as CSV (one file per indicator) for every upper-tier
local authority in England, then cached under data/raw so re-runs are offline
and fast. Run this module directly to (re)download:

    python data_fetch.py
"""
from __future__ import annotations

import io
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

import config

_SSL = ssl.create_default_context()


def _csv_url(indicator_id: int) -> str:
    return (
        f"{config.FINGERTIPS_BASE}/all_data/csv/by_indicator_id"
        f"?indicator_ids={indicator_id}"
        f"&child_area_type_id={config.FINGERTIPS_AREA_TYPE}"
        f"&parent_area_type_id={config.FINGERTIPS_PARENT_AREA_TYPE}"
    )


def _download(url: str, timeout: int = 120, retries: int = 3) -> str:
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001  (network errors are varied)
            last = exc
            print(f"    attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from last


def fetch_indicator(indicator_id: int, friendly: str, force: bool = False) -> pd.DataFrame:
    """Fetch one indicator's CSV (cached at data/raw/<id>_<friendly>.csv)."""
    cache = config.DATA_RAW / f"{indicator_id}_{friendly}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, low_memory=False)

    print(f"  downloading {indicator_id} -> {friendly}")
    text = _download(_csv_url(indicator_id))
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if df.empty:
        raise RuntimeError(f"indicator {indicator_id} returned no rows")
    df.to_csv(cache, index=False)
    return df


def fetch_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch the outcome and every feature indicator."""
    out: dict[str, pd.DataFrame] = {}
    print("Fetching REAL UK area-level data from OHID Fingertips...")

    out[config.AREA_OUTCOME["name"]] = fetch_indicator(
        config.AREA_OUTCOME["id"], config.AREA_OUTCOME["name"], force=force
    )
    for ind_id, friendly in config.AREA_FEATURES.items():
        out[friendly] = fetch_indicator(ind_id, friendly, force=force)

    print(f"Done. {len(out)} indicators cached under {config.DATA_RAW}")
    return out


if __name__ == "__main__":
    force = "--force" in sys.argv
    try:
        fetch_all(force=force)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFETCH FAILED: {exc}", file=sys.stderr)
        print(
            "If you are offline, connect to the internet and re-run. The Fingertips\n"
            "API is public and needs no key: https://fingertips.phe.org.uk/",
            file=sys.stderr,
        )
        sys.exit(1)
