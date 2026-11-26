"""Step 1 - cleaning: merge the raw deal files and remove unusable rows.

Every rule is a small DataFrame -> DataFrame function; `_logged` reports how many rows each rule removed.
"""
from functools import wraps

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import (extract_zips, get_logger, normalise_columns, read_table, timed, write_csv)

log = get_logger(__name__)
READABLE = {".csv", ".txt", ".xlsx", ".xls"}


def _logged(rule):
    """Decorator: log the number of rows a cleaning rule dropped."""
    @wraps(rule)
    def wrapper(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = rule(df)
        log.info("%-22s removed %9d rows -> %d left", rule.__name__, before - len(df), len(df))
        return df
    return wrapper


def load_raw_deals() -> pd.DataFrame:
    """Concatenate every deal file (one per settlement) found in data/raw."""
    extract_zips(cfg.DATA_RAW)
    files = sorted(p for p in cfg.DATA_RAW.iterdir() if p.suffix.lower() in READABLE)
    if not files:
        raise FileNotFoundError(f"No deal files in {cfg.DATA_RAW} - copy the 51 files (or the ZIP) there.")
    log.info("reading %d files", len(files))
    return pd.concat((normalise_columns(read_table(p)) for p in files), ignore_index=True)


# ------------------------------------------------------------------ cleaning rules
@_logged
def drop_unused_columns(df):
    return df.drop(columns=cfg.DROP_COLUMNS, errors="ignore")


@_logged
def coerce_types(df):
    """Blank strings -> NaN, numeric / date parsing (unparsable values become NaN and are dropped later)."""
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    for col in cfg.NUMERIC_COLUMNS + ["portion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["deal_date"] = pd.to_datetime(df["deal_date"], format=cfg.DATE_FORMAT, errors="coerce")
    return df


@_logged
def blank_invalid_year(df):
    """0 / 1900 / impossible years are placeholders, not real data: turn them into NaN (= unknown) and KEEP the deal."""
    year = df["year_built"]
    df["year_built"] = year.where(year.between(*cfg.VALID_RANGES["year_built"]))
    log.info("year_built unknown for %d of %d deals (kept)", df["year_built"].isna().sum(), len(df))
    return df


@_logged
def drop_empty_rows(df):
    log.info("missing values per column:\n%s", df.isna().sum()[lambda s: s > 0].to_string())
    return df.dropna(subset=[c for c in df.columns if c not in cfg.OPTIONAL_COLUMNS])


@_logged
def drop_zero_rows(df):
    return df[(df[cfg.NONZERO_COLUMNS] != 0).all(axis=1)]


@_logged
def drop_duplicates(df):
    return df.drop_duplicates()


@_logged
def drop_out_of_range(df):
    """Hard domain limits from config (negative values, impossible areas, absurd prices ...)."""
    df = df.assign(price_per_sqrm=df["deal_amount"] / df["asset_area"])
    mask = np.ones(len(df), dtype=bool)
    for col, (low, high) in cfg.VALID_RANGES.items():
        ok = df[col].between(low, high)
        mask &= (ok | df[col].isna()).to_numpy() if col in cfg.OPTIONAL_COLUMNS else ok.to_numpy()
    return df[mask]


@_logged
def keep_full_ownership(df):
    """portion < 1 means only a share of the property was sold, so deal_amount is not the price of the whole asset."""
    return df[np.isclose(df["portion"], cfg.FULL_OWNERSHIP_PORTION)]


@_logged
def keep_residential(df):
    """Drop non-comparable deals (land, farms, combination deals, buildings, shops, offices ...)."""
    return df[df["deal_nature"].isin(cfg.RESIDENTIAL_NATURES)]


# NOTE: statistical outlier removal (percentiles of price_per_sqrm / asset_area) is NOT done here any more:
# it needs statistics, so it runs after the train/test split, with train-only thresholds (features.py).

@_logged
def finalise_types(df):
    int_cols = [c for c in cfg.NUMERIC_COLUMNS if c not in ("deal_amount", "declared_amount", "asset_area", "year_built")]
    df[int_cols] = df[int_cols].astype("int64")
    df["deal_nature"] = df["deal_nature"].astype(str)
    return df.sort_values("deal_date").reset_index(drop=True)


RULES = [drop_unused_columns, coerce_types, blank_invalid_year, drop_empty_rows, drop_zero_rows, drop_duplicates,
         keep_full_ownership, keep_residential, drop_out_of_range, finalise_types]


@timed
def run() -> pd.DataFrame:
    df = load_raw_deals()
    log.info("raw rows: %d", len(df))
    for rule in RULES:
        df = rule(df)
    write_csv(df, cfg.CLEANED_CSV)
    log.info("cleaned dataset: %d rows -> %s", len(df), cfg.CLEANED_CSV)
    return df
