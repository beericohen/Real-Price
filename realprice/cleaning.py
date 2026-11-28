"""Step 1 - cleaning: merge the raw deal files and remove rows we cannot use.

Every rule is a small function that takes a table and returns a smaller (or fixed) table.
`run()` applies them one after the other and logs how many rows each rule removed.
"""
import logging

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import extract_zips, normalise_columns, read_table, write_csv

log = logging.getLogger(__name__)
READABLE_SUFFIXES = {".csv", ".txt", ".xlsx", ".xls"}


def load_raw_deals() -> pd.DataFrame:
    """Read every deal file (one per settlement) from data/raw and stack them into one table."""
    extract_zips(cfg.DATA_RAW)
    files = sorted(p for p in cfg.DATA_RAW.iterdir() if p.suffix.lower() in READABLE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No deal files in {cfg.DATA_RAW} - copy the 51 files (or the ZIP) there.")
    log.info("reading %d files", len(files))
    return pd.concat((normalise_columns(read_table(p)) for p in files), ignore_index=True)


# ------------------------------------------------------------------ cleaning rules
def drop_unused_columns(df):
    return df.drop(columns=cfg.DROP_COLUMNS, errors="ignore")


def coerce_types(df):
    """Empty strings -> missing, text -> numbers / dates. Values that cannot be parsed become NaN
    and are removed by a later rule."""
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    for col in cfg.NUMERIC_COLUMNS + ["portion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["deal_date"] = pd.to_datetime(df["deal_date"], format=cfg.DATE_FORMAT, errors="coerce")
    return df


def blank_invalid_year(df):
    """Years like 0 or 1900 are placeholders, not real data. Make them NaN (= unknown) and KEEP the deal."""
    year = df["year_built"]
    df["year_built"] = year.where(year.between(*cfg.VALID_RANGES["year_built"]))
    log.info("year_built unknown for %d of %d deals (kept)", df["year_built"].isna().sum(), len(df))
    return df


def drop_empty_rows(df):
    """Remove rows with a missing value in any column except the optional ones."""
    log.info("missing values per column:\n%s", df.isna().sum()[lambda s: s > 0].to_string())
    required = [c for c in df.columns if c not in cfg.OPTIONAL_COLUMNS]
    return df.dropna(subset=required)


def drop_zero_rows(df):
    """For these columns a 0 means "no data"."""
    return df[(df[cfg.NONZERO_COLUMNS] != 0).all(axis=1)]


def drop_duplicates(df):
    return df.drop_duplicates()


def drop_out_of_range(df):
    """Remove impossible values (tiny areas, absurd prices ...) using the limits in config.VALID_RANGES."""
    df = df.assign(price_per_sqrm=df["deal_amount"] / df["asset_area"])
    keep = pd.Series(True, index=df.index)
    for col, (low, high) in cfg.VALID_RANGES.items():
        in_range = df[col].between(low, high)
        if col in cfg.OPTIONAL_COLUMNS:
            in_range = in_range | df[col].isna()   # an unknown year is fine
        keep &= in_range
    return df[keep]


def keep_full_ownership(df):
    """portion < 1 means only a share of the property was sold, so deal_amount is not the price of the whole asset."""
    return df[np.isclose(df["portion"], cfg.FULL_OWNERSHIP_PORTION)]


def keep_residential(df):
    """Keep only normal dwellings; land, farms, shops, offices ... are not comparable."""
    return df[df["deal_nature"].isin(cfg.RESIDENTIAL_NATURES)]


def finalise_types(df):
    """Whole-number columns -> int64, and sort by date.

    room_num is NOT in this list: Israeli deals use half rooms (3.5, 4.5), and casting to int64 would silently
    round them down (3.5 -> 3). It stays a float.
    """
    int_columns = [c for c in cfg.NUMERIC_COLUMNS
                   if c not in ("deal_amount", "declared_amount", "asset_area", "year_built", "room_num")]
    df[int_columns] = df[int_columns].astype("int64")
    df["deal_nature"] = df["deal_nature"].astype(str)
    return df.sort_values("deal_date").reset_index(drop=True)


# The order matters.
# Note: statistical outlier removal (percentiles of price per sqm and area) is NOT done here. It needs
# statistics, so it happens after the train / test split, using the train data only (see features.py).
RULES = [drop_unused_columns, coerce_types, blank_invalid_year, drop_empty_rows, drop_zero_rows, drop_duplicates,
         keep_full_ownership, keep_residential, drop_out_of_range, finalise_types]


def run() -> pd.DataFrame:
    df = load_raw_deals()
    log.info("raw rows: %d", len(df))
    for rule in RULES:
        before = len(df)
        df = rule(df)
        log.info("%-22s removed %9d rows -> %d left", rule.__name__, before - len(df), len(df))
    write_csv(df, cfg.CLEANED_CSV)
    log.info("cleaned dataset: %d rows -> %s", len(df), cfg.CLEANED_CSV)
    return df
