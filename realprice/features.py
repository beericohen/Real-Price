"""Feature engineering that learns from the TRAIN split only (this is what lifted the model's R^2).

WHY "train only"?  If a statistic (an average price per gush, an outlier limit ...) is computed on ALL the data,
the test deals leak into the features and the test score looks better than it really is ("data leakage").
So every statistic is learned from the train split alone:

* `FeatureBuilder.fit(train)` learns the KMeans clusters, the per-gush numbers, the price-trend tables and the
  pool of earlier deals that the "comparables" features look into.
* Train rows get *out-of-fold* target encodings (a row never sees its own price) and comparables / trends
  that only look at EARLIER deals, so they can never contain their own price.
* Test rows (or a new deal in production) are encoded with the train statistics only.

The target of all models is log(price per sqm). The price itself is exp(prediction) * asset_area.
"""
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from . import config as cfg

log = logging.getLogger(__name__)
EARTH_RADIUS_M = 6_371_000.0
OTHER = "other"   # label for rare or unseen deal types

# Columns copied as they are from the dataset
BASE_NUMERIC = ["lat", "long", "year_built", "age", "asset_area", "room_num", "deal_day", "deal_month",
                "deal_year", "distance_from_school", "schools_within_1km", "distance_from_public_transport",
                "transit_within_500m", "distance_from_center"]
# The three target-encoding levels, from coarse to fine, each with its "number of deals" column
TE_COLUMNS = ["te_cluster", "n_cluster", "te_gush", "n_gush", "te_gush_chelka", "n_gush_chelka"]
# Everything that is derived from year_built (all of it is hidden when the year is unknown)
YEAR_COLUMNS = ["year_built", "age", "building_age_at_deal", "is_new_build"]
# Integer codes that the tree models should treat as categories, not as numbers
CATEGORICAL_NUMERIC = ["cluster", "nature_code"]


# ------------------------------------------------------------------ small helpers
def deal_days(df: pd.DataFrame) -> np.ndarray:
    """The deal date as a whole number of days. Comparables use days: "earlier" means an earlier day."""
    parts = df[["deal_year", "deal_month", "deal_day"]].rename(
        columns={"deal_year": "year", "deal_month": "month", "deal_day": "day"})
    return pd.to_datetime(parts).to_numpy("datetime64[D]").astype("int64")


def to_metres(lat, lon) -> np.ndarray:
    """Degrees -> (x, y) in metres with a simple flat-earth projection around Israel's latitude (31.5).
    For the short distances used here (a few km) it is within about 1% of the exact (Haversine) distance,
    and it lets us use scipy's fast KD-tree for the neighbour search."""
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([EARTH_RADIUS_M * np.cos(np.radians(31.5)) * lon, EARTH_RADIUS_M * lat])


def month_index(df: pd.DataFrame) -> np.ndarray:
    """Months counted from year 0, so every month has one whole number."""
    return (df["deal_year"] * 12 + df["deal_month"] - 1).to_numpy()


def add_gush_chelka(df: pd.DataFrame) -> pd.DataFrame:
    """Add the combined key "gush_chelka", e.g. "6400_12", that identifies one parcel."""
    df = df.copy()
    df["gush_chelka"] = df["gush"].astype(str) + "_" + df["chelka"].astype(str)
    return df


def smoothed_mean(sums, counts, prior, m):
    """Average with shrinkage:  (sum + m * prior) / (n + m).
    With few deals (small n) the result stays close to the prior (the average of the parent group);
    with many deals it is the group's own average."""
    return (sums + m * prior) / (counts + m)


def summary_stats(values: np.ndarray, with_count: bool) -> list:
    """median, mean and std (and optionally the count) of the non-NaN values in every row."""
    count = np.isfinite(values).sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # a row of only NaN gives NaN - exactly what we want
        stats = [np.nanmedian(values, 1), np.nanmean(values, 1), np.where(count >= 2, np.nanstd(values, 1), np.nan)]
    return stats + ([count.astype(float)] if with_count else [])


class FeatureBuilder:
    """Usage:  builder.fit_transform_train(train)  ->  features of the train rows (leak-free)
               builder.transform(other_rows)      ->  features of test rows / a new deal, from train statistics only"""

    # ------------------------------------------------------------ outliers (needs statistics, so it lives here)
    def fit_outlier_bounds(self, train: pd.DataFrame) -> None:
        """Remember the 1st and 99th train percentile of price per sqm and of area."""
        low, high = cfg.OUTLIER_PCT
        self.bounds = {col: (train[col].quantile(low), train[col].quantile(high)) for col in ("price_per_sqrm", "asset_area")}
        log.info("outlier bounds (train percentiles %s): %s", cfg.OUTLIER_PCT,
                 {k: (round(a), round(b)) for k, (a, b) in self.bounds.items()})

    def filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows outside the TRAIN bounds. The same bounds are applied to the test rows."""
        keep = np.ones(len(df), bool)
        for col, (low, high) in self.bounds.items():
            keep &= df[col].between(low, high).to_numpy()
        log.info("outlier filter removed %d of %d rows", (~keep).sum(), len(df))
        return df[keep].reset_index(drop=True)

    # ------------------------------------------------------------ fit (train rows only)
    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        """Learn everything that comes from the train data. `train` must already be outlier-filtered."""
        random_state = cfg.RANDOM_STATE
        log_price = np.log(train["price_per_sqrm"])
        self.global_mean = float(log_price.mean())

        # 1. Spatial clusters: KMeans on (lat, long). On a huge table it is fitted on a random sample of 200k rows.
        coords = train[["lat", "long"]].to_numpy()
        sample = coords
        if len(coords) > 200_000:
            sample = coords[np.random.default_rng(random_state).choice(len(coords), 200_000, replace=False)]
        self.kmeans = KMeans(cfg.N_CLUSTERS, n_init=3, random_state=random_state).fit(sample)
        self.train_cluster = self.kmeans.predict(coords)   # kept, so we never have to predict the train rows again

        # 2. Deal types that are common enough to keep. Rare or unseen ones become "other".
        counts = train["deal_nature"].value_counts()
        self.natures = sorted(counts[counts >= cfg.MIN_NATURE_COUNT].index.tolist())

        # 3. Numbers per gush (no price involved)
        by_gush = train.groupby("gush")
        self.gush_stats = pd.DataFrame({"gush_n_deals": by_gush.size(),
                                        "gush_median_year_built": by_gush["year_built"].median(),
                                        "gush_median_area": by_gush["asset_area"].median()})

        # 4. What target encoding needs later: the train deals' parcel keys and prices
        self.train_ref = add_gush_chelka(train)[["gush", "gush_chelka", "price_per_sqrm"]]

        self.fit_trend(train)
        self.fit_comparables(train)
        return self

    # ------------------------------------------------------------ area price trend
    def fit_trend(self, train: pd.DataFrame) -> None:
        """Sum and count of log(price per sqm) for every (cluster, month), stored as running totals.
        The sum over any window of months is then just a difference of two running totals (fast)."""
        month = month_index(train)
        self.first_month = int(month.min())
        n_months = int(month.max() - month.min() + 1)
        log_price = np.log(train["price_per_sqrm"].to_numpy())

        sums = np.zeros((cfg.N_CLUSTERS, n_months))
        counts = np.zeros((cfg.N_CLUSTERS, n_months))
        np.add.at(sums, (self.train_cluster, month - self.first_month), log_price)
        np.add.at(counts, (self.train_cluster, month - self.first_month), 1)

        zero_column = np.zeros((cfg.N_CLUSTERS, 1))   # so that "up to month -1" is a valid, empty total
        self.running_sum = np.hstack([zero_column, sums.cumsum(1)])
        self.running_count = np.hstack([zero_column, counts.cumsum(1)])

    def trend_features(self, df: pd.DataFrame, cluster: np.ndarray) -> pd.DataFrame:
        """Mean log(price per sqm) of the TREND_MONTHS full months BEFORE the deal's month (its own month is excluded).
        Uses the deal's cluster; if the cluster has too few deals, the national trend is used instead."""
        n_months = self.running_sum.shape[1] - 1
        # last usable month; for dates after the train data this is the last month we know
        last = np.clip(month_index(df) - 1 - self.first_month, -1, n_months - 1)
        first = np.maximum(last - cfg.TREND_MONTHS, -1)

        local_sum = self.running_sum[cluster, last + 1] - self.running_sum[cluster, first + 1]
        local_count = self.running_count[cluster, last + 1] - self.running_count[cluster, first + 1]
        national_sum = self.running_sum[:, last + 1].sum(0) - self.running_sum[:, first + 1].sum(0)
        national_count = self.running_count[:, last + 1].sum(0) - self.running_count[:, first + 1].sum(0)

        national = np.where(national_count > 0, national_sum / np.maximum(national_count, 1), np.nan)
        local = np.where(local_count >= cfg.TREND_MIN_DEALS, local_sum / np.maximum(local_count, 1), national)
        return pd.DataFrame({"area_trend_3m": local, "national_trend_3m": national,
                             "area_trend_deals": local_count}, index=df.index)

    # ------------------------------------------------------------ comparables: prices of nearby EARLIER deals
    def fit_comparables(self, train: pd.DataFrame) -> None:
        """Remember where, when and for how much every train deal sold, plus a KD-tree to search them."""
        self.pool_xy = to_metres(train["lat"], train["long"])
        self.pool_day = deal_days(train)
        self.pool_price = train["price_per_sqrm"].to_numpy(float)
        self.pool_tree = cKDTree(self.pool_xy)

    def comparable_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """For every deal: statistics of price per sqm of the nearest deals that happened BEFORE it.

        The COMP_MAX_CANDIDATES nearest deals are fetched, the ones from the same day or later are thrown away,
        and the statistics are taken over what is left (the k nearest, or those within a radius).
        In very dense areas the radius statistics therefore cover at most that many nearest earlier deals.
        This is a deliberate trade-off: it is much faster than an exact search.
        """
        query_xy, query_day = to_metres(df["lat"], df["long"]), deal_days(df)
        n_candidates = min(cfg.COMP_MAX_CANDIDATES, len(self.pool_price))

        names = [f"comp_k{k}_{stat}" for k in cfg.COMP_NEIGHBOURS for stat in ("median", "mean", "std")]
        names += [f"comp_r{r}_{stat}" for r in cfg.COMP_RADII_M for stat in ("median", "mean", "std", "n")]
        names += ["comp_nearest_m"]
        result = np.full((len(df), len(names)), np.nan)

        # Work in batches so the neighbour tables do not use too much memory
        for start in range(0, len(df), cfg.COMP_CHUNK):
            batch = slice(start, start + cfg.COMP_CHUNK)
            n_rows = len(query_xy[batch])
            distance, index = self.pool_tree.query(query_xy[batch], k=n_candidates, workers=-1)
            distance, index = distance.reshape(n_rows, -1), index.reshape(n_rows, -1)

            earlier = self.pool_day[index] < query_day[batch][:, None]   # True where the neighbour sold before this deal
            rank = np.cumsum(earlier, axis=1)                            # 1 = nearest earlier deal, 2 = second nearest ...
            prices = self.pool_price[index]

            columns = []
            for k in cfg.COMP_NEIGHBOURS:
                columns += summary_stats(np.where(earlier & (rank <= k), prices, np.nan), with_count=False)
            for radius in cfg.COMP_RADII_M:
                columns += summary_stats(np.where(earlier & (distance <= radius), prices, np.nan), with_count=True)
            first_earlier = earlier.argmax(1)   # position of the nearest earlier deal
            columns.append(np.where(earlier.any(1), distance[np.arange(n_rows), first_earlier], np.nan))
            result[batch] = np.column_stack(columns)
        return pd.DataFrame(result, columns=names, index=df.index)

    # ------------------------------------------------------------ target encoding (average price of a group)
    def target_encode(self, fit: pd.DataFrame, fit_cluster, apply: pd.DataFrame, apply_cluster) -> pd.DataFrame:
        """The average log(price per sqm) of the cluster, the gush and the parcel, learned from the `fit` rows
        and looked up for the `apply` rows. Three levels, each pulled towards the level above it
        (cluster -> gush -> parcel) so that a parcel with only 2 deals does not get a wild average."""
        m = cfg.TE_SMOOTHING
        fit_prices = pd.DataFrame({"y": np.log(fit["price_per_sqrm"].to_numpy()), "cluster": fit_cluster,
                                   "gush": fit["gush"].to_numpy(), "gush_chelka": fit["gush_chelka"].to_numpy()})
        prior = np.full(len(apply), self.global_mean)   # the first level is pulled towards the global average
        levels = (("cluster", np.asarray(apply_cluster)), ("gush", apply["gush"].to_numpy()),
                  ("gush_chelka", apply["gush_chelka"].to_numpy()))

        result = {}
        for name, apply_keys in levels:
            totals = fit_prices["y"].groupby(fit_prices[name]).agg(["sum", "count"])
            sums = pd.Series(apply_keys).map(totals["sum"]).fillna(0).to_numpy()
            counts = pd.Series(apply_keys).map(totals["count"]).fillna(0).to_numpy()
            prior = smoothed_mean(sums, counts, prior, m)   # this level becomes the prior of the next, finer one
            result["te_" + name] = prior
            result["n_" + name] = counts
        return pd.DataFrame(result, index=apply.index)

    # ------------------------------------------------------------ assembling the final table
    def row_features(self, df: pd.DataFrame, cluster: np.ndarray) -> pd.DataFrame:
        """Features computed from the row itself (and the train statistics). No price of the row is used."""
        feats = df[BASE_NUMERIC].copy()
        feats["deal_t"] = df["deal_year"] + (df["deal_month"] - 1) / 12   # time as one continuous number
        feats["area_per_room"] = df["asset_area"] / df["room_num"]
        feats["building_age_at_deal"] = (df["deal_year"] - df["year_built"]).clip(lower=0)
        known_year = feats["building_age_at_deal"].notna()   # an unknown year_built stays NaN
        feats["is_new_build"] = (feats["building_age_at_deal"] <= cfg.NEW_BUILD_MAX_AGE).astype(float).where(known_year)
        feats["lat_plus_long"] = df["lat"] + df["long"]       # two diagonal axes help trees split on location
        feats["lat_minus_long"] = df["lat"] - df["long"]
        feats["cluster"] = cluster

        nature = df["deal_nature"].where(df["deal_nature"].isin(self.natures), OTHER)
        feats["nature_code"] = pd.Categorical(nature, categories=self.natures + [OTHER]).codes

        gush_numbers = self.gush_stats.reindex(df["gush"])   # a gush the train data never saw gives NaN
        for col in self.gush_stats.columns:
            feats[col] = gush_numbers[col].to_numpy()
        feats["gush_n_deals"] = feats["gush_n_deals"].fillna(0)
        return pd.concat([feats, self.trend_features(df, cluster)], axis=1)

    def assemble(self, df: pd.DataFrame, feats: pd.DataFrame, te: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
        """Put all feature groups side by side and add the target and the columns needed to evaluate the model."""
        # cat_* = text keys. Only CatBoost uses them (it does its own target encoding on text).
        keys = df[["gush", "chelka", "gush_chelka", "deal_nature"]].astype(str)
        keys["deal_nature"] = keys["deal_nature"].where(df["deal_nature"].isin(self.natures), OTHER)
        out = pd.concat([feats, te, comps, keys.add_prefix("cat_")], axis=1)
        out[cfg.TARGET] = np.log(df["price_per_sqrm"].to_numpy())
        out["asset_area_raw"] = df["asset_area"].to_numpy()   # to turn price per sqm back into a price
        out["deal_amount"] = df["deal_amount"].to_numpy()     # the real price, to measure the error
        return out

    # ------------------------------------------------------------ public API
    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        """Features for the rows the builder was fitted on, without leakage:
        out-of-fold target encoding (each row is encoded by a model that did not see it) and comparables
        that only use earlier deals."""
        train = add_gush_chelka(train.reset_index(drop=True))
        te = pd.DataFrame(index=train.index, columns=TE_COLUMNS, dtype=float)
        folds = KFold(cfg.TE_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
        for fit_rows, apply_rows in folds.split(train):
            te.iloc[apply_rows] = self.target_encode(
                train.iloc[fit_rows], self.train_cluster[fit_rows],
                train.iloc[apply_rows], self.train_cluster[apply_rows]).to_numpy()
        feats = self.row_features(train, self.train_cluster)
        return self.assemble(train, feats, te, self.comparable_features(train))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for rows the builder has not seen (test rows, a new deal): train statistics only."""
        df = add_gush_chelka(df.reset_index(drop=True))
        cluster = self.kmeans.predict(df[["lat", "long"]].to_numpy())
        te = self.target_encode(self.train_ref, self.train_cluster, df, cluster)
        feats = self.row_features(df, cluster)
        return self.assemble(df, feats, te, self.comparable_features(df))

    def fit_transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        """Outlier bounds -> remove outliers -> fit -> leak-free features. Returns the features of the filtered train rows."""
        self.fit_outlier_bounds(train)
        train = self.filter_outliers(train)
        self.fit(train)
        return self.transform_train(train)


# ------------------------------------------------------------------ unknown year_built
def hide_year(feats: pd.DataFrame, fraction: float = 1.0, seed: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Set the year-based columns to NaN for a random `fraction` of the rows (1.0 = all rows).
    The neighbour and encoding features do not use the year, so hiding it afterwards is exact.
    Used (a) on the TRAIN rows, so the models also learn the "year unknown" case, and
    (b) on the TEST rows, to measure how good the model is when the user does not know the year."""
    feats = feats.copy()
    rows = np.random.default_rng(seed).random(len(feats)) < fraction
    columns = [c for c in YEAR_COLUMNS if c in feats.columns]
    feats.loc[rows, columns] = np.nan
    return feats


# ------------------------------------------------------------------ which columns each model sees
def feature_lists(columns) -> dict:
    """The input columns of each model family. The `cat_*` text keys are for CatBoost only."""
    text_keys = [c for c in columns if c.startswith("cat_")]
    numbers = [c for c in columns if c not in (cfg.TARGET, "asset_area_raw", "deal_amount") and c not in text_keys]
    return {"numeric_model": numbers,                # HistGradientBoosting and LightGBM
            "catboost": numbers + text_keys,
            "catboost_cat_features": text_keys}
