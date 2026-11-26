"""Split-aware feature engineering (the heart of the R^2 upgrade).

LEAKAGE RULES - every statistic below is learned from the TRAIN split only:

* `FeatureBuilder.fit(train)` learns outlier bounds, KMeans clusters, gush aggregates, the price-trend tables and
  the pool of past deals used for the spatial-temporal comparables.
* Train rows get *out-of-fold* target encodings (a row never sees its own target) and comparables / trends that
  look strictly into the past, so they can never contain their own price.
* Test rows (or a new deal in production) are encoded with train statistics only - test targets are never read.

Target of all models: log(price per sqm). The money value is exp(pred) * asset_area (see training.py).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from . import config as cfg
from .utils import get_logger

log = get_logger(__name__)
EARTH_R = 6_371_000.0
OTHER = "other"
RAW_CATEGORICAL = ["gush", "chelka", "gush_chelka", "deal_nature"]       # string keys, used by CatBoost
BASE_NUMERIC = ["lat", "long", "year_built", "age", "asset_area", "room_num", "deal_day", "deal_month",
                "deal_year", "distance_from_school", "schools_within_1km", "distance_from_public_transport",
                "transit_within_500m", "distance_from_center"]


# ------------------------------------------------------------------ small helpers
def _days(df: pd.DataFrame) -> np.ndarray:
    """Deal date as integer days (comparables use day resolution: 'strictly before' means an earlier day)."""
    parts = df[["deal_year", "deal_month", "deal_day"]].rename(
        columns={"deal_year": "year", "deal_month": "month", "deal_day": "day"})
    return pd.to_datetime(parts).to_numpy("datetime64[D]").astype("int64")


def _to_xy(lat, lon) -> np.ndarray:
    """Equirectangular projection to metres around Israel's latitude: within ~1% of true Haversine distance for
    the distances used here (<= a few km), and lets scipy's fast cKDTree do the neighbour search."""
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([EARTH_R * np.cos(np.radians(31.5)) * lon, EARTH_R * lat])


def _smoothed_mean(sums: pd.Series, counts: pd.Series, prior, m: float):
    """Bayesian shrinkage: (sum + m * prior) / (n + m).  Few deals -> close to the prior (parent group mean)."""
    return (sums + m * prior) / (counts + m)


class FeatureBuilder:
    """fit on train -> `transform_train` (leak-free, out-of-fold) / `transform` (test, production)."""

    # ------------------------------------------------------------ 1. cleaning that needs statistics
    def _fit_outlier_bounds(self, train: pd.DataFrame) -> None:
        lo, hi = cfg.OUTLIER_PCT
        self.bounds = {c: (train[c].quantile(lo), train[c].quantile(hi)) for c in ("price_per_sqrm", "asset_area")}
        log.info("outlier bounds (train percentiles %s): %s", cfg.OUTLIER_PCT,
                 {k: (round(a), round(b)) for k, (a, b) in self.bounds.items()})

    def filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop price_per_sqrm / asset_area extremes using the TRAIN bounds (also applied to test)."""
        keep = np.ones(len(df), bool)
        for col, (lo, hi) in self.bounds.items():
            keep &= df[col].between(lo, hi).to_numpy()
        log.info("outlier filter removed %d of %d rows", (~keep).sum(), len(df))
        return df[keep].reset_index(drop=True)

    # ------------------------------------------------------------ 2. fit (train only)
    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        """`train` must already be outlier-filtered. Learns every train-side statistic."""
        rng = cfg.RANDOM_STATE
        self.global_mean = float(np.log(train["price_per_sqrm"]).mean())
        # spatial clusters
        coords = train[["lat", "long"]].to_numpy()
        sample = coords if len(coords) <= 200_000 else coords[np.random.default_rng(rng).choice(len(coords), 200_000, False)]
        self.kmeans = KMeans(cfg.N_CLUSTERS, n_init=3, random_state=rng).fit(sample)
        # deal_nature categories (rare / unseen -> "other")
        counts = train["deal_nature"].value_counts()
        self.natures = sorted(counts[counts >= cfg.MIN_NATURE_COUNT].index.tolist())
        # gush-level aggregates (feature statistics, no target involved)
        by_gush = train.groupby("gush")
        self.gush_stats = pd.DataFrame({"gush_n_deals": by_gush.size(),
                                        "gush_median_year_built": by_gush["year_built"].median(),
                                        "gush_median_area": by_gush["asset_area"].median()})
        # pools for trend + comparables
        self._fit_trend(train)
        self._fit_comps(train)
        return self

    def _clusters(self, df) -> np.ndarray:
        return self.kmeans.predict(df[["lat", "long"]].to_numpy())

    # ------------------------------------------------------------ 3. price trend (detrending feature)
    @staticmethod
    def _month_index(df) -> np.ndarray:
        return (df["deal_year"] * 12 + df["deal_month"] - 1).to_numpy()

    def _fit_trend(self, train: pd.DataFrame) -> None:
        """Cumulative sums of log(ppsm) per (cluster, month) -> the mean over any window of past months in O(1)."""
        month, cluster = self._month_index(train), self._clusters(train)
        self.m0, n_months = int(month.min()), int(month.max() - month.min() + 1)
        y = np.log(train["price_per_sqrm"].to_numpy())
        sums, cnts = np.zeros((cfg.N_CLUSTERS, n_months)), np.zeros((cfg.N_CLUSTERS, n_months))
        np.add.at(sums, (cluster, month - self.m0), y)
        np.add.at(cnts, (cluster, month - self.m0), 1)
        pad = np.zeros((cfg.N_CLUSTERS, 1))
        self._cs_sum, self._cs_cnt = np.hstack([pad, sums.cumsum(1)]), np.hstack([pad, cnts.cumsum(1)])

    def _trend(self, df: pd.DataFrame, cluster: np.ndarray) -> pd.DataFrame:
        """Mean log(ppsm) of the TREND_MONTHS full calendar months BEFORE the deal's month (own month excluded)."""
        n_months = self._cs_sum.shape[1] - 1
        hi = np.clip(self._month_index(df) - 1 - self.m0, -1, n_months - 1)   # last usable month (future -> last known)
        lo = np.maximum(hi - cfg.TREND_MONTHS, -1)
        s = self._cs_sum[cluster, hi + 1] - self._cs_sum[cluster, lo + 1]
        c = self._cs_cnt[cluster, hi + 1] - self._cs_cnt[cluster, lo + 1]
        s_nat = self._cs_sum[:, hi + 1].sum(0) - self._cs_sum[:, lo + 1].sum(0)
        c_nat = self._cs_cnt[:, hi + 1].sum(0) - self._cs_cnt[:, lo + 1].sum(0)
        national = np.where(c_nat > 0, s_nat / np.maximum(c_nat, 1), np.nan)
        local = np.where(c >= cfg.TREND_MIN_DEALS, s / np.maximum(c, 1), national)  # sparse cluster -> national trend
        return pd.DataFrame({"area_trend_3m": local, "national_trend_3m": national,
                             "area_trend_deals": c}, index=df.index)

    # ------------------------------------------------------------ 4. spatial-temporal comparables
    def _fit_comps(self, train: pd.DataFrame) -> None:
        self.pool_xy = _to_xy(train["lat"], train["long"])
        self.pool_day = _days(train)
        self.pool_val = train["price_per_sqrm"].to_numpy(float)
        self.pool_tree = cKDTree(self.pool_xy)

    def _comps(self, df: pd.DataFrame) -> pd.DataFrame:
        """For every deal: price_per_sqrm statistics of the nearest PAST deals (day < this deal's day) of the pool.

        Nearest `COMP_MAX_CANDIDATES` pool deals are fetched, future ones are masked out, and the k-nearest /
        within-radius statistics are taken over what is left. (In very dense areas the radius statistics are
        therefore computed over at most that many nearest past deals - an intentional speed/accuracy trade-off.)
        """
        q_xy, q_day = _to_xy(df["lat"], df["long"]), _days(df)
        k_cand = min(cfg.COMP_MAX_CANDIDATES, len(self.pool_val))
        names = [f"comp_k{k}_{s}" for k in cfg.COMP_NEIGHBOURS for s in ("median", "mean", "std")] + \
                [f"comp_r{r}_{s}" for r in cfg.COMP_RADII_M for s in ("median", "mean", "std", "n")] + ["comp_nearest_m"]
        out = np.full((len(df), len(names)), np.nan)
        for start in range(0, len(df), cfg.COMP_CHUNK):
            sl = slice(start, start + cfg.COMP_CHUNK)
            dist, idx = self.pool_tree.query(q_xy[sl], k=k_cand, workers=-1)
            dist, idx = dist.reshape(len(q_xy[sl]), -1), idx.reshape(len(q_xy[sl]), -1)
            past = self.pool_day[idx] < q_day[sl][:, None]
            rank = np.cumsum(past, axis=1)                       # 1 = nearest past deal, 2 = second nearest ...
            vals = self.pool_val[idx]
            cols = []
            for k in cfg.COMP_NEIGHBOURS:
                cols += self._stats(np.where(past & (rank <= k), vals, np.nan), with_n=False)
            for r in cfg.COMP_RADII_M:
                cols += self._stats(np.where(past & (dist <= r), vals, np.nan), with_n=True)
            first = past.argmax(1)
            cols.append(np.where(past.any(1), dist[np.arange(len(dist)), first], np.nan))
            out[sl] = np.column_stack(cols)
        return pd.DataFrame(out, columns=names, index=df.index)

    @staticmethod
    def _stats(vals: np.ndarray, with_n: bool) -> list:
        """median / mean / std (/ count) over the non-NaN entries of every row."""
        n = np.isfinite(vals).sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows -> NaN is exactly what we want
            res = [np.nanmedian(vals, 1), np.nanmean(vals, 1), np.where(n >= 2, np.nanstd(vals, 1), np.nan)]
        return res + ([n.astype(float)] if with_n else [])

    # ------------------------------------------------------------ 5. target encoding (smoothed, hierarchical)
    def _te_levels(self, fit: pd.DataFrame, apply: pd.DataFrame, fit_cluster, apply_cluster) -> pd.DataFrame:
        """cluster -> gush -> gush_chelka: each level is shrunk toward the level above it (m = TE_SMOOTHING)."""
        m, y = cfg.TE_SMOOTHING, np.log(fit["price_per_sqrm"].to_numpy())
        frame = pd.DataFrame({"y": y, "cluster": fit_cluster, "gush": fit["gush"].to_numpy(),
                              "gc": fit["gush_chelka"].to_numpy()})
        prior = np.full(len(apply), self.global_mean)
        out = {}
        for name, key_fit, key_apply in (("te_cluster", frame["cluster"], np.asarray(apply_cluster)),
                                         ("te_gush", frame["gush"], apply["gush"].to_numpy()),
                                         ("te_gush_chelka", frame["gc"], apply["gush_chelka"].to_numpy())):
            agg = frame["y"].groupby(key_fit).agg(["sum", "count"])
            s = pd.Series(key_apply).map(agg["sum"]).fillna(0).to_numpy()
            c = pd.Series(key_apply).map(agg["count"]).fillna(0).to_numpy()
            prior = _smoothed_mean(s, c, prior, m)               # the result is the prior of the next, finer level
            out[name] = prior
            out[name.replace("te_", "n_")] = c
        return pd.DataFrame(out, index=apply.index)

    # ------------------------------------------------------------ 6. assemble
    def _assemble(self, df: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["gush_chelka"] = df["gush"].astype(str) + "_" + df["chelka"].astype(str)
        return df

    def _common(self, df: pd.DataFrame) -> pd.DataFrame:
        """Row-wise (stateless or train-state-only, target-free) features shared by train and test."""
        cluster = self._clusters(df)
        feats = df[BASE_NUMERIC].copy()
        feats["deal_t"] = df["deal_year"] + (df["deal_month"] - 1) / 12
        feats["area_per_room"] = df["asset_area"] / df["room_num"]
        feats["building_age_at_deal"] = (df["deal_year"] - df["year_built"]).clip(lower=0)
        known = feats["building_age_at_deal"].notna()          # year_built may be unknown (NaN stays NaN)
        feats["is_new_build"] = (feats["building_age_at_deal"] <= cfg.NEW_BUILD_MAX_AGE).astype(float).where(known)
        feats["lat_plus_long"], feats["lat_minus_long"] = df["lat"] + df["long"], df["lat"] - df["long"]
        feats["cluster"] = cluster
        nature = df["deal_nature"].where(df["deal_nature"].isin(self.natures), OTHER)
        feats["nature_code"] = pd.Categorical(nature, categories=self.natures + [OTHER]).codes
        stats = self.gush_stats.reindex(df["gush"])
        for col in self.gush_stats.columns:
            feats[col] = stats[col].to_numpy()
        feats["gush_n_deals"] = feats["gush_n_deals"].fillna(0)
        feats = pd.concat([feats, self._trend(df, cluster)], axis=1)
        return feats

    def _finish(self, df, feats, te, comps) -> pd.DataFrame:
        keys = df[["gush", "chelka", "gush_chelka", "deal_nature"]].astype(str)
        keys["deal_nature"] = keys["deal_nature"].where(df["deal_nature"].isin(self.natures), OTHER)
        out = pd.concat([feats, te, comps, keys.add_prefix("cat_")], axis=1)
        out["log_ppsm"] = np.log(df["price_per_sqrm"].to_numpy())
        out["asset_area_raw"] = df["asset_area"].to_numpy()
        out["deal_amount"] = df["deal_amount"].to_numpy()
        return out

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        """Features for the rows the builder was fitted on: out-of-fold target encoding + strictly-past comparables."""
        train = self._assemble(train.reset_index(drop=True), None)
        cluster = self._clusters(train)
        te = pd.DataFrame(index=train.index, columns=["te_cluster", "n_cluster", "te_gush", "n_gush",
                                                       "te_gush_chelka", "n_gush_chelka"], dtype=float)
        for fit_idx, app_idx in KFold(cfg.TE_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE).split(train):
            te.iloc[app_idx] = self._te_levels(train.iloc[fit_idx], train.iloc[app_idx],
                                               cluster[fit_idx], cluster[app_idx]).to_numpy()
        return self._finish(train, self._common(train), te, self._comps(train))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for unseen rows (test / production): train statistics only."""
        df = self._assemble(df.reset_index(drop=True), None)
        fit = self._assemble(self._train_ref, None)
        te = self._te_levels(fit, df, self._clusters(fit), self._clusters(df))
        return self._finish(df, self._common(df), te, self._comps(df))

    def fit_transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        """Convenience: outlier bounds -> filter -> fit -> OOF features. Returns the filtered-train feature frame."""
        self._fit_outlier_bounds(train)
        train = self.filter_outliers(train)
        self._train_ref = train[["gush", "chelka", "price_per_sqrm", "lat", "long"]].copy()
        self.fit(train)
        return self.transform_train(train)


# ------------------------------------------------------------------ unknown year_built
YEAR_COLUMNS = ["year_built", "age", "building_age_at_deal", "is_new_build"]   # everything derived from year_built


def hide_year(feats: pd.DataFrame, frac: float = 1.0, seed: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Set the year-derived columns to NaN for a random `frac` of rows (frac=1 -> all rows).
    Neighbour / encoding features do not depend on the year, so hiding it afterwards is exact.
    Used (a) to augment TRAIN so the models learn the 'unknown year' case, (b) to score TEST without the year."""
    feats = feats.copy()
    rows = np.random.default_rng(seed).random(len(feats)) < frac
    cols = [c for c in YEAR_COLUMNS if c in feats.columns]
    feats.loc[rows, cols] = np.nan
    return feats


# ------------------------------------------------------------------ feature lists per model family
TE_COLUMNS = ["te_cluster", "n_cluster", "te_gush", "n_gush", "te_gush_chelka", "n_gush_chelka"]
CATEGORICAL_NUMERIC = ["cluster", "nature_code"]      # integer-coded: HGB / LightGBM treat them as categorical


def feature_lists(columns) -> dict:
    """Which columns each model family sees. (`cat_*` raw string keys are CatBoost-only: it target-encodes them itself.)"""
    cols = [c for c in columns if c not in ("log_ppsm", "asset_area_raw", "deal_amount") and not c.startswith("cat_")]
    return {"numeric_model": cols,                                   # HGB + LightGBM (te_* + comps already inside)
            "catboost": cols + [c for c in columns if c.startswith("cat_")],
            "catboost_cat_features": [c for c in columns if c.startswith("cat_")]}
