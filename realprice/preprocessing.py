"""Step 4 - preprocessing: SPLIT FIRST, then every statistic is learned from the train split only (features.py).

Order of operations (this is what prevents data leakage):
  1. random train / test split of the raw cleaned+enriched deals
  2. percentile outlier bounds, KMeans, gush aggregates, price trend tables, comparables pool, target-encoding
     statistics  <- all fitted on TRAIN
  3. train rows: out-of-fold target encoding + strictly-past comparables;  test rows: train statistics only

Two feature views are produced from the same rows:
  * the full feature frame (spatial comps, encodings ...) for HistGradientBoosting / CatBoost / LightGBM
  * a small browser-computable matrix (`web_matrix`) for the model exported to the static website
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from .features import FeatureBuilder, OTHER, feature_lists, hide_year
from .utils import ensure_dirs, get_logger, read_csv, save_json, timed

log = get_logger(__name__)
WEB_NUMERIC = ["lat", "long", "year_built", "age", "asset_area", "room_num", "deal_day", "deal_month", "deal_year",
               "distance_from_school", "schools_within_1km", "distance_from_public_transport",
               "transit_within_500m", "distance_from_center", "area_per_room", "deal_t"]


def web_matrix(feats: pd.DataFrame, natures: list[str]) -> pd.DataFrame:
    """Only features the browser can compute from a map click + a form (no neighbours / encodings) + one-hot nature."""
    nature = feats["cat_deal_nature"]
    one_hot = pd.get_dummies(pd.Categorical(nature, categories=natures + [OTHER]), dtype="int8")
    one_hot.columns = [cfg.NATURE_PREFIX + c for c in natures + [OTHER]]
    one_hot.index = feats.index
    return pd.concat([feats[WEB_NUMERIC], one_hot], axis=1)


def split_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split of raw deals BEFORE any statistic is computed."""
    train, test = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame | None):
    """-> (builder, train_features, test_features | None). The one function used by the pipeline and by the final refit."""
    builder = FeatureBuilder()
    train_feats = hide_year(builder.fit_transform_train(train_raw), cfg.YEAR_MASK_FRAC)   # learn the "year unknown" case
    test_feats = builder.transform(builder.filter_outliers(test_raw)) if test_raw is not None else None
    return builder, train_feats, test_feats


@timed
def run() -> None:
    df = read_csv(cfg.DATASET_CSV)
    train_raw, test_raw = split_raw(df)
    log.info("split first: %d train / %d test rows (no statistics computed yet)", len(train_raw), len(test_raw))
    builder, train_feats, test_feats = build_features(train_raw, test_raw)
    ensure_dirs(cfg.DATA_PROCESSED)
    joblib.dump({"train": train_feats, "test": test_feats, "raw_all": df}, cfg.FEATURES_FILE, compress=3)
    joblib.dump(builder, cfg.BUILDER_FILE, compress=3)
    lists = feature_lists(train_feats.columns)
    save_json({"feature_lists": lists, "natures": builder.natures, "other_label": OTHER, "target": cfg.TARGET_PPSM,
               "n_rows": len(df), "n_train": len(train_feats), "n_test": len(test_feats),
               "current_year": cfg.CURRENT_YEAR, "web_numeric": WEB_NUMERIC}, cfg.META_JSON)
    log.info("features: %d (numeric model) / %d (catboost) | train %d | test %d",
             len(lists["numeric_model"]), len(lists["catboost"]), len(train_feats), len(test_feats))
