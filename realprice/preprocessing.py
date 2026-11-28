"""Step 4 - preprocessing: split FIRST, then build the features from the train split only (see features.py).

Order of operations (this order is what prevents data leakage):
  1. random train / test split of the cleaned + enriched deals
  2. fit everything on TRAIN: outlier limits, KMeans, per-gush numbers, price trends, comparables, target encoding
  3. train rows: out-of-fold target encoding + comparables from earlier deals;  test rows: train statistics only

Two views of the same rows are produced:
  * the full feature table (neighbour prices, encodings ...) for HistGradientBoosting / CatBoost / LightGBM
  * a small matrix (`web_matrix`) with only the features that the browser can compute from a map click and a form;
    the model exported to the website uses it
"""
import logging

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from .features import OTHER, FeatureBuilder, feature_lists, hide_year
from .utils import read_csv, save_json

log = logging.getLogger(__name__)

# The inputs of the website model. website/js/predictor.js computes exactly these (plus the one-hot deal type).
WEB_NUMERIC = ["lat", "long", "year_built", "age", "asset_area", "room_num", "deal_day", "deal_month", "deal_year",
               "distance_from_school", "schools_within_1km", "distance_from_public_transport",
               "transit_within_500m", "distance_from_center", "area_per_room", "deal_t"]


def web_matrix(feats: pd.DataFrame, natures: list[str]) -> pd.DataFrame:
    """The website model's input: WEB_NUMERIC columns + one 0/1 column per deal type ("nature__...")."""
    categories = natures + [OTHER]
    one_hot = pd.get_dummies(pd.Categorical(feats["cat_deal_nature"], categories=categories), dtype="int8")
    one_hot.columns = [cfg.NATURE_PREFIX + c for c in categories]
    one_hot.index = feats.index
    return pd.concat([feats[WEB_NUMERIC], one_hot], axis=1)


def split_raw(df: pd.DataFrame):
    """Random train / test split of the deals, BEFORE any statistic is computed."""
    train, test = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame | None):
    """Fit a FeatureBuilder on `train_raw` -> (builder, train features, test features or None).
    Used for the honest evaluation (with a test set) and for the final refit on all data (without one)."""
    builder = FeatureBuilder()
    train_feats = builder.fit_transform_train(train_raw)
    # Hide the year in some train rows, so that the models also learn the "year unknown" case
    train_feats = hide_year(train_feats, cfg.YEAR_MASK_FRAC)
    test_feats = None
    if test_raw is not None:
        test_feats = builder.transform(builder.filter_outliers(test_raw))
    return builder, train_feats, test_feats


def run() -> None:
    df = read_csv(cfg.DATASET_CSV)
    train_raw, test_raw = split_raw(df)
    log.info("split first: %d train / %d test rows (no statistics computed yet)", len(train_raw), len(test_raw))

    builder, train_feats, test_feats = build_features(train_raw, test_raw)

    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump({"train": train_feats, "test": test_feats, "raw_all": df}, cfg.FEATURES_FILE, compress=3)
    joblib.dump(builder, cfg.BUILDER_FILE, compress=3)

    lists = feature_lists(train_feats.columns)
    save_json({"feature_lists": lists, "natures": builder.natures, "other_label": OTHER, "target": cfg.TARGET,
               "n_rows": len(df), "n_train": len(train_feats), "n_test": len(test_feats),
               "current_year": cfg.CURRENT_YEAR, "web_numeric": WEB_NUMERIC}, cfg.META_JSON)
    log.info("features: %d (numeric model) / %d (catboost) | train %d | test %d",
             len(lists["numeric_model"]), len(lists["catboost"]), len(train_feats), len(test_feats))
