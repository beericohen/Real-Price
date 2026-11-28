"""Step 5 - training and evaluation.

Two groups of models are trained on the same target, log(price per sqm):

1. MAIN MODELS: HistGradientBoosting + CatBoost + LightGBM, blended into an ensemble. They use all features
   (neighbour prices, target encodings ...), so they are the most accurate. CatBoost and LightGBM are optional:
   if a package is not installed, that model is skipped with a warning.
2. WEBSITE MODEL: three small HistGradientBoosting models (median, 10th and 90th percentile) that use only the
   features the browser can compute. This is the model that is exported to the website (step 6).

All errors are reported on the ORIGINAL price scale (NIS):  price = exp(prediction) * asset_area.

The train split is cut once more into "fit" and "validation". The validation part is used for early stopping and
for the blend weights. The test split is never used for any decision, it is only used for the final report.
"""
import logging

import joblib
import matplotlib

matplotlib.use("Agg")   # draw into files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from . import config as cfg
from .features import CATEGORICAL_NUMERIC, hide_year
from .preprocessing import build_features, web_matrix
from .utils import load_json, save_json

log = logging.getLogger(__name__)
plt.rcParams.update(cfg.PLOT_STYLE)   # same chart style as the EDA charts, whichever step runs first

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

ENSEMBLE = "Ensemble (weighted blend)"


# ------------------------------------------------------------------ metrics
def evaluate(actual_price, predicted_log_ppsm, area) -> dict:
    """Error measures on the original price scale. `predicted_log_ppsm` is what the models output."""
    actual = np.asarray(actual_price, float)
    predicted = np.exp(np.asarray(predicted_log_ppsm, float)) * np.asarray(area, float)
    error = predicted - actual
    pct_error = np.abs(error) / actual
    log_actual, log_predicted = np.log(actual), np.log(predicted)
    return {
        "r2": float(1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum()),
        "rmse_nis": float(np.sqrt(np.mean(error ** 2))),
        "mae_nis": float(np.mean(np.abs(error))),
        "r2_log_price": float(1 - ((log_actual - log_predicted) ** 2).sum() / ((log_actual - log_actual.mean()) ** 2).sum()),
        "mape": float(pct_error.mean()),
        "median_ape": float(np.median(pct_error)),
        "within_10pct": float((pct_error <= .10).mean()),
        "within_20pct": float((pct_error <= .20).mean()),
    }


# ------------------------------------------------------------------ main models
def make_hgb(quantile=None, categorical_features=None, **overrides) -> HistGradientBoostingRegressor:
    """A HistGradientBoosting model with the project's settings (config.HGB_PARAMS).
    With `quantile` it predicts that percentile (e.g. 0.9) instead of the mean."""
    params = {**cfg.HGB_PARAMS, **overrides}
    if quantile:
        loss = {"loss": "quantile", "quantile": quantile}
    else:
        loss = {"loss": "squared_error"}
    return HistGradientBoostingRegressor(categorical_features=categorical_features, random_state=cfg.RANDOM_STATE,
                                         **params, **loss)


def model_input(name: str, X: pd.DataFrame, lists: dict) -> pd.DataFrame:
    """The columns, in the format, that model `name` expects."""
    if name == "CatBoost":
        return X[lists["catboost"]]   # includes the text keys that CatBoost encodes itself
    X = X[lists["numeric_model"]]
    if name == "LightGBM":
        X = X.copy()
        for col in categorical_columns(lists):   # LightGBM wants the "category" type for categorical columns
            X[col] = X[col].astype("category")
    return X


def categorical_columns(lists: dict) -> list[str]:
    return [c for c in CATEGORICAL_NUMERIC if c in lists["numeric_model"]]


def fit_main_models(lists, X_fit, y_fit, X_val, y_val) -> dict:
    """Train the main models on the fit part. The validation part tells them when to stop adding trees."""
    models = {}

    hgb = make_hgb(categorical_features=categorical_columns(lists))
    models["HistGradientBoosting"] = hgb.fit(model_input("HistGradientBoosting", X_fit, lists), y_fit)

    if CatBoostRegressor is not None:
        catboost = CatBoostRegressor(**cfg.CATBOOST_PARAMS, random_seed=cfg.RANDOM_STATE)
        catboost.fit(model_input("CatBoost", X_fit, lists), y_fit, cat_features=lists["catboost_cat_features"],
                     eval_set=(model_input("CatBoost", X_val, lists), y_val), use_best_model=True)
        models["CatBoost"] = catboost
    else:
        log.warning("catboost is not installed -> skipped (pip install catboost)")

    if lgb is not None:
        lightgbm = lgb.LGBMRegressor(**cfg.LGBM_PARAMS, random_state=cfg.RANDOM_STATE)
        lightgbm.fit(model_input("LightGBM", X_fit, lists), y_fit,
                     eval_set=[(model_input("LightGBM", X_val, lists), y_val)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
        models["LightGBM"] = lightgbm
    else:
        log.warning("lightgbm is not installed -> skipped (pip install lightgbm)")
    return models


def predict_main_models(models: dict, lists: dict, X: pd.DataFrame) -> dict:
    """{model name: predicted log(price per sqm)}"""
    return {name: model.predict(model_input(name, X, lists)) for name, model in models.items()}


def blend_weights(val_preds: dict, y_val) -> dict:
    """How much to trust each model. Non-negative least squares on the validation set, scaled to add up to 1."""
    names = list(val_preds)
    weights, _ = nnls(np.column_stack([val_preds[n] for n in names]), np.asarray(y_val, float))
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.full(len(names), 1 / len(names))
    return dict(zip(names, weights.round(4).tolist()))


def blend(preds: dict, weights: dict) -> np.ndarray:
    return sum(weights[name] * preds[name] for name in weights)


def split_fit_val(frame: pd.DataFrame):
    return train_test_split(frame, test_size=cfg.VAL_FRACTION, random_state=cfg.RANDOM_STATE)


def baseline_metrics(train, test, natures) -> dict:
    """A deliberately simple model to compare against: only the website features, and it predicts log(TOTAL price)
    instead of log(price per sqm). It shows how much the extra features and the per-sqm target really help."""
    X_train, X_test = web_matrix(train, natures), web_matrix(test, natures)
    model = HistGradientBoostingRegressor(max_iter=cfg.WEB_MAX_ITER, min_samples_leaf=80, early_stopping=True,
                                          random_state=cfg.RANDOM_STATE)
    model.fit(X_train, np.log(train["deal_amount"]))
    predicted_price = np.exp(model.predict(X_test))
    area = test["asset_area_raw"]
    return evaluate(test["deal_amount"], np.log(predicted_price / area), area)


# ------------------------------------------------------------------ website model
def train_web_models(X, y) -> dict:
    """Median + 10th + 90th percentile models on the browser-computable features.
    Fixed number of trees (no early stopping): the browser has to evaluate every tree."""
    settings = dict(early_stopping=False, max_iter=cfg.WEB_MAX_ITER, learning_rate=0.08,
                    min_samples_leaf=20, l2_regularization=1.0, max_leaf_nodes=63)
    models = {"median": make_hgb(**settings).fit(X, y)}
    for name, quantile in cfg.QUANTILES.items():
        models[name] = make_hgb(quantile, **settings).fit(X, y)
    return models


# ------------------------------------------------------------------ report pictures
def save_diagnostic_plots(model, X_test, y_test, actual_price, predicted_log_ppsm, area) -> None:
    """Two pictures: predicted vs actual price, and which features the HistGradientBoosting model relies on."""
    cfg.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    predicted_price = np.exp(predicted_log_ppsm) * area

    fig, ax = plt.subplots(figsize=(6, 6))
    hexes = ax.hexbin(actual_price / 1e6, predicted_price / 1e6, gridsize=60, bins="log", xscale="log", yscale="log",
                      cmap="viridis", mincnt=1)
    limits = [actual_price.min() / 1e6, actual_price.max() / 1e6]
    ax.plot(limits, limits, color="red", lw=1)   # the perfect-prediction line
    ax.set(title="Predicted vs actual price (test, ensemble)", xlabel="actual (M NIS)", ylabel="predicted (M NIS)")
    fig.colorbar(hexes)
    fig.savefig(cfg.TRAINING_DIR / "predicted_vs_actual.png", dpi=110)
    plt.close(fig)

    # Permutation importance: shuffle one column and see how much the error grows
    sample = X_test.sample(min(len(X_test), cfg.PERMUTATION_ROWS), random_state=cfg.RANDOM_STATE)
    importance = permutation_importance(model, sample, y_test.loc[sample.index], n_repeats=3,
                                        random_state=cfg.RANDOM_STATE, scoring="neg_mean_squared_error", n_jobs=-1)
    top = pd.Series(importance.importances_mean, index=X_test.columns).nlargest(25).sort_values()
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top.index, top.values, color="#2457d6")
    ax.set(title="Permutation importance - HistGradientBoosting (MSE of log ppsm)")
    fig.tight_layout()
    fig.savefig(cfg.TRAINING_DIR / "feature_importance.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ the three stages of run()
def evaluate_main_models(train, test, test_blind, lists, meta):
    """Honest evaluation: fit on train, choose blend weights on validation, report on test.
    Returns (models, weights, test predictions, comparison table)."""
    target = meta["target"]
    fit_part, val_part = split_fit_val(train)
    models = fit_main_models(lists, fit_part, fit_part[target], val_part, val_part[target])

    val_pred = predict_main_models(models, lists, val_part)
    weights = blend_weights(val_pred, val_part[target])

    test_pred = predict_main_models(models, lists, test)
    test_pred[ENSEMBLE] = blend(test_pred, weights)
    # The same test deals again, but pretending that the year built is unknown (as on most listings)
    blind_pred = predict_main_models(models, lists, test_blind)
    blind_pred[ENSEMBLE] = blend(blind_pred, weights)

    def score(pred):
        return evaluate(test["deal_amount"], pred, test["asset_area_raw"])

    rows = [{"model": "Baseline (basic features, log total-price target)",
             **baseline_metrics(train, test, meta["natures"])}]
    rows += [{"model": name, **score(pred)} for name, pred in test_pred.items()]
    rows += [{"model": f"{name} - year_built UNKNOWN", **score(pred)} for name, pred in blind_pred.items()]
    comparison = pd.DataFrame(rows)

    cfg.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(cfg.TRAINING_DIR / "model_comparison.csv", index=False)
    log.info("TEST performance (original NIS scale):\n%s",
             comparison[["model", "r2", "rmse_nis", "mae_nis", "median_ape", "within_20pct"]].round(4).to_string(index=False))
    log.info("blend weights: %s", weights)
    return models, weights, test_pred, comparison


def evaluate_web_model(web_models, test, test_blind, meta):
    """Score the website model on the test deals. Returns (metrics, metrics with unknown year, interval coverage)."""
    target = meta["target"]
    predictions = {name: model.predict(web_matrix(test, meta["natures"])) for name, model in web_models.items()}
    low = np.minimum(predictions["low"], predictions["high"])
    high = np.maximum(predictions["low"], predictions["high"])

    def score(pred):
        return evaluate(test["deal_amount"], pred, test["asset_area_raw"])

    metrics = score(predictions["median"])
    metrics_blind = score(web_models["median"].predict(web_matrix(test_blind, meta["natures"])))
    # Coverage: how often the real price lies inside the 10th-90th percentile range (should be about 80%)
    coverage = float(((test[target] >= low) & (test[target] <= high)).mean())
    return metrics, metrics_blind, coverage


def refit_on_all_data(raw_all, lists, weights, meta):
    """Production models: the features and models are rebuilt on ALL deals (train + test), because more data
    gives a better model. The numbers reported above come from the honest evaluation, before this step."""
    target = meta["target"]
    builder, all_feats, _ = build_features(raw_all, None)
    fit_part, val_part = split_fit_val(all_feats)
    main_models = fit_main_models(lists, fit_part, fit_part[target], val_part, val_part[target])
    web_models = train_web_models(web_matrix(all_feats, meta["natures"]), all_feats[target])
    return {"main": main_models, "weights": weights, "builder": builder, "web_models": web_models}


def run() -> dict:
    meta, data = load_json(cfg.META_JSON), joblib.load(cfg.FEATURES_FILE)
    train, test, lists = data["train"], data["test"], meta["feature_lists"]
    target = meta["target"]
    test_blind = hide_year(test)   # test rows with the year built hidden

    # ---- 1. main models: honest evaluation
    models, weights, test_pred, comparison = evaluate_main_models(train, test, test_blind, lists, meta)

    # ---- 2. website model: trained on train, scored on test
    web_models = train_web_models(web_matrix(train, meta["natures"]), train[target])
    web_metrics, web_metrics_blind, coverage = evaluate_web_model(web_models, test, test_blind, meta)

    metrics = {"test": web_metrics, "test_year_unknown": web_metrics_blind, "interval_coverage": coverage,
               "interval": [cfg.QUANTILES["low"], cfg.QUANTILES["high"]],
               "ensemble_test": evaluate(test["deal_amount"], test_pred[ENSEMBLE], test["asset_area_raw"]),
               "blend_weights": weights, "n_train": len(train), "n_test": len(test),
               "comparison": comparison.round(5).to_dict("records")}
    save_diagnostic_plots(models["HistGradientBoosting"], test[lists["numeric_model"]], test[target],
                          test["deal_amount"].to_numpy(), test_pred["HistGradientBoosting"],
                          test["asset_area_raw"].to_numpy())

    # ---- 3. models for production
    if cfg.REFIT_ON_ALL:
        final = refit_on_all_data(data["raw_all"], lists, weights, meta)
    else:
        final = {"main": models, "weights": weights, "builder": joblib.load(cfg.BUILDER_FILE), "web_models": web_models}
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({**final, "meta": meta}, cfg.MODEL_FILE)
    save_json(metrics, cfg.METRICS_JSON)
    return metrics
