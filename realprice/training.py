"""Step 5 - training: HistGradientBoosting + CatBoost + LightGBM on log(price per sqm), blended into an ensemble.

* Target is log(price_per_sqm). Every metric is computed on the ORIGINAL scale: price = exp(pred) * asset_area.
* The train split is cut once more into fit / validation: the validation part drives early stopping and the
  blend-weight optimisation, the untouched TEST split is only used for the final report.
* A small separate HistGradientBoosting ("web model") is trained on browser-computable features only, because the
  static website cannot compute neighbour statistics. It is the model exported by exporting.py.
* CatBoost / LightGBM are optional imports: if one is not installed it is skipped with a warning.
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from . import config as cfg
from .features import hide_year
from .preprocessing import build_features, web_matrix
from .utils import ensure_dirs, get_logger, load_json, save_json, timed

log = get_logger(__name__)
try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


# ------------------------------------------------------------------ metrics
def evaluate(actual_total, pred_ppsm_log, area) -> dict:
    """R2 / RMSE / MAE on the original price scale (NIS) + relative-error metrics."""
    actual, pred = np.asarray(actual_total, float), np.exp(np.asarray(pred_ppsm_log, float)) * np.asarray(area, float)
    err, ape = pred - actual, np.abs(pred - actual) / actual
    la, lp = np.log(actual), np.log(pred)
    return {"r2": float(1 - (err ** 2).sum() / ((actual - actual.mean()) ** 2).sum()),
            "rmse_nis": float(np.sqrt(np.mean(err ** 2))), "mae_nis": float(np.mean(np.abs(err))),
            "r2_log_price": float(1 - ((la - lp) ** 2).sum() / ((la - la.mean()) ** 2).sum()),
            "mape": float(ape.mean()), "median_ape": float(np.median(ape)),
            "within_10pct": float((ape <= .10).mean()), "within_20pct": float((ape <= .20).mean())}


# ------------------------------------------------------------------ model factories
def make_hgb(quantile: float | None = None, categorical_features=None, **overrides) -> HistGradientBoostingRegressor:
    """HGB with the production settings; categorical columns use sklearn's native handling (no manual one-hot)."""
    params = {**cfg.HGB_PARAMS, **overrides}
    loss = {"loss": "quantile", "quantile": quantile} if quantile else {"loss": "squared_error"}
    return HistGradientBoostingRegressor(categorical_features=categorical_features, random_state=cfg.RANDOM_STATE,
                                         **params, **loss)


def as_lgbm_frame(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """LightGBM wants category dtype for categorical columns (codes must be consistent between fit and predict)."""
    X = X.copy()
    for col in categorical:
        X[col] = X[col].astype("category")
    return X


class ModelSet:
    """Fits the three main models on one (fit, validation) pair and predicts log(ppsm) for any feature frame."""

    def __init__(self, lists: dict):
        self.lists, self.models = lists, {}
        self.cat_numeric = [c for c in ("cluster", "nature_code") if c in lists["numeric_model"]]

    def fit(self, X_fit, y_fit, X_val, y_val) -> "ModelSet":
        cols = self.lists["numeric_model"]
        self.models["HistGradientBoosting"] = make_hgb(categorical_features=self.cat_numeric).fit(X_fit[cols], y_fit)
        if CatBoostRegressor is not None:
            cb = CatBoostRegressor(**cfg.CATBOOST_PARAMS, random_seed=cfg.RANDOM_STATE)
            cb.fit(X_fit[self.lists["catboost"]], y_fit, cat_features=self.lists["catboost_cat_features"],
                   eval_set=(X_val[self.lists["catboost"]], y_val), use_best_model=True)
            self.models["CatBoost"] = cb
        else:
            log.warning("catboost is not installed -> skipped (pip install catboost)")
        if LGBMRegressor is not None:
            import lightgbm as lgb
            lg = LGBMRegressor(**cfg.LGBM_PARAMS, random_state=cfg.RANDOM_STATE)
            lg.fit(as_lgbm_frame(X_fit[cols], self.cat_numeric), y_fit, eval_set=[(as_lgbm_frame(X_val[cols], self.cat_numeric), y_val)],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            self.models["LightGBM"] = lg
        else:
            log.warning("lightgbm is not installed -> skipped (pip install lightgbm)")
        return self

    def predict(self, X: pd.DataFrame) -> dict:
        cols = self.lists["numeric_model"]
        out = {}
        for name, model in self.models.items():
            if name == "CatBoost":
                out[name] = model.predict(X[self.lists["catboost"]])
            elif name == "LightGBM":
                out[name] = model.predict(as_lgbm_frame(X[cols], self.cat_numeric))
            else:
                out[name] = model.predict(X[cols])
        return out


def blend_weights(val_preds: dict, y_val) -> dict:
    """Non-negative least squares on the validation set, normalised to sum 1 (a convex combination)."""
    names = list(val_preds)
    w, _ = nnls(np.column_stack([val_preds[n] for n in names]), np.asarray(y_val, float))
    w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1 / len(names))
    return dict(zip(names, w.round(4).tolist()))


def blend(preds: dict, weights: dict) -> np.ndarray:
    return sum(weights[n] * preds[n] for n in weights)


# ------------------------------------------------------------------ website (browser) model
def train_web_models(Xw_train, y_train, Xw_val, y_val) -> dict:
    """median + 10% / 90% quantile HGB on browser-computable features (plain one-hot, exportable tree layout)."""
    shared = dict(categorical_features=None, early_stopping=False, max_iter=cfg.MAX_BOOST_ITER, learning_rate=0.08,
                  min_samples_leaf=20, l2_regularization=1.0, max_leaf_nodes=63)
    models = {"median": make_hgb(**shared).fit(Xw_train, y_train)}
    for name, q in cfg.QUANTILES.items():
        models[name] = make_hgb(q, **shared).fit(Xw_train, y_train)
    return models


def _save_diagnostics(model, X_test, y_test, actual, pred_ppsm, area) -> None:
    ensure_dirs(cfg.TRAINING_DIR)
    pred = np.exp(pred_ppsm) * area
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(actual / 1e6, pred / 1e6, gridsize=60, bins="log", xscale="log", yscale="log", cmap="viridis", mincnt=1)
    lims = [actual.min() / 1e6, actual.max() / 1e6]
    ax.plot(lims, lims, color="red", lw=1)
    ax.set(title="Predicted vs actual price (test, ensemble)", xlabel="actual (M NIS)", ylabel="predicted (M NIS)")
    fig.colorbar(hb)
    fig.savefig(cfg.TRAINING_DIR / "predicted_vs_actual.png", dpi=110)
    plt.close(fig)
    sample = X_test.sample(min(len(X_test), cfg.PERMUTATION_ROWS), random_state=cfg.RANDOM_STATE)
    imp = permutation_importance(model, sample, y_test.loc[sample.index], n_repeats=3, random_state=cfg.RANDOM_STATE,
                                 scoring="neg_mean_squared_error", n_jobs=-1)
    series = pd.Series(imp.importances_mean, index=X_test.columns).nlargest(25).sort_values()
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(series.index, series.values, color="#2457d6")
    ax.set(title="Permutation importance - HistGradientBoosting (MSE of log ppsm)")
    fig.tight_layout()
    fig.savefig(cfg.TRAINING_DIR / "feature_importance.png", dpi=110)
    plt.close(fig)


def _split_fit_val(frame: pd.DataFrame):
    return train_test_split(frame, test_size=cfg.VAL_FRACTION, random_state=cfg.RANDOM_STATE)


@timed
def run() -> dict:
    meta, data = load_json(cfg.META_JSON), joblib.load(cfg.FEATURES_FILE)
    train, test, lists = data["train"], data["test"], meta["feature_lists"]
    target = meta["target"]

    # ---- honest evaluation: fit on train(fit part), validate/blend on train(val part), report on TEST
    fit_df, val_df = _split_fit_val(train)
    mset = ModelSet(lists).fit(fit_df, fit_df[target], val_df, val_df[target])
    val_pred, test_pred = mset.predict(val_df), mset.predict(test)
    weights = blend_weights(val_pred, val_df[target])
    test_pred["Ensemble (weighted blend)"] = blend(test_pred, weights)
    rows = [{"model": name, **evaluate(test["deal_amount"], p, test["asset_area_raw"])} for name, p in test_pred.items()]

    # ---- same test rows, but pretending year_built is unknown (typical for Yad2 listings)
    test_blind = hide_year(test)
    blind = mset.predict(test_blind)
    blind["Ensemble (weighted blend)"] = blend(blind, weights)
    rows += [{"model": f"{name} - year_built UNKNOWN", **evaluate(test["deal_amount"], p, test["asset_area_raw"])}
             for name, p in blind.items()]

    # ---- baseline = the previous setup (log(deal_amount) target, no new features) for an honest before/after
    rows.insert(0, {"model": "Baseline (old: basic features, log price target)", **baseline(train, test, meta)})
    comparison = pd.DataFrame(rows)
    ensure_dirs(cfg.TRAINING_DIR)
    comparison.to_csv(cfg.TRAINING_DIR / "model_comparison.csv", index=False)
    log.info("TEST performance (original NIS scale):\n%s",
             comparison[["model", "r2", "rmse_nis", "mae_nis", "median_ape", "within_20pct"]].round(4).to_string(index=False))
    log.info("blend weights: %s", weights)

    # ---- website model (browser-computable features only)
    Xw_tr, Xw_te = web_matrix(train, meta["natures"]), web_matrix(test, meta["natures"])
    web_models = train_web_models(Xw_tr, train[target], None, None)
    web_pred = {k: m.predict(Xw_te) for k, m in web_models.items()}
    low, high = np.minimum(web_pred["low"], web_pred["high"]), np.maximum(web_pred["low"], web_pred["high"])
    web_metrics = evaluate(test["deal_amount"], web_pred["median"], test["asset_area_raw"])
    web_blind = evaluate(test["deal_amount"], web_models["median"].predict(web_matrix(test_blind, meta["natures"])),
                         test["asset_area_raw"])
    coverage = float(((test[target] >= low) & (test[target] <= high)).mean())

    metrics = {"test": web_metrics, "test_year_unknown": web_blind, "interval_coverage": coverage, "interval": [cfg.QUANTILES["low"], cfg.QUANTILES["high"]],
               "ensemble_test": evaluate(test["deal_amount"], test_pred["Ensemble (weighted blend)"], test["asset_area_raw"]),
               "blend_weights": weights, "n_train": len(train), "n_test": len(test),
               "comparison": comparison.round(5).to_dict("records")}
    _save_diagnostics(mset.models["HistGradientBoosting"], test[lists["numeric_model"]], test[target],
                      test["deal_amount"].to_numpy(), test_pred["HistGradientBoosting"], test["asset_area_raw"].to_numpy())

    # ---- production: rebuild features on ALL rows (train statistics now include the test rows) and refit
    final_sets = {"main": mset, "weights": weights, "builder": joblib.load(cfg.BUILDER_FILE)}
    if cfg.REFIT_ON_ALL:
        builder, all_feats, _ = build_features(data["raw_all"], None)
        f_df, v_df = _split_fit_val(all_feats)
        final_sets = {"main": ModelSet(lists).fit(f_df, f_df[target], v_df, v_df[target]), "weights": weights,
                      "builder": builder}
        Xw_all = web_matrix(all_feats, meta["natures"])
        web_models = train_web_models(Xw_all, all_feats[target], None, None)
    ensure_dirs(cfg.MODELS_DIR)
    joblib.dump({**final_sets, "web_models": web_models, "meta": meta}, cfg.MODEL_FILE)
    save_json(metrics, cfg.METRICS_JSON)
    return metrics


def baseline(train: pd.DataFrame, test: pd.DataFrame, meta: dict) -> dict:
    """Old recipe on the same cleaned rows: min_samples_leaf=80, target log(total price), website-style features."""
    Xtr, Xte = web_matrix(train, meta["natures"]), web_matrix(test, meta["natures"])
    old = HistGradientBoostingRegressor(max_iter=cfg.MAX_BOOST_ITER, min_samples_leaf=80, early_stopping=True,
                                        random_state=cfg.RANDOM_STATE).fit(Xtr, np.log(train["deal_amount"]))
    pred_total = np.exp(old.predict(Xte))
    return evaluate(test["deal_amount"], np.log(pred_total / test["asset_area_raw"]), test["asset_area_raw"])
