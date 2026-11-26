"""Step 6 - exporting: turn the fitted scikit-learn trees into plain arrays the browser can evaluate.

Each model becomes five flat arrays (feature, threshold-or-leaf-value, left, right, tree roots) plus the
baseline score. Prediction in JavaScript is then: walk every tree, add the leaf values, add the baseline.
The export is verified against scikit-learn's own predictions before it is written.
The output is a .js file (not .json) so the site also works when index.html is opened straight from disk.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .features import hide_year
from .preprocessing import web_matrix
from .utils import ensure_dirs, get_logger, load_json, timed

log = get_logger(__name__)
PARITY_ROWS = 300


def flatten_model(model) -> dict:
    """HistGradientBoosting -> {base, roots, f, t, l, r}. Leaves have f = -1 and store their value in t."""
    feature, threshold, left, right, roots, offset, miss = [], [], [], [], [], 0, []
    for iteration in model._predictors:            # one predictor per boosting iteration (regression)
        nodes = iteration[0].nodes
        leaf = nodes["is_leaf"].astype(bool)
        roots.append(offset)
        feature.append(np.where(leaf, -1, nodes["feature_idx"]))
        threshold.append(np.where(leaf, nodes["value"], nodes["num_threshold"]))
        miss.append(nodes["missing_go_to_left"].astype(int))      # where an UNKNOWN value (NaN) goes at this split
        left.append(np.where(leaf, 0, nodes["left"] + offset))
        right.append(np.where(leaf, 0, nodes["right"] + offset))
        offset += len(nodes)
    return {"base": float(np.ravel(model._baseline_prediction)[0]), "roots": roots,
            "f": np.concatenate(feature).astype(int), "t": np.concatenate(threshold).astype(float),
            "l": np.concatenate(left).astype(int), "r": np.concatenate(right).astype(int),
            "m": np.concatenate(miss).astype(int)}


def predict_flat(flat: dict, X: np.ndarray) -> np.ndarray:
    """Reference implementation of the JavaScript evaluator (vectorised over rows) - used for the parity check."""
    total = np.full(len(X), flat["base"])
    rows = np.arange(len(X))
    for root in flat["roots"]:
        node = np.full(len(X), root)
        while True:
            active = flat["f"][node] >= 0
            if not active.any():
                break
            feat = flat["f"][node[active]]
            value = X[rows[active], feat]
            go_left = np.where(np.isnan(value), flat["m"][node[active]] == 1, value <= flat["t"][node[active]])
            node[active] = np.where(go_left, flat["l"][node[active]], flat["r"][node[active]])
        total += flat["t"][node]
    return total


def _compact(flat: dict) -> dict:
    """JSON-friendly + size-reduced version (rounded floats)."""
    return {"base": flat["base"], "roots": flat["roots"], "f": flat["f"].tolist(), "l": flat["l"].tolist(),
            "r": flat["r"].tolist(), "m": flat["m"].tolist(), "t": [float(f"{v:.8g}") for v in flat["t"]]}


@timed
def run() -> None:
    bundle = joblib.load(cfg.MODEL_FILE)
    meta = bundle["meta"]
    test = joblib.load(cfg.FEATURES_FILE)["test"].head(PARITY_ROWS)
    X_df = web_matrix(pd.concat([test, hide_year(test)]), meta["natures"])     # known AND unknown year_built
    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(float)

    exported = {}
    for name, model in bundle["web_models"].items():
        flat = flatten_model(model)
        gap = np.abs(predict_flat(flat, X) - model.predict(X_df)).max()
        if gap > 1e-3:
            raise RuntimeError(f"export mismatch for '{name}' model (max gap {gap:.5f})")
        log.info("model '%s': %d trees, %d nodes, parity gap %.2e", name, len(flat["roots"]), len(flat["f"]), gap)
        exported[name] = _compact(flat)

    metrics = load_json(cfg.METRICS_JSON)
    payload = {"feature_names": feature_names, "nature_categories": meta["natures"],
               "other_label": meta["other_label"], "nature_prefix": cfg.NATURE_PREFIX,
               "interval": metrics["interval"], "models": exported,
               "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "metrics": metrics["test"],
               "interval_coverage": metrics["interval_coverage"], "n_train": metrics["n_train"] + metrics["n_test"], "target": "log_price_per_sqm"}
    ensure_dirs(cfg.WEBSITE_DATA_DIR)
    out = cfg.WEBSITE_DATA_DIR / "model.js"
    out.write_text("window.RP_MODEL=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";",
                   encoding="utf-8")
    log.info("model exported -> %s (%.1f MB)", out, out.stat().st_size / 1e6)
