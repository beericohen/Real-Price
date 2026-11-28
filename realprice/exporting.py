"""Step 6 - exporting: turn the trained scikit-learn trees into plain arrays that the browser can evaluate.

A gradient-boosted model is a list of decision trees. To predict, you walk down every tree to a leaf,
add up the leaf values, and add a starting value ("base"). So each model is stored as flat arrays:

    f     the feature index tested at each node (-1 for a leaf)
    t     the threshold at a node, or the leaf's value if the node is a leaf
    l, r  the index of the left / right child
    m     where a missing value (NaN, e.g. unknown year built) goes: 1 = left, 0 = right
    roots the index of each tree's first node
    base  the starting value

website/js/predictor.js reads exactly this layout. Before writing anything, the arrays are tested against
scikit-learn's own predictions - the export stops with an error if they differ.

The output is a .js file (not .json), so the site also works when index.html is opened straight from the disk.
"""
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .features import hide_year
from .preprocessing import web_matrix
from .utils import load_json

log = logging.getLogger(__name__)
PARITY_ROWS = 300   # how many test rows are used to check the export


def flatten_model(model) -> dict:
    """HistGradientBoostingRegressor -> the flat arrays described above.

    This reads scikit-learn's private attributes (_predictors, _baseline_prediction), because there is no public
    way to get at the trees. That is why the parity check in run() is important after a scikit-learn upgrade.
    """
    feature, threshold, left, right, missing_left, roots = [], [], [], [], [], []
    offset = 0   # node numbers of every tree are shifted so that all trees share one array
    for iteration in model._predictors:   # one entry per boosting round; [0] is the tree (there is one output)
        nodes = iteration[0].nodes
        is_leaf = nodes["is_leaf"].astype(bool)
        roots.append(offset)
        feature.append(np.where(is_leaf, -1, nodes["feature_idx"]))
        threshold.append(np.where(is_leaf, nodes["value"], nodes["num_threshold"]))
        missing_left.append(nodes["missing_go_to_left"].astype(int))
        left.append(np.where(is_leaf, 0, nodes["left"] + offset))
        right.append(np.where(is_leaf, 0, nodes["right"] + offset))
        offset += len(nodes)
    return {"base": float(np.ravel(model._baseline_prediction)[0]), "roots": roots,
            "f": np.concatenate(feature).astype(int), "t": np.concatenate(threshold).astype(float),
            "l": np.concatenate(left).astype(int), "r": np.concatenate(right).astype(int),
            "m": np.concatenate(missing_left).astype(int)}


def predict_flat(flat: dict, X: np.ndarray) -> np.ndarray:
    """The same algorithm as the JavaScript predictor, in numpy (for all rows at once). Used for the parity check."""
    total = np.full(len(X), flat["base"])
    rows = np.arange(len(X))
    for root in flat["roots"]:
        node = np.full(len(X), root)   # every row starts at the root of this tree
        while True:
            active = flat["f"][node] >= 0   # rows that have not reached a leaf yet
            if not active.any():
                break
            value = X[rows[active], flat["f"][node[active]]]
            go_left = np.where(np.isnan(value), flat["m"][node[active]] == 1, value <= flat["t"][node[active]])
            node[active] = np.where(go_left, flat["l"][node[active]], flat["r"][node[active]])
        total += flat["t"][node]   # at a leaf, t holds the leaf value
    return total


def to_json_friendly(flat: dict) -> dict:
    """Plain lists; thresholds are rounded to 8 significant digits to keep the file small."""
    return {"base": flat["base"], "roots": flat["roots"], "f": flat["f"].tolist(), "l": flat["l"].tolist(),
            "r": flat["r"].tolist(), "m": flat["m"].tolist(), "t": [float(f"{v:.8g}") for v in flat["t"]]}


def run() -> None:
    bundle = joblib.load(cfg.MODEL_FILE)
    meta = bundle["meta"]

    # Test rows twice: with the year built known and with it unknown (NaN), so both paths are checked
    test = joblib.load(cfg.FEATURES_FILE)["test"].head(PARITY_ROWS)
    X_frame = web_matrix(pd.concat([test, hide_year(test)]), meta["natures"])
    feature_names = X_frame.columns.tolist()
    X = X_frame.to_numpy(float)

    exported = {}
    for name, model in bundle["web_models"].items():
        flat = flatten_model(model)
        gap = np.abs(predict_flat(flat, X) - model.predict(X_frame)).max()
        if gap > 1e-3:
            raise RuntimeError(f"export mismatch for '{name}' model (max gap {gap:.5f})")
        log.info("model '%s': %d trees, %d nodes, parity gap %.2e", name, len(flat["roots"]), len(flat["f"]), gap)
        exported[name] = to_json_friendly(flat)

    metrics = load_json(cfg.METRICS_JSON)
    payload = {"feature_names": feature_names, "nature_categories": meta["natures"],
               "other_label": meta["other_label"], "nature_prefix": cfg.NATURE_PREFIX,
               "interval": metrics["interval"], "models": exported,
               "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "metrics": metrics["test"],
               "interval_coverage": metrics["interval_coverage"], "n_train": metrics["n_train"] + metrics["n_test"],
               "target": "log_price_per_sqm"}
    cfg.WEBSITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.WEBSITE_DATA_DIR / "model.js"
    out.write_text("window.RP_MODEL=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";",
                   encoding="utf-8")
    log.info("model exported -> %s (%.1f MB)", out, out.stat().st_size / 1e6)
