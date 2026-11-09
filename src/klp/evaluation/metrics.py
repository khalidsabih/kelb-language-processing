from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, target_names: list[str] | None = None) -> dict[str, float]:
    """Per-target MAE/RMSE/R2 plus their unweighted means across targets.

    Keys: ``mae``, ``rmse``, ``r2`` (means) and ``<metric>_<target>``.
    """
    y_true_arr = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)
    if target_names is None:
        target_names = list(getattr(y_true, "columns", range(y_true_arr.shape[1])))

    metrics: dict[str, float] = {}
    for i, name in enumerate(target_names):
        t, p = y_true_arr[:, i], y_pred_arr[:, i]
        metrics[f"mae_{name}"] = float(mean_absolute_error(t, p))
        metrics[f"rmse_{name}"] = float(np.sqrt(mean_squared_error(t, p)))
        metrics[f"r2_{name}"] = float(r2_score(t, p))

    for key in ("mae", "rmse", "r2"):
        metrics[key] = float(np.mean([metrics[f"{key}_{name}"] for name in target_names]))
    return metrics
