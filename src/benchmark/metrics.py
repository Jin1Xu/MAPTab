import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score, f1_score

from hyperimpute.plugins.utils.metrics import RMSE
from utils.runtime import compute_logreg_auroc


def ws_score(imputed: pd.DataFrame, ground: pd.DataFrame, continuous_indices) -> float:
    if not continuous_indices:
        return float("nan")
    res = 0.0
    for col in continuous_indices:
        res += wasserstein_distance(
            np.asarray(ground)[:, col], np.asarray(imputed)[:, col]
        )
    return res


def continuous_rmse(
    imputed: pd.DataFrame, ground: pd.DataFrame, mask: pd.DataFrame, continuous_indices
) -> float:
    if not continuous_indices:
        return float("nan")
    return RMSE(
        np.asarray(imputed)[:, continuous_indices],
        np.asarray(ground)[:, continuous_indices],
        np.asarray(mask)[:, continuous_indices],
    )


def _project_to_valid_categories(
    values: np.ndarray, valid_values: np.ndarray
) -> np.ndarray:
    if valid_values.size == 0:
        return values.astype(np.float32, copy=False)
    values = values.astype(np.float32, copy=False).reshape(-1, 1)
    valid_values = np.asarray(valid_values, dtype=np.float32).reshape(1, -1)
    nearest_idx = np.argmin(np.abs(values - valid_values), axis=1)
    return valid_values.reshape(-1)[nearest_idx]


def categorical_scores(
    imputed: pd.DataFrame, ground: pd.DataFrame, mask: pd.DataFrame, categorical_indices
):
    if not categorical_indices:
        return float("nan"), float("nan")

    acc_scores = []
    f1_scores = []
    imputed_np = np.asarray(imputed)
    ground_np = np.asarray(ground)
    mask_np = np.asarray(mask)

    for col in categorical_indices:
        missing_mask = mask_np[:, col] > 0
        if not np.any(missing_mask):
            continue

        y_true = ground_np[missing_mask, col]
        valid_values = np.unique(ground_np[:, col])
        y_pred = _project_to_valid_categories(
            imputed_np[missing_mask, col], valid_values
        )

        acc_scores.append(float(accuracy_score(y_true, y_pred)))
        f1_scores.append(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
        )

    if not acc_scores:
        return float("nan"), float("nan")
    return float(np.mean(acc_scores)), float(np.mean(f1_scores))


def compute_auroc(
    train_imputed: pd.DataFrame,
    y_train: pd.Series,
    eval_imputed: pd.DataFrame,
    y_eval: pd.Series,
) -> float:
    return compute_logreg_auroc(train_imputed, y_train, eval_imputed, y_eval)
