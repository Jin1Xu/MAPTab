import copy
from time import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.preprocessing import infer_feature_type_indices_from_names
from .metrics import (
    categorical_scores,
    compute_auroc,
    continuous_rmse,
    ws_score,
)
from .scenarios import simulate_scenarios, split_train_test
from utils.runtime import is_classification


def _ensure_dataframe(values, reference: pd.DataFrame) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.copy()
    if isinstance(values, np.ndarray):
        return pd.DataFrame(values, columns=reference.columns, index=reference.index)
    if torch.is_tensor(values):
        return pd.DataFrame(
            values.detach().cpu().numpy(),
            columns=reference.columns,
            index=reference.index,
        )
    return pd.DataFrame(
        np.asarray(values), columns=reference.columns, index=reference.index
    )


def benchmark_model(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_train_miss: pd.DataFrame,
    train_mask: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_test_miss: pd.DataFrame,
    test_mask: pd.DataFrame,
    problem_type: str,
):
    start = time()
    print(
        f"[Benchmark] Start model='{model_name}', train_shape={x_train_miss.shape}, "
        f"test_shape={x_test_miss.shape}"
    )
    if not hasattr(model, "fit"):
        raise AttributeError(
            f"Imputer '{type(model).__name__}' does not implement fit(...), cannot run OOS evaluation."
        )
    if not hasattr(model, "transform"):
        raise AttributeError(
            f"Imputer '{type(model).__name__}' does not implement transform(...), cannot run OOS evaluation."
        )
    fitted_model = model.fit(x_train_miss.copy())
    fitted_model = model if fitted_model is None else fitted_model
    imputed_train = _ensure_dataframe(
        fitted_model.transform(x_train_miss.copy()), x_train_miss
    )
    imputed_test = _ensure_dataframe(
        fitted_model.transform(x_test_miss.copy()), x_test_miss
    )

    categorical_indices, continuous_indices, unknown_indices = (
        infer_feature_type_indices_from_names(x_train.columns)
    )
    if unknown_indices:
        unknown_cols = [str(x_train.columns[idx]) for idx in unknown_indices]
        raise ValueError(
            f"Unexpected feature prefixes during benchmark scoring: {unknown_cols}"
        )

    train_cat_acc, train_cat_f1 = categorical_scores(
        imputed_train, x_train, train_mask, categorical_indices
    )
    test_cat_acc, test_cat_f1 = categorical_scores(
        imputed_test, x_test, test_mask, categorical_indices
    )

    is_metrics = {
        "rmse": continuous_rmse(imputed_train, x_train, train_mask, continuous_indices),
        "wasserstein": ws_score(imputed_train, x_train, continuous_indices),
        "categorical_accuracy": train_cat_acc,
        "categorical_macro_f1": train_cat_f1,
        "auroc": (
            compute_auroc(imputed_train, y_train, imputed_train, y_train)
            if is_classification(problem_type)
            else None
        ),
    }
    oos_metrics = {
        "rmse": continuous_rmse(imputed_test, x_test, test_mask, continuous_indices),
        "wasserstein": ws_score(imputed_test, x_test, continuous_indices),
        "categorical_accuracy": test_cat_acc,
        "categorical_macro_f1": test_cat_f1,
        "auroc": (
            compute_auroc(imputed_train, y_train, imputed_test, y_test)
            if is_classification(problem_type)
            else None
        ),
    }

    elapsed = time() - start
    print(f"[Benchmark] model='{model_name}' done in {elapsed:.2f}s ⏰")
    return {"is": is_metrics, "oos": oos_metrics}


def evaluate_dataset(
    name: str,
    evaluated_model: Any,
    x_raw: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    split_seed: int,
    scenarios=None,
    miss_pct=None,
    sample_columns: bool = True,
):
    if scenarios is None:
        scenarios = ["MAR"]
    if miss_pct is None:
        miss_pct = [0.3]

    x_train, x_test, y_train, y_test = split_train_test(
        x_raw,
        y,
        problem_type=problem_type,
        random_state=split_seed,
    )
    categorical_indices, continuous_indices, unknown_indices = (
        infer_feature_type_indices_from_names(x_train.columns)
    )
    if unknown_indices:
        unknown_cols = [str(x_train.columns[idx]) for idx in unknown_indices]
        raise ValueError(
            f"Unexpected feature prefixes during benchmark scoring: {unknown_cols}"
        )

    imputation_scenarios = simulate_scenarios(
        x_train,
        x_test,
        sample_columns=sample_columns,
        random_state=split_seed,
    )
    print(
        f"[Evaluate] Start | name={name} | scenarios={scenarios} | miss_pct={miss_pct} | "
        f"train_shape={x_train.shape} | test_shape={x_test.shape}"
    )

    metric_names = [
        "rmse",
        "wasserstein",
        "categorical_accuracy",
        "categorical_macro_f1",
    ]
    if is_classification(problem_type):
        metric_names.append("auroc")
    split_results = {
        split: {metric: {} for metric in metric_names} for split in ["is", "oos"]
    }

    for scenario in scenarios:
        print(f"[Evaluate] Scenario={scenario} started")
        for split in ["is", "oos"]:
            for metric in metric_names:
                split_results[split][metric][scenario] = {}

        for missingness in miss_pct:
            for split in ["is", "oos"]:
                for metric in metric_names:
                    split_results[split][metric][scenario][missingness] = {}

            try:
                train_x, train_x_miss, train_mask = imputation_scenarios[scenario][
                    missingness
                ]["is"]
                test_x, test_x_miss, test_mask = imputation_scenarios[scenario][
                    missingness
                ]["oos"]
                evaluated_metrics = benchmark_model(
                    name,
                    copy.deepcopy(evaluated_model),
                    train_x,
                    y_train,
                    train_x_miss,
                    train_mask,
                    test_x,
                    y_test,
                    test_x_miss,
                    test_mask,
                    problem_type,
                )
                for split in ["is", "oos"]:
                    for metric in metric_names:
                        split_results[split][metric][scenario][missingness]["our"] = (
                            evaluated_metrics[split][metric]
                        )
            except KeyboardInterrupt:
                print("[Evaluate] Interrupted by user, stopping current run.")
                raise
            except Exception as exc:
                raise RuntimeError(
                    "Benchmark scenario failed | "
                    f"trial={split_seed} | scenario={scenario} | miss_pct={float(missingness):.2f}"
                ) from exc

    return split_results
