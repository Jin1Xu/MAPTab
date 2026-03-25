import copy
import sys
from time import time
from typing import Any, Optional

import torch
from joblib import Parallel, delayed

from .evaluate import evaluate_dataset
from utils.runtime import is_classification
from hyperimpute.utils.distributions import enable_reproducible_results
from hyperimpute.utils.metrics import generate_score, print_score


def _validate_and_normalize_gpu_ids(gpu_ids):
    if gpu_ids is None:
        return None
    if len(gpu_ids) == 0:
        return None
    ids = [int(x) for x in gpu_ids]

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_ids is set but CUDA is not available.")

    device_count = torch.cuda.device_count()
    bad_ids = [x for x in ids if x < 0 or x >= device_count]
    if bad_ids:
        raise ValueError(
            f"Invalid gpu_ids={bad_ids}. Available GPU ids are 0..{device_count - 1}."
        )
    return ids


def _assign_model_device(evaluated_model: Any, gpu_id: int) -> None:
    device = torch.device(f"cuda:{gpu_id}")
    if hasattr(evaluated_model, "device"):
        evaluated_model.device = device


def _set_model_progress_meta(evaluated_model: Any, trial_idx: int, n_iter: int) -> None:
    del n_iter
    show_progress = trial_idx == 0
    desc = "Training🔥"
    position = 0

    if hasattr(evaluated_model, "progress_desc"):
        evaluated_model.progress_desc = desc
    if hasattr(evaluated_model, "progress_position"):
        evaluated_model.progress_position = position
    if hasattr(evaluated_model, "progress_disable"):
        evaluated_model.progress_disable = not show_progress
    if hasattr(evaluated_model, "log_disable"):
        evaluated_model.log_disable = not show_progress


def _print_metric_block(
    title: str, headers, rows, highlight_values: bool = False
) -> None:
    print(title)
    if not rows:
        print("  (no rows)")
        return
    use_ansi_bold = sys.stdout.isatty()
    for idx, row in enumerate(rows, start=1):
        fields = []
        for i in range(len(headers)):
            value = row[i]
            if i >= 2 and highlight_values:
                value = f"\033[1m{value}\033[0m" if use_ansi_bold else f"**{value}**"
            fields.append(f"{headers[i]}={value}")
        print(f"  {idx}. " + " | ".join(fields))


def compare_models(
    name: str,
    evaluated_model: Any,
    x_raw,
    y,
    problem_type: str,
    evaluated_header: Optional[str] = None,
    scenarios=None,
    miss_pct=None,
    n_iter: int = 2,
    sample_columns: bool = True,
    display_results: bool = True,
    n_jobs: int = 1,
    gpu_ids=None,
):
    if scenarios is None:
        scenarios = ["MAR"]
    if miss_pct is None:
        miss_pct = [0.3]
    gpu_ids = _validate_and_normalize_gpu_ids(gpu_ids)
    print(
        f"[Compare] Start | name={name} | n_iter={n_iter} | n_jobs={n_jobs} | "
        f"scenarios={scenarios} | miss_pct={miss_pct} | gpu_ids={gpu_ids}"
    )

    enable_reproducible_results()
    start = time()
    use_auroc = is_classification(problem_type)
    metric_names = [
        "rmse",
        "wasserstein",
        "categorical_accuracy",
        "categorical_macro_f1",
    ]
    if use_auroc:
        metric_names.append("auroc")

    def init_metric_store():
        return {
            split: {metric: {} for metric in metric_names} for split in ["is", "oos"]
        }

    def add_metric_value(
        store: dict,
        split: str,
        metric: str,
        scenario: str,
        missingness: float,
        method: str,
        score: float,
    ):
        store[split][metric].setdefault(scenario, {})
        store[split][metric][scenario].setdefault(missingness, {})
        store[split][metric][scenario][missingness].setdefault(method, [])
        store[split][metric][scenario][missingness][method].append(score)

    aggregated = init_metric_store()

    def eval_local(it: int):
        enable_reproducible_results(it)
        local_model = copy.deepcopy(evaluated_model)
        _set_model_progress_meta(local_model, it, n_iter)
        if gpu_ids is not None:
            gpu_id = gpu_ids[it % len(gpu_ids)]
            _assign_model_device(local_model, gpu_id)
            print(f"[Compare] trial={it} assigned to cuda:{gpu_id}")
        else:
            print(f"[Compare] trial={it} using default device")
        return evaluate_dataset(
            name=name,
            evaluated_model=local_model,
            x_raw=x_raw,
            y=y,
            problem_type=problem_type,
            split_seed=it,
            scenarios=scenarios,
            miss_pct=miss_pct,
            sample_columns=sample_columns,
        )

    try:
        repeated = Parallel(n_jobs=n_jobs)(
            delayed(eval_local)(it) for it in range(n_iter)
        )
    except KeyboardInterrupt:
        print("[Compare] Interrupted by user, stopping parallel workers.")
        raise

    for local_results in repeated:
        for split in ["is", "oos"]:
            for metric in metric_names:
                for scenario in local_results[split][metric]:
                    for missingness in local_results[split][metric][scenario]:
                        for method, score in local_results[split][metric][scenario][
                            missingness
                        ].items():
                            add_metric_value(
                                aggregated,
                                split,
                                metric,
                                scenario,
                                missingness,
                                method,
                                score,
                            )

    evaluated_header = evaluated_header or f"Evaluated: {evaluated_model.name()}"
    headers = ["Scenario", "miss_pct [0, 1]", evaluated_header]
    metric_titles = {
        "rmse": "Continuous RMSE (lower is better)",
        "wasserstein": "Continuous Wasserstein (lower is better)",
        "categorical_accuracy": "Categorical Accuracy (higher is better)",
        "categorical_macro_f1": "Categorical Macro-F1 (higher is better)",
        "auroc": "AUROC (higher is better)",
    }
    final_results = {"headers": headers, "is": {}, "oos": {}}
    summary_rows = {"is": {}, "oos": {}}

    for split in ["is", "oos"]:
        for metric in metric_names:
            final_results[split][metric] = []
            summary_rows[split][metric] = []
            for scenario in aggregated[split][metric]:
                for missingness in aggregated[split][metric][scenario]:
                    miss_pct_label = f"{float(missingness):.2f}"
                    local_str = [scenario, miss_pct_label]
                    local_num = [scenario, missingness]
                    for method in ["our"]:
                        mean_score, std_score = generate_score(
                            aggregated[split][metric][scenario][missingness][method]
                        )
                        local_str.append(print_score((mean_score, std_score)))
                        local_num.append((mean_score, std_score))
                    summary_rows[split][metric].append(local_str)
                    final_results[split][metric].append(local_num)

    if display_results:
        elapsed = time() - start
        print(f"[Compare] Aggregation done in {elapsed:.2f}s")
        print(
            f"=== ✅ Benchmark Summary | name={name} | trials={n_iter} | scenarios={len(scenarios)} | elapsed={elapsed:.2f}s ==="
        )
        for split, split_title in [
            ("is", "IS (train split)"),
            ("oos", "OOS (test split)"),
        ]:
            print(f"[Compare] {split_title}")
            for metric in metric_names:
                _print_metric_block(
                    metric_titles[metric],
                    headers,
                    summary_rows[split][metric],
                    highlight_values=metric
                    in {"rmse", "categorical_accuracy", "auroc"},
                )
        if not use_auroc:
            print("[Compare] Regression task detected, skip AUROC summary")

    return final_results
