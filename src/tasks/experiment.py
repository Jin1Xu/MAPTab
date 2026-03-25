import json
from datetime import datetime
from pathlib import Path

from data.loader import get_dataset
from benchmark.core import compare_models
from benchmark.scenarios import fit_scale_data
from tasks.common import build_run_config, to_jsonable
from trainer.trainer import Trainer
from utils.runtime import compute_logreg_auroc, is_classification


def _build_output_path(args) -> Path:
    output_dir = Path(args.output_root) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    note_suffix = f"-{args.note}" if args.note else ""
    return output_dir / f"{args.dataset}{note_suffix}-{timestamp}.json"


def _run_single_experiment(args, x, y, org_auroc_score=None):
    trainer = Trainer(args)
    print("[Experiment] Trainer initialized")

    output_file = _build_output_path(args)
    results = {}
    print("[Experiment] Running benchmark compare_models")
    results[args.dataset] = compare_models(
        name=args.exp_name,
        evaluated_model=trainer,
        x_raw=x,
        y=y,
        scenarios=args.scenarios,
        miss_pct=args.miss_pct,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
        gpu_ids=args.gpu_ids,
        problem_type=args.problem_type,
    )
    if org_auroc_score is not None:
        results[args.dataset]["org_auroc_score"] = org_auroc_score
    results[args.dataset]["run_config"] = build_run_config(args)

    # with output_file.open("w", encoding="utf-8") as f:
    #     json.dump(to_jsonable(results), f, indent=4, allow_nan=False)
    return output_file


def run_experiment(args):
    if args.problem_type is None:
        raise ValueError(
            "problem_type is required. Please set it in src/config/default.yaml under datasets[dataset]."
        )

    print(
        f"[Experiment] Start | dataset={args.dataset} | scenarios={args.scenarios} | "
        f"miss_pct={args.miss_pct} | n_iter={args.n_iter} | n_jobs={args.n_jobs} | gpu_ids={args.gpu_ids}"
    )
    x, y = get_dataset(args.dataset, args.data_root)

    org_auroc_score = None
    if is_classification(args.problem_type):
        x_scaled, _ = fit_scale_data(x)
        org_auroc_score = compute_logreg_auroc(x_scaled, y)
        print(f"[Experiment] Original AUROC: {org_auroc_score:.4f}")

    return _run_single_experiment(args, x, y, org_auroc_score=org_auroc_score)
