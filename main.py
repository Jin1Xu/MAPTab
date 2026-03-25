from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.loader import get_dataset
from trainer.trainer import Trainer
from config.args import (
    get_args_parser,
    load_config_args,
    merge_dataset_config,
    resolve_target_datasets,
)
from utils.runtime import format_bold


def _green(value) -> str:
    text = str(value)
    if sys.stdout.isatty():
        return f"\033[32m{text}\033[0m"
    return text


def _seed_runtime(args) -> None:
    if hasattr(args, "seed") and args.seed is not None:
        Trainer.set_seed(int(args.seed))
        print(f"[Main] Global seed set to {format_bold(args.seed)}")


def _validate_selected_datasets(args, datasets):
    print(f"[Main] Validating datasets before execution: {datasets}")
    validation_errors = {}

    for dataset in datasets:
        try:
            x, y = get_dataset(dataset, args.data_root)
            print(
                f"[Main] Dataset check passed ✅ | dataset={_green(dataset)} | X_shape={x.shape} | y_shape={y.shape}"
            )
        except Exception as exc:
            validation_errors[dataset] = str(exc)

    if validation_errors:
        details = "; ".join(
            f"{name}: {message}" for name, message in validation_errors.items()
        )
        raise ValueError(f"Dataset validation failed before execution. {details}")


def _run_single_task(args):
    _seed_runtime(args)
    from tasks.experiment import run_experiment

    return run_experiment(args)


def _run_batch_tasks(args, datasets):
    green_datasets = [_green(dataset) for dataset in datasets]
    print(f"[Main] Batch mode enabled | datasets={green_datasets}")
    output_dir = Path(args.output_root) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}
    for dataset in datasets:
        dataset_args = deepcopy(args)
        dataset_args.dataset = dataset
        dataset_args = merge_dataset_config(dataset_args)
        print(f"[Main] Running dataset={_green(dataset)}")
        output_files[dataset] = str(_run_single_task(dataset_args))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest_path = output_dir / f"batch-experiment-{timestamp}.json"
    manifest = {
        "exp_name": args.exp_name,
        "datasets": datasets,
        "output_files": output_files,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    return manifest_path


def main():
    print("[Main] Building argument parser")
    parser = get_args_parser()
    cli_args = parser.parse_args()
    if cli_args.config:
        print(f"[Main] Loading config file: {cli_args.config}")
    else:
        print("[Main] Loading default config: src/config/default.yaml")
    args = load_config_args(cli_args.config)
    target_datasets = resolve_target_datasets(args)
    _validate_selected_datasets(args, target_datasets)
    if len(target_datasets) == 1:
        args.dataset = target_datasets[0]
        args = merge_dataset_config(args)
    if len(target_datasets) > 1:
        dataset_desc = f"datasets={[_green(dataset) for dataset in target_datasets]}"
    else:
        dataset_desc = f"dataset={_green(args.dataset)}"
    print(
        f"[Main] Runtime args ready | {dataset_desc} | "
        f"batch_size={format_bold(getattr(args, 'batch_size', 'NA'))} | "
        f"n_iter={getattr(args, 'n_iter', 'NA')} | n_jobs={getattr(args, 'n_jobs', 'NA')}"
    )
    if len(target_datasets) > 1:
        output_file = _run_batch_tasks(args, target_datasets)
    else:
        output_file = _run_single_task(args)

    print(f"[Main] Done 🎉 Output saved to: {output_file}")


if __name__ == "__main__":
    main()
