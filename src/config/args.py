import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


def get_args_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("TabularImputer", add_help=add_help)
    parser.add_argument(
        "--config",
        default="",
        type=str,
        help=(
            "Path to a JSON/YAML config file. " "If unset, use src/config/default.yaml."
        ),
    )
    return parser


def get_dataset_config(dataset: str, config_data: Optional[dict] = None) -> dict:
    if config_data is None:
        config_data = _load_config_file(str(DEFAULT_CONFIG_PATH))
    datasets_config = config_data.get("datasets", {})
    if not isinstance(datasets_config, dict):
        raise ValueError("Key 'datasets' in default.yaml must be a mapping.")
    if dataset not in datasets_config:
        raise KeyError(
            f"Unsupported dataset '{dataset}'. Add config in src/config/default.yaml first."
        )
    return deepcopy(datasets_config[dataset])


def get_configured_dataset_names(config_data: Optional[dict] = None) -> list[str]:
    if config_data is None:
        config_data = _load_config_file(str(DEFAULT_CONFIG_PATH))
    datasets_config = config_data.get("datasets", {})
    if not isinstance(datasets_config, dict):
        raise ValueError("Key 'datasets' in default.yaml must be a mapping.")
    return list(datasets_config.keys())


def resolve_target_datasets(args: argparse.Namespace) -> list[str]:
    batch_datasets = getattr(args, "target_datasets", None)
    if batch_datasets is None:
        raise ValueError("Config must define 'target_datasets'.")

    raw_items = (
        [batch_datasets] if isinstance(batch_datasets, str) else list(batch_datasets)
    )
    normalized = [str(item).strip() for item in raw_items if str(item).strip()]
    if not normalized:
        raise ValueError("'target_datasets' must contain at least one dataset.")

    if len(normalized) == 1 and normalized[0].lower() == "all":
        return get_configured_dataset_names(getattr(args, "_config_data", None))

    return normalized


def merge_dataset_config(args: argparse.Namespace) -> argparse.Namespace:
    config_data = getattr(args, "_config_data", None)
    dataset_config = get_dataset_config(args.dataset, config_data=config_data)
    for key, value in dataset_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args


def _load_config_file(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            import yaml
        except Exception as exc:
            raise ImportError(
                "YAML config requires PyYAML. Install it with: pip install pyyaml"
            ) from exc

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file root must be a key-value map: {path}")
    return data


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config_args(config_path: str = "") -> argparse.Namespace:
    base_path = str(DEFAULT_CONFIG_PATH)
    base_config = _load_config_file(base_path)

    resolved_path = base_path
    config_data = base_config
    if config_path:
        resolved_path = str(Path(config_path))
        if Path(resolved_path).resolve() != Path(base_path).resolve():
            override_config = _load_config_file(resolved_path)
            config_data = _deep_merge_dicts(base_config, override_config)

    public_config = {
        key: deepcopy(value) for key, value in config_data.items() if key != "datasets"
    }
    args = argparse.Namespace(**public_config)
    target_datasets = getattr(args, "target_datasets", None)
    if isinstance(target_datasets, str):
        target_items = [target_datasets]
    elif isinstance(target_datasets, (list, tuple)):
        target_items = list(target_datasets)
    else:
        target_items = []
    if len(target_items) == 1:
        candidate = str(target_items[0]).strip()
        if candidate and candidate.lower() != "all":
            args.dataset = candidate
    args.config = resolved_path
    args._config_data = deepcopy(config_data)
    return args
