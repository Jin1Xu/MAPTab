import math

import numpy as np


def to_jsonable(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        return item
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "__fspath__"):
        return str(value)
    return str(value)


def build_run_config(args) -> dict:
    return {
        key: to_jsonable(value)
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
