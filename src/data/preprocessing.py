from __future__ import annotations

from typing import Callable

import numpy as np

EPS = 1e-8


def infer_feature_type_indices_from_names(feature_names):
    categorical_indices = []
    continuous_indices = []
    unknown_indices = []
    names = [] if feature_names is None else feature_names

    for idx, name in enumerate(names):
        lowered = str(name).strip().lower()
        if lowered.startswith("cat"):
            categorical_indices.append(idx)
        elif lowered.startswith("num"):
            continuous_indices.append(idx)
        else:
            unknown_indices.append(idx)

    return categorical_indices, continuous_indices, unknown_indices


class TabularDataProcessor:
    def __init__(self):
        self.feature_names = None
        self.norm_parameters = None

    def set_feature_names(self, feature_names) -> None:
        self.feature_names = list(feature_names) if feature_names is not None else None

    @staticmethod
    def _build_missing_mask(x_raw: np.ndarray) -> np.ndarray:
        return 1.0 - np.isnan(x_raw).astype(np.float32)

    def _resolve_categorical_feature_mask(self, x_raw: np.ndarray):
        dim = x_raw.shape[1]
        if self.feature_names is not None:
            name_cat, name_con, unknown = infer_feature_type_indices_from_names(
                self.feature_names
            )
            if unknown:
                unknown_cols = [str(self.feature_names[idx]) for idx in unknown]
                raise ValueError(
                    "All feature columns must start with 'num' or 'cat'. "
                    f"Columns without valid prefix: {unknown_cols}"
                )
            if len(name_cat) + len(name_con) != dim:
                raise ValueError(
                    "Feature name count does not match input dimension when resolving column prefixes."
                )
            cat_mask = np.zeros(dim, dtype=bool)
            cat_mask[name_cat] = True
            return cat_mask, "name_prefix"

    def fit_transform(self, x_raw: np.ndarray):
        dim = x_raw.shape[1]
        m_np = self._build_missing_mask(x_raw)
        cat_mask, type_source = self._resolve_categorical_feature_mask(x_raw)
        x_proc = np.zeros_like(x_raw, dtype=np.float32)

        min_val = np.zeros(dim, dtype=np.float32)
        max_val = np.ones(dim, dtype=np.float32)
        cat_id_to_value = [None] * dim
        cat_value_to_id = [None] * dim
        cat_cardinalities = [0] * dim

        for j in range(dim):
            observed_mask = m_np[:, j] > EPS
            observed_values = x_raw[observed_mask, j]
            if bool(cat_mask[j]):
                if observed_values.size == 0:
                    uniq = np.array([0.0], dtype=np.float32)
                else:
                    uniq = np.sort(np.unique(observed_values.astype(np.float32)))
                mapper = {float(v): i for i, v in enumerate(uniq.tolist())}
                encoded = np.zeros(x_raw.shape[0], dtype=np.float32)
                for idx in np.where(observed_mask)[0]:
                    encoded[idx] = float(mapper.get(float(x_raw[idx, j]), 0))
                x_proc[:, j] = encoded
                cat_id_to_value[j] = uniq.astype(np.float32)
                cat_value_to_id[j] = mapper
                cat_cardinalities[j] = int(len(uniq))
            else:
                if observed_values.size == 0:
                    col_min, col_max = 0.0, 1.0
                else:
                    col_min = float(np.min(observed_values))
                    col_max = float(np.max(observed_values))
                    if abs(col_max - col_min) < EPS:
                        col_max = col_min + 1.0
                min_val[j] = col_min
                max_val[j] = col_max
                norm_col = (x_raw[:, j] - col_min) / (col_max - col_min + EPS)
                norm_col[~observed_mask] = 0.0
                x_proc[:, j] = norm_col.astype(np.float32)

        x_proc = np.nan_to_num(x_proc, nan=0.0)
        self.norm_parameters = {
            "feature_type_source": type_source,
            "min": min_val,
            "max": max_val,
            "categorical_feature_mask": cat_mask.astype(bool),
            "cat_cardinalities": [int(v) for v in cat_cardinalities],
            "cat_id_to_value": cat_id_to_value,
            "cat_value_to_id": cat_value_to_id,
        }
        return x_proc.astype(np.float32), m_np.astype(np.float32)

    def transform(self, x_raw: np.ndarray):
        if self.norm_parameters is None:
            raise RuntimeError("TabularDataProcessor is not fitted.")

        cat_mask = self.norm_parameters["categorical_feature_mask"]
        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]
        cat_value_to_id = self.norm_parameters["cat_value_to_id"]
        m_np = self._build_missing_mask(x_raw)
        x_proc = np.zeros_like(x_raw, dtype=np.float32)

        for j in range(x_raw.shape[1]):
            observed_mask = m_np[:, j] > EPS
            if bool(cat_mask[j]):
                mapper = cat_value_to_id[j] or {}
                encoded = np.zeros(x_raw.shape[0], dtype=np.float32)
                for idx in np.where(observed_mask)[0]:
                    encoded[idx] = float(mapper.get(float(x_raw[idx, j]), 0))
                x_proc[:, j] = encoded
            else:
                norm_col = (x_raw[:, j] - min_val[j]) / (max_val[j] - min_val[j] + EPS)
                norm_col[~observed_mask] = 0.0
                x_proc[:, j] = norm_col.astype(np.float32)

        x_proc = np.nan_to_num(x_proc, nan=0.0)
        return x_proc.astype(np.float32), m_np.astype(np.float32)

    def decode_predictions(self, pred: np.ndarray) -> np.ndarray:
        if self.norm_parameters is None:
            raise RuntimeError("TabularDataProcessor is not fitted.")

        out = np.zeros_like(pred, dtype=np.float32)
        cat_mask = self.norm_parameters["categorical_feature_mask"]
        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]
        cat_id_to_value = self.norm_parameters["cat_id_to_value"]

        for j in range(pred.shape[1]):
            if bool(cat_mask[j]):
                values = cat_id_to_value[j]
                if values is None or len(values) == 0:
                    out[:, j] = pred[:, j]
                else:
                    ids = np.rint(pred[:, j]).astype(np.int64)
                    ids = np.clip(ids, 0, len(values) - 1)
                    out[:, j] = values[ids]
            else:
                out[:, j] = pred[:, j] * (max_val[j] - min_val[j] + EPS) + min_val[j]

        return out

    @staticmethod
    def _format_index_list(indices, max_items: int = 80) -> str:
        indices = [int(i) for i in indices]
        if len(indices) <= max_items:
            return str(indices)
        head = indices[: max_items // 2]
        tail = indices[-(max_items // 2) :]
        return f"{head} ... {tail} (total={len(indices)})"

    def log_feature_type_summary(self, log_fn: Callable[[str], None]) -> None:
        if self.norm_parameters is None:
            return

        cat_mask = self.norm_parameters["categorical_feature_mask"]
        source = self.norm_parameters.get("feature_type_source", "name_prefix")
        cat_indices = np.where(cat_mask)[0].tolist()
        con_indices = np.where(~cat_mask)[0].tolist()
        log_fn(
            f"[Trainer] feature type source={source} | categorical={len(cat_indices)} | continuous={len(con_indices)}"
        )
        log_fn(
            f"[Trainer] categorical feature indices: {self._format_index_list(cat_indices)}"
        )
        log_fn(
            f"[Trainer] continuous feature indices: {self._format_index_list(con_indices)}"
        )
