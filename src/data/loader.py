from pathlib import Path

import pandas as pd

from data.preprocessing import infer_feature_type_indices_from_names


def _resolve_data_root(data_root: str) -> Path:
    configured = Path(data_root)
    if configured.exists():
        return configured

    raise FileNotFoundError(f"Configured data root does not exist: '{configured}'")


def _encode_categorical_series(series: pd.Series) -> pd.Series:
    categorical = pd.Categorical(series)
    codes = pd.Series(categorical.codes, index=series.index, dtype="float32")
    return codes.where(series.notna(), other=float("nan"))


def _prepare_csv_dataset(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Loaded CSV is empty.")

    columns = list(df.columns)
    if columns[0] != "idx":
        raise ValueError("Dataset CSV must use 'idx' as the first column.")
    if columns[-1] != "label":
        raise ValueError("Dataset CSV must use 'label' as the last column.")

    # The first column idx is excluded from training, and the last column label is the target.
    feature_frame = df.iloc[:, 1:-1].copy()
    label_col = "label"
    y = df[label_col].copy()

    cat_indices, num_indices, unknown_indices = infer_feature_type_indices_from_names(
        feature_frame.columns
    )
    if unknown_indices:
        unknown_cols = [str(feature_frame.columns[idx]) for idx in unknown_indices]
        raise ValueError(
            "All feature columns must start with 'num' or 'cat'. "
            f"Columns without valid prefix: {unknown_cols}"
        )

    for col in feature_frame.columns:
        lowered = str(col).strip().lower()
        if lowered.startswith("cat"):
            feature_frame[col] = _encode_categorical_series(feature_frame[col])
        else:
            feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.Categorical(y).codes, index=y.index, name=label_col)
    else:
        y = pd.to_numeric(y, errors="coerce")

    if feature_frame.isna().any().any():
        missing_cols = feature_frame.columns[feature_frame.isna().any()].tolist()
        raise ValueError(
            "Input CSV must be complete before missing-value simulation. "
            f"Found missing/invalid values in feature columns: {missing_cols}"
        )
    if y.isna().any():
        raise ValueError(
            "Input CSV label column must be complete and contain no missing/invalid values."
        )

    feature_frame.attrs["categorical_feature_indices"] = cat_indices
    feature_frame.attrs["continuous_feature_indices"] = num_indices
    return feature_frame, y


def get_dataset(dataset: str, data_root: str):
    dataset = dataset.lower()
    print(f"[Data] Loading dataset='{dataset}' from data_root='{data_root}'")

    root = _resolve_data_root(data_root)
    csv_path = root / f"{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    x, y = _prepare_csv_dataset(df)
    print(
        f"[Data] Loaded CSV: path='{csv_path}', X_shape={x.shape}, y_shape={y.shape}, "
        f"continuous={len(x.attrs.get('continuous_feature_indices', []))}, "
        f"categorical={len(x.attrs.get('categorical_feature_indices', []))}"
    )
    return x, y
