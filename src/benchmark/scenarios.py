import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Optional

from data.preprocessing import infer_feature_type_indices_from_names
from utils.runtime import is_classification

from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.distributions import enable_reproducible_results


def split_train_test(
    x_raw: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    random_state: int,
):
    stratify = None
    if is_classification(problem_type):
        label_counts = y.value_counts()
        if len(label_counts) > 1 and int(label_counts.min()) >= 2:
            stratify = y

    x_train, x_test, y_train, y_test = train_test_split(
        x_raw,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )
    return (
        x_train.reset_index(drop=True),
        x_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def fit_scale_data(x: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    categorical_indices, continuous_indices, unknown = (
        infer_feature_type_indices_from_names(x.columns)
    )
    if unknown:
        unknown_cols = [str(x.columns[idx]) for idx in unknown]
        raise ValueError(
            "All benchmark feature columns must start with 'num' or 'cat'. "
            f"Columns without valid prefix: {unknown_cols}"
        )

    preproc = MinMaxScaler()
    scaled = x.astype("float32").copy()
    if continuous_indices:
        scaled_values = preproc.fit_transform(
            scaled.iloc[:, continuous_indices].astype("float32")
        )
        scaled.iloc[:, continuous_indices] = scaled_values.astype("float32")
    return scaled, preproc


def transform_scaled_data(x: pd.DataFrame, preproc: MinMaxScaler) -> pd.DataFrame:
    """Scale test-set continuous columns with training-set statistics to avoid data leakage."""
    _, continuous_indices, unknown = infer_feature_type_indices_from_names(x.columns)
    if unknown:
        unknown_cols = [str(x.columns[idx]) for idx in unknown]
        raise ValueError(
            "All benchmark feature columns must start with 'num' or 'cat'. "
            f"Columns without valid prefix: {unknown_cols}"
        )

    scaled = x.astype("float32").copy()
    if continuous_indices:
        scaled_values = preproc.transform(scaled.iloc[:, continuous_indices])
        scaled.iloc[:, continuous_indices] = scaled_values.astype("float32")
    return scaled


def _sample_target_columns(
    columns,
    column_limit: int,
    sample_columns: bool,
) -> np.ndarray:
    columns = np.asarray(columns)
    column_limit = min(len(columns), column_limit)
    if sample_columns:
        return columns[np.random.choice(len(columns), size=column_limit, replace=False)]
    return columns[list(range(column_limit))]


def _ampute_with_selected_columns(
    x: pd.DataFrame,
    mechanism: str,
    p_miss: float,
    sampled_columns,
    sample_columns: bool,
):
    columns = x.columns
    x_simulated = simulate_nan(
        x[sampled_columns].to_numpy(dtype=np.float32, copy=True),
        p_miss,
        mechanism,
        sample_columns=sample_columns,
    )

    isolated_mask = pd.DataFrame(
        np.asarray(x_simulated["mask"], dtype=np.float32),
        columns=sampled_columns,
        index=x.index,
    )
    isolated_x_miss = pd.DataFrame(
        np.asarray(x_simulated["X_incomp"], dtype=np.float32),
        columns=sampled_columns,
        index=x.index,
    )

    mask = pd.DataFrame(
        np.zeros(x.shape, dtype=np.float32), columns=columns, index=x.index
    )
    mask[sampled_columns] = isolated_mask

    x_miss = x.copy()
    x_miss[sampled_columns] = isolated_x_miss

    return x.copy(), x_miss, mask


def simulate_scenarios(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    column_limit: Optional[int] = None,
    sample_columns: bool = True,
    random_state: int = 0,
):
    enable_reproducible_results(int(random_state))
    np.random.seed(int(random_state))
    if column_limit is None:
        column_limit = x_train.shape[1]

    x_train_scaled, preproc = fit_scale_data(x_train)
    x_test_scaled = transform_scaled_data(x_test, preproc)

    datasets = {}
    mechanisms = ["MAR", "MNAR", "MCAR"]
    percentages = [0.1, 0.3, 0.5, 0.7, 0.9]

    for mechanism in mechanisms:
        datasets[mechanism] = {}
        for p_miss in percentages:
            sampled_columns = _sample_target_columns(
                x_train_scaled.columns,
                column_limit=column_limit,
                sample_columns=sample_columns,
            )
            datasets[mechanism][p_miss] = {
                "is": _ampute_with_selected_columns(
                    x_train_scaled,
                    mechanism,
                    p_miss,
                    sampled_columns,
                    sample_columns=sample_columns,
                ),
                "oos": _ampute_with_selected_columns(
                    x_test_scaled,
                    mechanism,
                    p_miss,
                    sampled_columns,
                    sample_columns=sample_columns,
                ),
            }
    return datasets
