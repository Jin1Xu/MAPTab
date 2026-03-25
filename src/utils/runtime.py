from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def format_bold(value) -> str:
    text = str(value)
    if sys.stdout.isatty():
        return f"\033[1m{text}\033[0m"
    return f"**{text}**"


def is_classification(problem_type: str) -> bool:
    return str(problem_type).strip().lower() == "classification"


def compute_logreg_auroc(
    train_x,
    y_train,
    eval_x=None,
    y_eval=None,
) -> float:
    train_x_np = np.asarray(train_x)
    y_train_np = np.asarray(y_train)
    eval_x_np = train_x_np if eval_x is None else np.asarray(eval_x)
    y_eval_np = y_train_np if y_eval is None else np.asarray(y_eval)

    train_classes = np.unique(y_train_np)
    eval_classes = np.unique(y_eval_np)
    if train_classes.size < 2 or eval_classes.size < 2:
        return float("nan")

    clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0).fit(
        train_x_np, y_train_np
    )

    try:
        if train_classes.size > 2:
            return float(
                roc_auc_score(
                    y_eval_np,
                    clf.predict_proba(eval_x_np),
                    multi_class="ovr",
                    labels=clf.classes_,
                )
            )
        return float(roc_auc_score(y_eval_np, clf.predict_proba(eval_x_np)[:, 1]))
    except ValueError:
        return float("nan")
