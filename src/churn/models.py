import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from churn.config import RANDOM_STATE
from churn.features import FEATURE_COLUMNS


def fit_recency_rule(train: pd.DataFrame) -> int:
    """Return the cut-off d (churn if recency_days > d) that maximises F1 on the training data."""
    y = train["churned"].to_numpy()
    recency = train["recency_days"].to_numpy()
    best_d, best_f1 = int(recency.min()), -1.0
    for d in np.unique(recency):
        f1 = f1_score(y, (recency > d).astype(int), zero_division=0)
        if f1 > best_f1:
            best_d, best_f1 = int(d), f1
    return best_d


def predict_recency_rule(df: pd.DataFrame, d: int) -> np.ndarray:
    return (df["recency_days"].to_numpy() > d).astype(int)


def fit_logistic(train: pd.DataFrame):
    model = make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(train[FEATURE_COLUMNS], train["churned"])
    return model


def predict_logistic(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
