import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def top_fraction_lift(y, score, fraction: float = 0.1) -> float:
    """Churn rate among the top `fraction` of scores divided by the overall churn rate."""
    y = np.asarray(y)
    score = np.asarray(score)
    base_rate = y.mean()
    n_top = max(1, int(np.ceil(len(y) * fraction)))
    order = np.argsort(-score, kind="mergesort")
    return float(y[order[:n_top]].mean() / base_rate)


def evaluate(y, score, proba=None) -> dict:
    y = np.asarray(y)
    if y.min() == y.max():
        raise ValueError("Evaluation needs both churned and retained customers.")
    return {
        "n": len(y),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "lift_top_decile": top_fraction_lift(y, score),
        "brier": float(brier_score_loss(y, proba)) if proba is not None else np.nan,
    }


def decile_table(y, score) -> pd.DataFrame:
    """Churn rate and lift per score decile; decile 1 holds the highest scores."""
    df = pd.DataFrame({"churned": np.asarray(y), "score": np.asarray(score)})
    df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    n = len(df)
    df["decile"] = (np.arange(n) * 10 // n) + 1
    table = df.groupby("decile").agg(n=("churned", "size"), churn_rate=("churned", "mean"))
    table["lift"] = table["churn_rate"] / df["churned"].mean()
    return table.reset_index()
