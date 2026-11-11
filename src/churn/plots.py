from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y, score, title: str, path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y, score, ax=ax, name="model")
    ax.set_title(title, fontsize=9)
    _save(fig, path)


def plot_reliability(y, proba, title: str, path) -> None:
    sns.set_theme(style="whitegrid")
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.legend()
    ax.set_title(title, fontsize=9)
    _save(fig, path)


def plot_lift_by_decile(decile_df, title: str, path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=decile_df, x="decile", y="lift", color="steelblue", ax=ax)
    ax.axhline(1.0, color="grey", linestyle="--")
    ax.set_xlabel("Score decile (1 = highest risk)")
    ax.set_ylabel("Lift vs base rate")
    ax.set_title(title, fontsize=9)
    _save(fig, path)


def plot_monthly_churn(monthly_df, title: str, path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=monthly_df, x="snapshot_date", y="churn_rate", marker="o", ax=ax)
    ax.set_xlabel("Snapshot date")
    ax.set_ylabel("Churn rate")
    ax.set_title(title, fontsize=9)
    fig.autofmt_xdate()
    _save(fig, path)


def plot_feature_importance(model, feature_names, title: str, path) -> None:
    sns.set_theme(style="whitegrid")
    coef = model[-1].coef_[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=coef, y=list(feature_names), color="steelblue", ax=ax)
    ax.set_xlabel("Standardised log-coefficient")
    ax.set_title(title, fontsize=9)
    _save(fig, path)
