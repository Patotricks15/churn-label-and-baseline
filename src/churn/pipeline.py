import numpy as np
import pandas as pd

from churn.labels import build_snapshot
from churn.metrics import evaluate
from churn.models import (
    fit_logistic,
    fit_recency_rule,
    predict_logistic,
    predict_recency_rule,
)


def build_panel(tx: pd.DataFrame, snapshots, h_days: int) -> pd.DataFrame:
    return pd.concat(
        [build_snapshot(tx, snap, h_days) for snap in snapshots], ignore_index=True
    )


def run_experiment(tx, train_snapshots, test_snapshots, h_days: int):
    """Train on earlier snapshots, test on later ones. Single-order customers are excluded from the main metrics."""
    train_panel = build_panel(tx, train_snapshots, h_days)
    test_panel = build_panel(tx, test_snapshots, h_days)
    if train_panel["snapshot_date"].max() >= test_panel["snapshot_date"].min():
        raise ValueError("Train snapshots must all precede test snapshots.")

    main_train = train_panel[~train_panel["single_order"]]
    d = fit_recency_rule(main_train)
    logit = fit_logistic(main_train)

    rows, preds = [], []
    for snap, part in test_panel.groupby("snapshot_date"):
        single = part[part["single_order"]]
        main = part[~part["single_order"]].copy()
        y = main["churned"].to_numpy()
        proba = predict_logistic(logit, main)

        common = {
            "snapshot_date": snap,
            "h_days": h_days,
            "n_single_order": len(single),
            "single_order_base_rate": float(single["churned"].mean()) if len(single) else np.nan,
        }
        rows.append({**common, "model": "recency_rule", "threshold_d": d,
                     **evaluate(y, main["recency_days"].to_numpy())})
        rows.append({**common, "model": "logistic_regression", "threshold_d": np.nan,
                     **evaluate(y, proba, proba)})

        main["proba"] = proba
        main["rule_pred"] = predict_recency_rule(main, d)
        preds.append(main)

    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def monthly_churn_rate(tx: pd.DataFrame, h_days: int, start: str) -> pd.DataFrame:
    data_end = tx["invoice_date"].max().normalize()
    starts = pd.date_range(start, data_end - pd.Timedelta(days=h_days), freq="MS")
    rows = []
    for snap in starts:
        main = build_snapshot(tx, snap, h_days)
        main = main[~main["single_order"]]
        rows.append({"snapshot_date": snap, "n": len(main), "churn_rate": main["churned"].mean()})
    return pd.DataFrame(rows)


def h_sensitivity(tx, train_snapshots, test_snapshots, h_values) -> pd.DataFrame:
    frames = [run_experiment(tx, train_snapshots, test_snapshots, h)[0] for h in h_values]
    return pd.concat(frames, ignore_index=True)
