import argparse
from pathlib import Path

from churn.config import (
    H_DAYS,
    H_SENSITIVITY_DAYS,
    MODEL_VERSION,
    TEST_SNAPSHOTS,
    TRAIN_SNAPSHOTS,
)
from churn.data import clean_transactions, load_transactions
from churn.features import FEATURE_COLUMNS
from churn.metrics import decile_table
from churn.models import fit_logistic
from churn.pipeline import build_panel, h_sensitivity, monthly_churn_rate, run_experiment
from churn.plots import (
    plot_feature_importance,
    plot_lift_by_decile,
    plot_monthly_churn,
    plot_pr_curve,
    plot_reliability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CH01 churn baseline.")
    parser.add_argument("--input", required=True, help="Path to the Online Retail II CSV or XLSX file.")
    parser.add_argument("--out-dir", default="reports", help="Directory for metrics and figures.")
    parser.add_argument("--h", type=int, default=H_DAYS, help="Churn window in days.")
    args = parser.parse_args()

    out = Path(args.out_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    tx = clean_transactions(load_transactions(args.input))
    results, predictions = run_experiment(tx, TRAIN_SNAPSHOTS, TEST_SNAPSHOTS, args.h)
    results.to_csv(out / "ch01_metrics.csv", index=False)

    sensitivity = h_sensitivity(tx, TRAIN_SNAPSHOTS, TEST_SNAPSHOTS, H_SENSITIVITY_DAYS)
    sensitivity.to_csv(out / "ch01_h_sensitivity.csv", index=False)

    monthly = monthly_churn_rate(tx, args.h, start=TRAIN_SNAPSHOTS[0])
    monthly.to_csv(out / "ch01_monthly_churn.csv", index=False)

    latest = predictions[predictions["snapshot_date"] == predictions["snapshot_date"].max()]
    latest_date = latest["snapshot_date"].iloc[0].date()
    tag = f"Online Retail II | snapshot {latest_date} | H={args.h}d | {MODEL_VERSION}"

    plot_pr_curve(latest["churned"], latest["proba"], f"PR curve | {tag}", figures / "pr_curve.png")
    plot_reliability(latest["churned"], latest["proba"], f"Reliability | {tag}", figures / "reliability.png")
    decile = decile_table(latest["churned"], latest["proba"])
    plot_lift_by_decile(decile, f"Lift by decile | {tag}", figures / "lift_by_decile.png")
    plot_monthly_churn(monthly, f"Monthly churn rate | Online Retail II | H={args.h}d | {MODEL_VERSION}",
                       figures / "monthly_churn_rate.png")
    train_panel = build_panel(tx, TRAIN_SNAPSHOTS, args.h)
    model = fit_logistic(train_panel[~train_panel["single_order"]])
    plot_feature_importance(model, FEATURE_COLUMNS, f"Logistic coefficients | {tag}",
                            figures / "feature_importance.png")

    cols = ["snapshot_date", "model", "n", "base_rate", "pr_auc", "roc_auc", "lift_top_decile", "brier"]
    print("Per-snapshot results (single-order customers excluded):")
    print(results[cols].to_string(index=False))
    print(f"\nLatest test snapshot {latest_date}, H={args.h}d:")
    print(results[results["snapshot_date"] == results["snapshot_date"].max()][cols].to_string(index=False))


if __name__ == "__main__":
    main()
