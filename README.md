# churn-label-and-baseline (CH01)

**Question:** How many customers churn in a fixed window, and how well does recency predict it?

This repository defines a transaction-only churn label, builds leakage-free snapshot panels, and compares a recency rule with a logistic regression on RFM features.

## Data

- **Dataset:** Online Retail II UCI, Kaggle slug [`mashlyn/online-retail-ii-uci`](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).
- **License (per Kaggle metadata):** CC0.
- **Read path:** `/kaggle/input/online-retail-ii-uci/` (see [data/README.md](data/README.md)).
- **Retrieval:** the dataset has not been downloaded. Metadata was checked on 2026-11-08 (see the plan's Appendix A).

## Churn definition

- **Churn window:** `H = 90` days. A customer churns if they make no purchase in `[snapshot, snapshot + 90 days)`.
- **Features:** only transactions strictly before the snapshot (RFM and tenure).
- **Cleaning before labelling:** cancellations (invoice starts with `C`), returns (quantity ≤ 0), zero-price lines, and rows without a customer ID are removed.
- **Population:** customers with at least one purchase before the snapshot.
- **Single-order customers:** kept in the panel and reported separately. The main metrics exclude them because their label is weakly identified.

## Snapshots and validation

- **Train snapshots:** 2010-03-01, 2010-06-01, 2010-09-01, 2010-12-01.
- **Test snapshots:** 2011-03-01, 2011-06-01, 2011-09-01. The headline result is the latest test snapshot, 2011-09-01.
- **Split:** time-ordered only. Train snapshots precede test snapshots, and the code rejects any violation.
- The last label window (2011-09-01 + 90 days = 2011-11-30) fits inside the dataset, which ends on 2011-12-09. The code raises an error if a label window would extend past the last observed date.

## Models

| Model | Definition |
|---|---|
| Recency rule (baseline) | Churn if `recency_days > d`. `d` maximises F1 on the training snapshots. Score for PR-AUC and lift is `recency_days`. |
| Logistic regression | `log1p` → standard scaling → `LogisticRegression(random_state=42)` on `recency_days`, `frequency`, `monetary`, `tenure_days`, `avg_order_value`. |

## Metrics

- **PR-AUC** (primary, churn is imbalanced) and **ROC-AUC**.
- **Brier score** for the logistic regression. The recency rule is not a probability, so its Brier score is left empty.
- **Lift in the top decile:** churn rate among the top 10% of scores divided by the base rate.

## Outputs

Running the script writes to `reports/`:

- `ch01_metrics.csv`: per-snapshot metrics for both models.
- `ch01_h_sensitivity.csv`: the same metrics for `H` in {30, 60, 90} days.
- `ch01_monthly_churn.csv`: monthly churn rate for the main population.
- `figures/`: PR curve, reliability curve, lift by decile, monthly churn rate, and logistic coefficients. Titles state the dataset, snapshot date, `H`, and model version.

## Results

**Not yet run on the real dataset.** The pipeline has been checked on randomly generated data, which only confirms that the code runs. Those numbers are not results and are not reported here. Headline PR-AUC and lift will be filled in after running on Kaggle.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make test
make reproduce INPUT=/kaggle/input/online-retail-ii-uci/<file>.xlsx
```

## Limitation

Online Retail II is a single UK online retailer over about two years. Churn patterns here may not transfer to subscription businesses without validation, and the 90-day window is a modelling choice that the sensitivity table checks but does not settle.

## Layout

```text
src/churn/       label rule, snapshot builder, features, models, metrics, plots, pipeline
scripts/         run_baseline.py (end-to-end run)
tests/           label rule, snapshot cut (leakage), lift calculation, cleaning
reports/         metrics CSVs and figures
data/            git-ignored except data/README.md
```
