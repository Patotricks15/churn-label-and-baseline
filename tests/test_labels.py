import numpy as np
import pandas as pd
import pytest

from churn.labels import build_snapshot
from churn.metrics import decile_table, evaluate, top_fraction_lift


def _tx(rows):
    """rows: (customer_id, invoice, date, revenue)"""
    df = pd.DataFrame(rows, columns=["customer_id", "invoice", "invoice_date", "revenue"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def test_customer_with_purchase_in_window_is_retained():
    tx = _tx([
        (1, "A1", "2010-01-05", 10.0),
        (1, "A2", "2010-03-10", 10.0),  # inside [snapshot, snapshot + 30d)
        (2, "B1", "2010-01-05", 10.0),  # no purchase in window
        (2, "B0", "2010-01-20", 10.0),
        (3, "C1", "2010-03-30", 10.0),  # active, bought in the window
        (3, "C0", "2010-01-01", 10.0),
        (4, "D0", "2010-02-01", 10.0),
        (4, "D1", "2010-03-01", 10.0),  # bought exactly on snapshot: counts in window
        (5, "E1", "2010-06-01", 10.0),  # after the observed window, provides data end
    ])
    snap = build_snapshot(tx, "2010-03-01", 30).set_index("customer_id")
    assert snap.loc[1, "churned"] == 0
    assert snap.loc[2, "churned"] == 1
    assert snap.loc[3, "churned"] == 0
    assert snap.loc[4, "churned"] == 0


def test_purchase_exactly_at_window_end_does_not_count():
    tx = _tx([
        (1, "A0", "2010-01-01", 10.0),
        (1, "A1", "2010-03-31", 10.0),  # snapshot + 30 days, outside [snapshot, snapshot + 30d)
        (2, "B0", "2010-01-01", 10.0),
        (2, "B1", "2010-03-01", 10.0),
        (9, "Z", "2010-06-01", 10.0),
    ])
    snap = build_snapshot(tx, "2010-03-01", 30).set_index("customer_id")
    assert snap.loc[1, "churned"] == 1
    assert snap.loc[2, "churned"] == 0


def test_single_order_customers_are_flagged():
    tx = _tx([
        (1, "A0", "2010-01-01", 10.0),
        (2, "B0", "2010-01-02", 10.0),
        (2, "B1", "2010-01-15", 10.0),
        (9, "Z", "2010-06-01", 10.0),
    ])
    snap = build_snapshot(tx, "2010-02-01", 30).set_index("customer_id")
    assert bool(snap.loc[1, "single_order"]) is True
    assert bool(snap.loc[2, "single_order"]) is False


def test_window_beyond_observed_data_is_rejected():
    tx = _tx([
        (1, "A0", "2010-01-01", 10.0),
        (1, "A1", "2010-02-01", 10.0),
    ])
    with pytest.raises(ValueError, match="after the last observed date"):
        build_snapshot(tx, "2010-02-01", 90)


def test_lift_top_decile_matches_hand_calculation():
    y = np.array([1] * 5 + [0] * 95)
    score = np.arange(100)[::-1]  # the first 10 scores are the highest
    # top 10% = 10 rows, of which 5 churned: rate 0.5, base 0.05, lift 10
    assert top_fraction_lift(y, score) == pytest.approx(10.0)


def test_lift_uses_ceiling_for_small_samples():
    y = np.array([1, 0, 0])
    score = np.array([0.9, 0.5, 0.1])
    # ceil(0.1 * 3) = 1 row in top decile: rate 1.0, base 1/3, lift 3
    assert top_fraction_lift(y, score) == pytest.approx(3.0)


def test_decile_table_lift_matches_top_decile():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=1000)
    score = y + rng.normal(0, 0.5, size=1000)
    table = decile_table(y, score)
    assert list(table["decile"]) == list(range(1, 11))
    assert table["n"].sum() == 1000
    assert table.iloc[0]["lift"] == pytest.approx(top_fraction_lift(y, score), rel=1e-9)


def test_evaluate_reports_brier_only_with_probabilities():
    y = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.8, 0.3, 0.6])
    assert np.isnan(evaluate(y, score)["brier"])
    assert evaluate(y, score, score)["brier"] == pytest.approx(
        np.mean((score - y) ** 2)
    )
