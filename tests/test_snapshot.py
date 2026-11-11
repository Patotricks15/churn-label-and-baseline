import pandas as pd

from churn.data import clean_transactions
from churn.features import rfm_features
from churn.labels import build_snapshot


def _raw(rows):
    return pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
    )


def test_clean_drops_cancellations_returns_and_unknown_customers():
    raw = _raw([
        ["100", "X", "d", 2, "2010-01-01 10:00", 5.0, 1.0, "UK"],
        ["C101", "X", "d", -1, "2010-01-02 10:00", 5.0, 1.0, "UK"],   # cancellation
        ["102", "X", "d", -3, "2010-01-03 10:00", 5.0, 1.0, "UK"],    # return line
        ["103", "X", "d", 1, "2010-01-04 10:00", 0.0, 1.0, "UK"],     # zero price
        ["104", "X", "d", 1, "2010-01-05 10:00", 5.0, None, "UK"],    # no customer id
    ])
    clean = clean_transactions(raw)
    assert list(clean["invoice"]) == ["100"]
    assert clean.loc[0, "revenue"] == 10.0
    assert clean["customer_id"].dtype == "int64"


def test_features_use_only_rows_before_snapshot():
    tx = clean_transactions(_raw([
        ["100", "X", "d", 1, "2010-01-10 10:00", 10.0, 1.0, "UK"],
        ["101", "X", "d", 1, "2010-02-10 10:00", 10.0, 1.0, "UK"],
    ]))
    feats = rfm_features(tx[tx["invoice_date"] < "2010-02-01"], pd.Timestamp("2010-02-01"))
    assert feats.loc[1, "frequency"] == 1
    assert feats.loc[1, "recency_days"] == 21  # whole days from 2010-01-10 10:00 to the snapshot
    assert feats.loc[1, "monetary"] == 10.0


def test_snapshot_cut_is_not_affected_by_future_transactions():
    base = [
        ["100", "X", "d", 1, "2010-01-10 10:00", 10.0, 1.0, "UK"],
        ["101", "X", "d", 2, "2010-01-20 10:00", 10.0, 2.0, "UK"],
        ["102", "X", "d", 1, "2010-06-01 10:00", 10.0, 2.0, "UK"],
        ["103", "X", "d", 1, "2010-06-05 10:00", 10.0, 3.0, "UK"],
    ]
    future_extra = [
        ["104", "X", "d", 50, "2010-02-15 10:00", 99.0, 1.0, "UK"],
        ["105", "X", "d", 1, "2010-03-01 09:00", 99.0, 2.0, "UK"],
    ]
    tx_a = clean_transactions(_raw(base))
    tx_b = clean_transactions(_raw(base + future_extra))

    snap_a = build_snapshot(tx_a, "2010-02-01", 60).set_index("customer_id")
    snap_b = build_snapshot(tx_b, "2010-02-01", 60).set_index("customer_id")

    feature_cols = ["recency_days", "frequency", "monetary", "tenure_days", "avg_order_value"]
    pd.testing.assert_frame_equal(snap_a.loc[[1, 2], feature_cols], snap_b.loc[[1, 2], feature_cols])
    # The label does see the future: customer 1 buys again inside the window in tx_b.
    assert snap_a.loc[1, "churned"] == 1
    assert snap_b.loc[1, "churned"] == 0
