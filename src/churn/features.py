import pandas as pd

FEATURE_COLUMNS = ["recency_days", "frequency", "monetary", "tenure_days", "avg_order_value"]


def rfm_features(past: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """RFM-style features from transactions strictly before the snapshot."""
    snapshot = pd.Timestamp(snapshot)
    grouped = past.groupby("customer_id")
    out = pd.DataFrame(
        {
            "recency_days": (snapshot - grouped["invoice_date"].max()).dt.days,
            "frequency": grouped["invoice"].nunique(),
            "monetary": grouped["revenue"].sum(),
            "tenure_days": (snapshot - grouped["invoice_date"].min()).dt.days,
        }
    )
    out["avg_order_value"] = out["monetary"] / out["frequency"]
    return out
