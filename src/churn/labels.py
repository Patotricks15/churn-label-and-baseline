import pandas as pd

from churn.features import rfm_features


def build_snapshot(tx: pd.DataFrame, snapshot: str | pd.Timestamp, h_days: int) -> pd.DataFrame:
    """One row per customer active before `snapshot`.

    Features use transactions in (-inf, snapshot). The label uses [snapshot, snapshot + h_days).
    """
    snapshot = pd.Timestamp(snapshot)
    window_end = snapshot + pd.Timedelta(days=h_days)
    data_end = tx["invoice_date"].max().normalize()
    if window_end > data_end:
        raise ValueError(
            f"Label window for snapshot {snapshot.date()} ends {window_end.date()}, "
            f"after the last observed date {data_end.date()}."
        )

    past = tx[tx["invoice_date"] < snapshot]
    future = tx[(tx["invoice_date"] >= snapshot) & (tx["invoice_date"] < window_end)]

    panel = rfm_features(past, snapshot)
    panel["churned"] = (~panel.index.isin(future["customer_id"].unique())).astype(int)
    panel["single_order"] = panel["frequency"] == 1
    panel["snapshot_date"] = snapshot
    return panel.rename_axis("customer_id").reset_index()
