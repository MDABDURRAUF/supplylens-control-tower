"""Monitor open orders: who is at risk of arriving late or causing a stockout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import supplylens as sl
from scipy.stats import norm

PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]


def split_as_of(orders: pd.DataFrame, as_of) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split orders into what is known and what is still open on ``as_of``.

    ``history`` is every order already delivered by ``as_of``; its outcome is
    known. ``open_orders`` were placed on or before ``as_of`` and have not
    arrived; their delivery date and reject count are removed, so nothing
    downstream can accidentally use information from the future.
    """
    as_of = pd.Timestamp(as_of)
    placed = orders[orders["order_date"] <= as_of]
    history = placed[placed["delivery_date"] <= as_of].reset_index(drop=True)
    open_orders = placed[placed["delivery_date"] > as_of].drop(columns=["delivery_date", "units_rejected"])
    return history, open_orders.reset_index(drop=True)


def fit_model(history: pd.DataFrame) -> sl.LateDeliveryModel:
    """Train the SupplyLens late-delivery model on delivered orders only.

    If the history records which shipments were expedited, that flag is used
    as a predictor, so the effect of expediting is learned from the data
    instead of being assumed.
    """
    extra = ("expedited",) if "expedited" in history.columns and history["expedited"].nunique() > 1 else ()
    return sl.train_late_delivery_model(history, extra_numeric=extra)


def open_order_board(
    model: sl.LateDeliveryModel,
    history: pd.DataFrame,
    open_orders: pd.DataFrame,
    as_of,
    on_hand: pd.Series | dict | None = None,
    late_cost_pct: float = 0.10,
    stockout_cost_pct: float = 0.25,
    risk_threshold: float = 0.30,
    review_days: int = 3,
) -> pd.DataFrame:
    """Rank open orders by the money at risk.

    For each open order:

    ``stage``
        ``pending release`` if placed within ``review_days`` of ``as_of``
        (supplier and carrier can still be changed), ``overdue`` if the
        promised date has passed, otherwise ``in transit``.
    ``late_probability``
        From the model; 1.0 for orders that are already overdue.
    ``stockout_probability``
        Chance the SKU runs out before this order arrives *if it is late by
        the typical delay*. Stock available is what is on hand plus other open
        orders for the SKU promised earlier; demand over the wait is treated
        as normal with the SKU's historical daily mean and spread. NaN when
        no stock level is given for the SKU.
    ``value_at_risk``
        goods value x late probability x (``late_cost_pct`` +
        ``stockout_cost_pct`` x stockout probability). This is the ranking key.
    ``priority``
        Critical: late probability at or above ``risk_threshold`` and
        stockout probability at least 50%. High: late probability at or above
        the threshold. Medium: late probability at least half the threshold,
        or stockout probability at least 50%. Otherwise Low.
    """
    as_of = pd.Timestamp(as_of)
    if open_orders.empty:
        return pd.DataFrame()
    board = open_orders.copy()
    board["goods_value"] = board["quantity"] * board["unit_price"]
    board["days_to_due"] = (board["promised_date"] - as_of).dt.days
    board["stage"] = np.select(
        [board["days_to_due"] < 0, (as_of - board["order_date"]).dt.days <= review_days],
        ["overdue", "pending release"],
        default="in transit",
    )
    board["late_probability"] = model.predict_proba(open_orders).to_numpy()
    board.loc[board["stage"] == "overdue", "late_probability"] = 1.0

    # Stockout risk if the order arrives late by the typical delay.
    hist = sl.data.add_derived_columns(history)
    typical_delay = float(hist.loc[hist["is_late"], "days_late"].mean()) if hist["is_late"].any() else 0.0
    demand = sl.inventory_policy(history).set_index("sku")
    stock = pd.Series(on_hand, dtype=float) if on_hand is not None else pd.Series(dtype=float)

    board = board.sort_values(["sku", "promised_date", "order_id"])
    inbound_before = board.groupby("sku")["quantity"].cumsum() - board["quantity"]
    available = board["sku"].map(stock) + inbound_before
    wait_days = (board["days_to_due"] + typical_delay).clip(lower=1)
    mean_demand = board["sku"].map(demand["avg_daily_demand"]) * wait_days
    sd_demand = (board["sku"].map(demand["sd_daily_demand"]) * np.sqrt(wait_days)).replace(0, np.nan)
    z = (available - mean_demand) / sd_demand
    board["on_hand"] = board["sku"].map(stock)
    board["stockout_probability"] = np.where(z.isna() & available.notna(), (available < mean_demand).astype(float), norm.sf(z))
    board.loc[available.isna(), "stockout_probability"] = np.nan

    p_stock = board["stockout_probability"].fillna(0.0)
    board["value_at_risk"] = board["goods_value"] * board["late_probability"] * (late_cost_pct + stockout_cost_pct * p_stock)
    board["priority"] = np.select(
        [
            (board["late_probability"] >= risk_threshold) & (p_stock >= 0.5),
            board["late_probability"] >= risk_threshold,
            (board["late_probability"] >= risk_threshold / 2) | (p_stock >= 0.5),
        ],
        ["Critical", "High", "Medium"],
        default="Low",
    )
    cols = ["order_id", "priority", "stage", "order_date", "promised_date", "days_to_due", "vendor", "carrier", "region",
            "category", "sku", "quantity", "unit_price", "freight_cost", "expedited", "goods_value", "late_probability",
            "on_hand", "stockout_probability", "value_at_risk"]
    return board[[c for c in cols if c in board.columns]].sort_values("value_at_risk", ascending=False).reset_index(drop=True)
