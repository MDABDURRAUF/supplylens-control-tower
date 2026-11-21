"""The daily briefing: what needs a decision today, and what the work so far has produced."""

from __future__ import annotations

import pandas as pd
import supplylens as sl

from .options import OptionContext, recommend
from .store import ActionStore
from .world import SyntheticWorld


def top_reason(model: sl.LateDeliveryModel, order: pd.Series) -> str:
    """The single factor raising this order's late risk the most, in plain words."""
    contrib = model.contributions(pd.DataFrame([order])).iloc[0]
    factor = contrib.idxmax()
    if contrib[factor] <= 0:
        return "no single factor stands out"
    promised_days = (pd.Timestamp(order["promised_date"]) - pd.Timestamp(order["order_date"])).days
    words = {
        "vendor": f"{order['vendor']}'s delivery record",
        "carrier": f"carrier {order['carrier']}",
        "region": f"deliveries to the {order['region']}",
        "category": f"the {order['category']} category",
        "order size": f"the large order size ({order['quantity']:,.0f} units)",
        "time of year": f"the time of year ({pd.Timestamp(order['order_date']):%B})",
        "promised lead time": f"the {promised_days}-day promised lead time",
        "expedited": "not being expedited",
    }
    return words.get(factor, factor)


def recommendations(board: pd.DataFrame, open_orders: pd.DataFrame, ctx: OptionContext, exclude=(), top_n: int | None = 5,
                    min_saving: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Orders where a response beats the plan, highest value at risk first.

    Returns a table (one row per order: what to do, the expected saving, the
    risk before and after, and the main reason for the risk) and a dict
    mapping order id to ``(recommended option row, keep row)`` for creating
    actions. Orders in ``exclude`` are skipped.
    """
    exclude = set(exclude)
    rows, picks = [], {}
    for _, b in board.iterrows():
        if b["order_id"] in exclude:
            continue
        order = open_orders[open_orders["order_id"] == b["order_id"]].iloc[0]
        opts = ctx.options(order, b["stage"])
        rec = recommend(opts, min_saving=min_saving)
        if rec["kind"] == "keep":
            continue
        keep = opts[opts["kind"] == "keep"].iloc[0]
        picks[b["order_id"]] = (rec, keep)
        rows.append(
            {
                "order_id": b["order_id"], "priority": b["priority"], "stage": b["stage"], "vendor": b["vendor"],
                "sku": b["sku"], "quantity": b["quantity"], "goods_value": b["goods_value"],
                "value_at_risk": b["value_at_risk"], "recommendation": rec["option"], "kind": rec["kind"],
                "expected_saving": rec["saving_vs_keep"], "risk_before": keep["late_probability"],
                "risk_after": rec["late_probability"], "main_reason": top_reason(ctx.model, order),
            }
        )
        if top_n is not None and len(rows) >= top_n:
            break
    return pd.DataFrame(rows), picks


def whats_new(orders: pd.DataFrame, as_of, days: int = 1) -> dict:
    """What happened in the last ``days`` days: orders placed and deliveries received."""
    as_of = pd.Timestamp(as_of)
    since = as_of - pd.Timedelta(days=days)
    placed = orders[(orders["order_date"] > since) & (orders["order_date"] <= as_of)]
    arrived = orders[(orders["delivery_date"] > since) & (orders["delivery_date"] <= as_of)]
    late = arrived[arrived["delivery_date"] > arrived["promised_date"]]
    return {
        "orders_placed": int(len(placed)),
        "value_placed": float((placed["quantity"] * placed["unit_price"]).sum()),
        "deliveries": int(len(arrived)),
        "late_deliveries": int(len(late)),
    }


def weekly_on_time(history: pd.DataFrame, weeks: int = 12) -> pd.DataFrame:
    """On-time rate of deliveries per week, for the most recent ``weeks`` weeks."""
    h = history.assign(on_time=history["delivery_date"] <= history["promised_date"])
    weekly = h.groupby(pd.Grouper(key="delivery_date", freq="W-MON", label="left", closed="left")).agg(
        deliveries=("order_id", "size"), on_time_rate=("on_time", "mean")
    )
    return weekly[weekly["deliveries"] > 0].tail(weeks).reset_index().rename(columns={"delivery_date": "week"})


def pipeline_counts(board: pd.DataFrame, actions: pd.DataFrame, ledger: pd.DataFrame) -> dict:
    """How much work sits at each stage of the loop: monitor, decide, track, record, review."""
    with_action = set(actions["order_id"]) if len(actions) else set()
    at_risk = board[board["priority"].isin(["Critical", "High"])] if len(board) else board
    recorded = ledger["final_cost"].notna() if len(ledger) else pd.Series(dtype=bool)
    return {
        "needs_attention": int((~at_risk["order_id"].isin(with_action)).sum()) if len(at_risk) else 0,
        "awaiting_decision": int((actions["decision"] == "pending").sum()) if len(actions) else 0,
        "in_progress": int(((actions["decision"] == "approved") & actions["status"].isin(["open", "in progress"])).sum())
        if len(actions) else 0,
        "overdue": int(actions["overdue"].sum()) if len(actions) and "overdue" in actions else 0,
        "awaiting_outcome": int(((ledger["decision"] != "pending") & ~recorded).sum()) if len(ledger) else 0,
        "outcomes_recorded": int(recorded.sum()) if len(ledger) else 0,
    }


def results_to_date(ledger: pd.DataFrame) -> dict:
    """Headline results of every outcome recorded so far."""
    done = ledger[ledger["final_cost"].notna()] if len(ledger) else ledger
    adopted = done[done["adopted"] == 1] if len(done) else done
    known = done[done["baseline_cost"].notna()] if len(done) else done
    return {
        "outcomes": int(len(done)),
        "adopted": int(len(adopted)),
        "estimated_saving": float(adopted["est_saving"].sum()) if len(adopted) else 0.0,
        "realized_saving": float(known["realized_saving"].sum()) if len(known) else 0.0,
        "observed_cost": float(done["final_cost"].sum()) if len(done) else 0.0,
        "on_time_adopted": float(1 - adopted["late"].mean()) if len(adopted) else float("nan"),
        "on_time_baseline": float(1 - known["baseline_late"].mean()) if len(known) else float("nan"),
        "late_deliveries_avoided": int((known["baseline_late"] - known["late"]).sum()) if len(known) else 0,
    }


def record_simulated_outcomes(world: SyntheticWorld, store: ActionStore, as_of, late_cost_pct: float) -> int:
    """Record the outcome of every decided action whose order has arrived by ``as_of``.

    The outcome is what the simulated world produced for the decision that
    was followed (the action's option if approved, the original plan if
    rejected), together with the baseline outcome. Returns how many were
    recorded.
    """
    as_of = pd.Timestamp(as_of)
    ledger = store.ledger()
    if ledger.empty:
        return 0
    recorded = 0
    for _, a in ledger[ledger["final_cost"].isna() & (ledger["decision"] != "pending")].iterrows():
        adopted = a["decision"] == "approved"
        baseline = world.baseline_option(a["order_id"])
        followed = {k: a[k] for k in ("vendor", "carrier", "expedited", "split_vendor")} if adopted else baseline
        result = world.realize_option(a["order_id"], followed, late_cost_pct)
        if result["delivery_date"] > as_of:
            continue
        base = world.realize_option(a["order_id"], baseline, late_cost_pct)
        store.record_outcome(int(a["action_id"]), adopted, result["delivery_date"], result["late"], result["total_cost"],
                             as_of, baseline_cost=base["total_cost"], baseline_late=base["late"], source="simulation")
        recorded += 1
    return recorded


def run_autopilot(world: SyntheticWorld, store: ActionStore, as_of, weeks: int, owner: str = "Autopilot",
                  late_cost_pct: float = 0.10, min_saving: float = 0.0, progress=None) -> dict:
    """Follow every recommendation for ``weeks`` weeks and record what happens.

    Each week: move the clock forward seven days, retrain on what has been
    delivered, create and approve the recommended action for every order
    placed that week (where a response beats the plan), then record outcomes
    for orders that have arrived. The clock position is saved in the store.
    """
    from .monitor import fit_model, split_as_of  # local import avoids a cycle

    as_of = pd.Timestamp(as_of)
    last = world.orders["order_date"].max()
    created = recorded = 0
    for week in range(weeks):
        previous, as_of = as_of, min(as_of + pd.Timedelta(days=7), last)
        history, _ = split_as_of(world.orders, as_of)
        placed = world.orders[(world.orders["order_date"] > previous) & (world.orders["order_date"] <= as_of)]
        existing = store.actions()
        placed = placed[~placed["order_id"].isin(existing["order_id"])] if len(existing) else placed
        if len(placed):
            ctx = OptionContext.build(fit_model(history), history, late_cost_pct=late_cost_pct)
            for _, order in placed.drop(columns=["delivery_date", "units_rejected"]).iterrows():
                opts = ctx.options(order, "pending release")
                rec = recommend(opts, min_saving=min_saving)
                if rec["kind"] == "keep":
                    continue
                aid = store.add_action(order["order_id"], rec, opts[opts["kind"] == "keep"].iloc[0], owner,
                                       order["order_date"] + pd.Timedelta(days=2), order["order_date"],
                                       note="autopilot", decision="approved")
                store.update_action(aid, status="done")
                created += 1
        recorded += record_simulated_outcomes(world, store, as_of, late_cost_pct)
        store.set_meta("as_of", str(as_of.date()))
        if progress is not None:
            progress((week + 1) / weeks)
        if as_of >= last:
            break
    return {"as_of": as_of, "actions_created": created, "outcomes_recorded": recorded}
