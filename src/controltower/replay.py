"""Replay history to test a decision policy against a simple baseline.

The replay walks forward through time. At each step it knows only what had
been delivered by then, trains the model on that, and decides what to do with
the orders placed since the previous step. Each decision is then played out in
the simulated world, together with the baseline of keeping the original
supplier and carrier, so both outcomes are observed for every order.

Three things are reported and kept apart:

* **estimated saving**  what the model expected at decision time;
* **realized saving**   baseline cost minus policy cost as actually played out;
* **observed cost**     what the chosen option actually cost.

The realized saving is only observable because the world is simulated. With
real orders the baseline outcome of a changed order is never seen.
"""

from __future__ import annotations

import pandas as pd

from .monitor import fit_model, split_as_of
from .options import OptionContext, recommend
from .world import SyntheticWorld


def run_policy_replay(
    world: SyntheticWorld,
    start,
    end,
    step_days: int = 7,
    late_cost_pct: float = 0.10,
    min_saving: float = 0.0,
    min_history: int = 300,
    progress=None,
) -> pd.DataFrame:
    """Replay ``start``..``end`` and return one row per order decided.

    Policy: for each newly placed order, follow ``recommend`` (lowest expected
    cost among options no riskier than the plan, if it saves more than
    ``min_saving``); otherwise keep the plan. Baseline: always keep the plan.

    Each decision is evaluated independently: a changed order does not alter
    later orders or the history the model is trained on.

    ``progress`` is an optional callable taking a fraction between 0 and 1.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    steps = pd.date_range(start, end, freq=f"{step_days}D")
    if len(steps) < 2:
        raise ValueError("The replay period must cover at least two steps")
    rows = []
    for i, as_of in enumerate(steps[1:], start=1):
        history, _ = split_as_of(world.orders, as_of)
        new = world.orders[(world.orders["order_date"] > steps[i - 1]) & (world.orders["order_date"] <= as_of)]
        if len(history) >= min_history and not new.empty:
            ctx = OptionContext.build(fit_model(history), history, late_cost_pct=late_cost_pct)
            for _, order in new.drop(columns=["delivery_date", "units_rejected"]).iterrows():
                opts = ctx.options(order, stage="pending release")
                keep = opts[opts["kind"] == "keep"].iloc[0]
                choice = recommend(opts, min_saving=min_saving)
                base = world.realize_option(order["order_id"], world.baseline_option(order["order_id"]), late_cost_pct)
                actual = world.realize_option(order["order_id"], choice, late_cost_pct)
                rows.append(
                    {
                        "decided_on": as_of,
                        "order_id": order["order_id"],
                        "option": choice["option"],
                        "kind": choice["kind"],
                        "changed": choice["kind"] != "keep",
                        "est_keep_cost": keep["expected_cost"],
                        "est_policy_cost": choice["expected_cost"],
                        "est_saving": keep["expected_cost"] - choice["expected_cost"],
                        "est_keep_risk": keep["late_probability"],
                        "est_policy_risk": choice["late_probability"],
                        "baseline_cost": base["total_cost"],
                        "policy_cost": actual["total_cost"],
                        "realized_saving": base["total_cost"] - actual["total_cost"],
                        "baseline_late_share": base["late_share"],
                        "policy_late_share": actual["late_share"],
                        "goods_value": base["goods_cost"],
                    }
                )
        if progress is not None:
            progress(i / (len(steps) - 1))
    return pd.DataFrame(rows)


def summarize_replay(replay: pd.DataFrame) -> dict:
    """Headline comparison of the policy with the baseline.

    On-time rates are weighted by goods value. ``estimate_ratio`` is realized
    saving divided by estimated saving: 1.0 means the model's promise was
    kept exactly, below 1.0 means it over-promised.
    """
    if replay.empty:
        raise ValueError("The replay produced no decisions")
    value = replay["goods_value"].sum()
    est = float(replay["est_saving"].sum())
    realized = float(replay["realized_saving"].sum())
    return {
        "orders": int(len(replay)),
        "orders_changed": int(replay["changed"].sum()),
        "share_changed": float(replay["changed"].mean()),
        "baseline_cost": float(replay["baseline_cost"].sum()),
        "policy_cost": float(replay["policy_cost"].sum()),
        "estimated_saving": est,
        "realized_saving": realized,
        "realized_saving_pct": realized / float(replay["baseline_cost"].sum()),
        "estimate_ratio": realized / est if est else float("nan"),
        "baseline_on_time": float(1 - (replay["baseline_late_share"] * replay["goods_value"]).sum() / value),
        "policy_on_time": float(1 - (replay["policy_late_share"] * replay["goods_value"]).sum() / value),
    }


def replay_by_option(replay: pd.DataFrame) -> pd.DataFrame:
    """Estimated and realized saving for each kind of decision."""
    g = replay.groupby("kind")
    out = pd.DataFrame(
        {
            "orders": g.size(),
            "estimated_saving": g["est_saving"].sum(),
            "realized_saving": g["realized_saving"].sum(),
            "est_risk_before": g["est_keep_risk"].mean(),
            "est_risk_after": g["est_policy_risk"].mean(),
            "late_share_baseline": g["baseline_late_share"].mean(),
            "late_share_policy": g["policy_late_share"].mean(),
        }
    )
    return out.sort_values("realized_saving", ascending=False).reset_index()


def review_actions(ledger: pd.DataFrame) -> dict:
    """Performance of the actions an analyst recorded, from the store's ledger.

    Compares adopted with not-adopted actions on observed results. This is a
    description, not proof of cause: the two groups can differ for reasons
    other than the decision. ``realized_saving`` is filled only where the
    baseline outcome was recorded (simulation).
    """
    done = ledger[ledger["final_cost"].notna()]
    out = {
        "actions": int(len(ledger)),
        "with_outcome": int(len(done)),
        "adopted": int((done["adopted"] == 1).sum()),
        "estimated_saving_adopted": float(done.loc[done["adopted"] == 1, "est_saving"].sum()),
        "observed_cost": float(done["final_cost"].sum()),
        "mean_estimate_error": float(done["est_error"].mean()) if len(done) else float("nan"),
        "realized_saving_known": float(done["realized_saving"].sum(skipna=True)),
        "outcomes_with_baseline": int(done["baseline_cost"].notna().sum()),
    }
    for flag, name in [(1, "adopted"), (0, "not_adopted")]:
        grp = done[done["adopted"] == flag]
        out[f"on_time_{name}"] = float(1 - grp["late"].mean()) if len(grp) else float("nan")
    return out
