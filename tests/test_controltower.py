import numpy as np
import pandas as pd
import pytest

import controltower as ct
from controltower.world import EXPEDITE_COST_MULTIPLIER

AS_OF = pd.Timestamp("2025-06-30")
LATE_COST = 0.10


@pytest.fixture(scope="module")
def world():
    return ct.SyntheticWorld(seed=7, n_orders=4000)


@pytest.fixture(scope="module")
def snapshot(world):
    history, open_orders = ct.split_as_of(world.orders, AS_OF)
    model = ct.fit_model(history)
    ctx = ct.OptionContext.build(model, history, late_cost_pct=LATE_COST)
    board = ct.open_order_board(model, history, open_orders, AS_OF, on_hand=world.stock_snapshot(AS_OF, history))
    return history, open_orders, model, ctx, board


def order_row(open_orders, order_id):
    return open_orders[open_orders["order_id"] == order_id].iloc[0]


# ---------------------------------------------------------------------- world
def test_world_is_reproducible_and_history_matches_realize(world):
    again = ct.SyntheticWorld(seed=7, n_orders=4000)
    pd.testing.assert_frame_equal(world.orders, again.orders)
    for _, o in world.orders.sample(25, random_state=0).iterrows():
        r = world.realize(o["order_id"])
        assert r["delivery_date"] == o["delivery_date"] and r["freight_cost"] == o["freight_cost"]
        assert r["unit_price"] == o["unit_price"] and r["late"] == (o["delivery_date"] > o["promised_date"])


def test_expediting_never_makes_an_order_later_and_costs_more(world):
    normal = world.orders[world.orders["expedited"] == 0].head(300)
    flipped = 0
    for oid in normal["order_id"]:
        base, fast = world.realize(oid), world.realize(oid, expedited=1)
        # Same order, same luck: expediting can only turn late into on time.
        assert not (fast["late"] and not base["late"])
        assert fast["freight_cost"] == pytest.approx(base["freight_cost"] * EXPEDITE_COST_MULTIPLIER, abs=0.02)
        assert world.true_late_probability(oid, expedited=1) < world.true_late_probability(oid)
        flipped += base["late"] and not fast["late"]
    assert flipped > 0


def test_realize_rejects_impossible_decisions(world):
    oid = world.orders.loc[world.orders["category"] == "Electronics", "order_id"].iloc[0]
    outsider = next(v for v in world.vendors if v not in world.vendors_for("Electronics"))
    with pytest.raises(ValueError, match="does not supply"):
        world.realize(oid, vendor=outsider)
    with pytest.raises(ValueError, match="Unknown carrier"):
        world.realize(oid, carrier="Nobody Freight")


def test_realize_option_costs_add_up_for_a_split(world):
    o = world.orders[world.orders["category"] != "Electronics"].iloc[0]
    other = next(v for v in world.vendors_for(o["category"]) if v != o["vendor"])
    option = {"vendor": o["vendor"], "carrier": o["carrier"], "expedited": 0, "split_vendor": other}
    res = world.realize_option(o["order_id"], option, LATE_COST)
    a = world.realize(o["order_id"], o["vendor"], o["carrier"], 0, o["quantity"] / 2, leg=1)
    b = world.realize(o["order_id"], other, o["carrier"], 0, o["quantity"] / 2, leg=2)
    assert res["goods_cost"] == pytest.approx(a["goods_cost"] + b["goods_cost"])
    late_goods = a["goods_cost"] * a["late"] + b["goods_cost"] * b["late"]
    assert res["total_cost"] == pytest.approx(res["goods_cost"] + res["freight_cost"] + LATE_COST * late_goods)
    assert res["late_share"] == pytest.approx(late_goods / res["goods_cost"])
    # The baseline option reproduces the recorded order.
    base = world.realize_option(o["order_id"], world.baseline_option(o["order_id"]), LATE_COST)
    assert base["delivery_date"] == o["delivery_date"]


# -------------------------------------------------------------------- monitor
def test_split_as_of_hides_the_future(world):
    history, open_orders = ct.split_as_of(world.orders, AS_OF)
    assert (history["delivery_date"] <= AS_OF).all()
    assert "delivery_date" not in open_orders.columns and "units_rejected" not in open_orders.columns
    assert (open_orders["order_date"] <= AS_OF).all()
    placed = (world.orders["order_date"] <= AS_OF).sum()
    assert len(history) + len(open_orders) == placed
    assert not set(history["order_id"]) & set(open_orders["order_id"])


def test_model_learns_that_expediting_helps(snapshot):
    _, _, model, _, _ = snapshot
    assert model.extra_numeric == ("expedited",)
    drivers = model.drivers().set_index("feature")
    assert drivers.loc["expedited", "odds_ratio"] < 1
    assert model.metrics["roc_auc"] > 0.6 and model.metrics["brier_score"] < model.metrics["brier_baseline"]


def test_board_stages_priorities_and_ranking(snapshot):
    _, open_orders, _, _, board = snapshot
    assert len(board) == len(open_orders)
    assert board["value_at_risk"].is_monotonic_decreasing
    assert set(board["priority"]) <= set(ct.PRIORITY_ORDER)
    overdue = board[board["stage"] == "overdue"]
    assert (overdue["days_to_due"] < 0).all() and (overdue["late_probability"] == 1.0).all()
    pending = board[board["stage"] == "pending release"]
    assert ((AS_OF - pending["order_date"]).dt.days <= 3).all()
    assert board["stockout_probability"].between(0, 1).all()
    critical = board[board["priority"] == "Critical"]
    assert ((critical["late_probability"] >= 0.30) & (critical["stockout_probability"] >= 0.5)).all()
    row = board.iloc[0]
    expected = row["goods_value"] * row["late_probability"] * (0.10 + 0.25 * row["stockout_probability"])
    assert row["value_at_risk"] == pytest.approx(expected)


def test_more_stock_means_less_stockout_risk(snapshot, world):
    history, open_orders, model, _, board = snapshot
    plenty = world.stock_snapshot(AS_OF, history) * 100
    rich = ct.open_order_board(model, history, open_orders, AS_OF, on_hand=plenty)
    assert rich["stockout_probability"].max() < 0.01
    none = ct.open_order_board(model, history, open_orders, AS_OF)
    assert none["stockout_probability"].isna().all()
    assert ct.open_order_board(model, history, open_orders.iloc[0:0], AS_OF).empty


# -------------------------------------------------------------------- options
def test_options_for_a_pending_order(snapshot):
    _, open_orders, _, ctx, board = snapshot
    pending = board[(board["stage"] == "pending release") & (board["category"] != "Electronics") & (board["expedited"] == 0)]
    order = order_row(open_orders, pending.iloc[0]["order_id"])
    opts = ctx.options(order, "pending release")
    assert opts["expected_cost"].is_monotonic_increasing
    assert {"keep", "expedite", "carrier", "supplier", "split"} <= set(opts["kind"])
    np.testing.assert_allclose(opts["expected_cost"], opts[["goods_cost", "freight_cost", "expected_late_cost"]].sum(axis=1))
    np.testing.assert_allclose(opts["expected_late_cost"], opts["late_probability"] * LATE_COST * opts["goods_cost"])
    keep = opts[opts["kind"] == "keep"].iloc[0]
    assert keep["saving_vs_keep"] == 0 and keep["risk_change"] == 0
    # The plan is costed from the order itself, not from estimates.
    assert keep["goods_cost"] == pytest.approx(order["quantity"] * order["unit_price"])
    assert keep["freight_cost"] == pytest.approx(order["freight_cost"])
    fast = opts[opts["kind"] == "expedite"].iloc[0]
    assert fast["late_probability"] < keep["late_probability"] and fast["freight_cost"] > keep["freight_cost"]
    assert (opts.loc[opts["kind"] == "split", "split_vendor"] != order["vendor"]).all()


def test_in_transit_orders_can_only_be_expedited(snapshot):
    _, open_orders, _, ctx, board = snapshot
    moving = board[(board["stage"] == "in transit") & (board["expedited"] == 0)]
    opts = ctx.options(order_row(open_orders, moving.iloc[0]["order_id"]), "in transit")
    assert set(opts["kind"]) == {"keep", "expedite"}
    already = board[board["expedited"] == 1]
    if len(already):
        row = already.iloc[0]
        assert "expedite" not in set(ctx.options(order_row(open_orders, row["order_id"]), row["stage"])["kind"])


def test_recommendation_is_never_riskier_and_respects_min_saving(snapshot):
    _, open_orders, _, ctx, board = snapshot
    changed = 0
    for oid in board.loc[board["stage"] == "pending release", "order_id"]:
        opts = ctx.options(order_row(open_orders, oid), "pending release")
        rec = ct.recommend(opts)
        assert rec["risk_change"] <= 0 and rec["saving_vs_keep"] >= 0
        changed += rec["kind"] != "keep"
        assert ct.recommend(opts, min_saving=1e9)["kind"] == "keep"
    assert changed > 0


# ---------------------------------------------------------------------- store
def test_action_and_outcome_round_trip(snapshot):
    _, open_orders, _, ctx, board = snapshot
    oid = board.loc[board["stage"] == "pending release", "order_id"].iloc[0]
    opts = ctx.options(order_row(open_orders, oid), "pending release")
    keep, option = opts[opts["kind"] == "keep"].iloc[0], opts.iloc[0]

    store = ct.ActionStore(":memory:")
    aid = store.add_action(oid, option, keep, owner=" Dana ", due_date="2025-07-02", created_on=AS_OF, note="check price")
    actions = store.actions(as_of=AS_OF)
    assert len(actions) == 1 and actions.loc[0, "owner"] == "Dana" and actions.loc[0, "decision"] == "pending"
    assert actions.loc[0, "est_saving"] == pytest.approx(keep["expected_cost"] - option["expected_cost"])
    assert not actions.loc[0, "overdue"]
    assert store.actions(as_of="2025-07-10").loc[0, "overdue"]

    store.update_action(aid, decision="approved", status="done")
    assert not store.actions(as_of="2025-07-10").loc[0, "overdue"]  # finished actions are never overdue
    with pytest.raises(ValueError):
        store.update_action(aid, status="finished")
    with pytest.raises(ValueError):
        store.update_action(aid, est_keep_cost=0)
    with pytest.raises(KeyError):
        store.update_action(999, status="done")
    with pytest.raises(ValueError, match="owner"):
        store.add_action(oid, option, keep, owner="  ", due_date="2025-07-02", created_on=AS_OF)

    ledger = store.ledger()
    assert pd.isna(ledger.loc[0, "final_cost"])
    cost = float(option["expected_cost"]) + 50
    store.record_outcome(aid, adopted=True, delivery_date="2025-07-12", late=False, final_cost=cost,
                         recorded_on="2025-07-13", baseline_cost=cost + 120, baseline_late=True, source="simulation")
    row = store.ledger().iloc[0]
    assert row["est_error"] == pytest.approx(50) and row["realized_saving"] == pytest.approx(120)
    store.record_outcome(aid, adopted=False, delivery_date="2025-07-12", late=True, final_cost=cost, recorded_on="2025-07-13")
    row = store.ledger().iloc[0]  # replaced, and now compared with the plan's estimate
    assert row["adopted"] == 0 and row["est_error"] == pytest.approx(cost - keep["expected_cost"])
    assert pd.isna(row["realized_saving"])

    review = ct.review_actions(store.ledger())
    assert review["with_outcome"] == 1 and review["adopted"] == 0 and review["outcomes_with_baseline"] == 0
    store.set_meta("as_of", "2025-07-13")
    assert store.get_meta("as_of") == "2025-07-13" and store.get_meta("missing", "x") == "x"
    store.reset()
    assert store.actions().empty


def test_store_persists_to_disk(tmp_path, snapshot):
    _, open_orders, _, ctx, board = snapshot
    oid = board["order_id"].iloc[0]
    opts = ctx.options(order_row(open_orders, oid), board["stage"].iloc[0])
    path = tmp_path / "ct.db"
    first = ct.ActionStore(path)
    first.add_action(oid, opts.iloc[0], opts[opts["kind"] == "keep"].iloc[0], "Sam", "2025-07-05", AS_OF)
    first.close()
    assert len(ct.ActionStore(path).actions()) == 1


# --------------------------------------------------------------------- replay
@pytest.fixture(scope="module")
def replay(world):
    return ct.run_policy_replay(world, "2025-03-01", "2025-06-28", late_cost_pct=LATE_COST)


def test_replay_decisions_use_only_the_past(world, replay):
    assert len(replay) > 200 and replay["order_id"].is_unique
    dates = world.orders.set_index("order_id")["order_date"].reindex(replay["order_id"]).to_numpy()
    assert (dates <= replay["decided_on"].to_numpy()).all()
    kept = replay[~replay["changed"]]
    # Keeping the plan is exactly the baseline, so nothing is saved or lost.
    assert (kept["realized_saving"] == 0).all() and (kept["est_saving"] == 0).all()
    changed = replay[replay["changed"]]
    assert (changed["est_saving"] > 0).all() and (changed["est_policy_risk"] <= changed["est_keep_risk"] + 1e-12).all()


def test_replay_summary_is_consistent(replay):
    s = ct.summarize_replay(replay)
    assert s["orders"] == len(replay)
    assert s["realized_saving"] == pytest.approx(s["baseline_cost"] - s["policy_cost"])
    assert s["estimate_ratio"] == pytest.approx(s["realized_saving"] / s["estimated_saving"])
    # On this simulated world the policy should actually help, not only on paper.
    assert s["realized_saving"] > 0 and s["policy_on_time"] > s["baseline_on_time"]
    by = ct.replay_by_option(replay)
    assert by["orders"].sum() == len(replay)
    assert by["realized_saving"].sum() == pytest.approx(s["realized_saving"])


def test_min_saving_reduces_the_number_of_changes(world, replay):
    strict = ct.run_policy_replay(world, "2025-05-01", "2025-06-28", min_saving=200)
    loose = replay[replay["decided_on"] > "2025-05-01"]
    assert strict["changed"].mean() < loose["changed"].mean()
    with pytest.raises(ValueError, match="at least two steps"):
        ct.run_policy_replay(world, "2025-06-01", "2025-06-03")
    with pytest.raises(ValueError):
        ct.summarize_replay(replay.iloc[0:0])


# ------------------------------------------------- briefing, autopilot, outputs
def test_recommendations_table_matches_the_options(snapshot):
    _, open_orders, _, ctx, board = snapshot
    recs, picks = ct.recommendations(board, open_orders, ctx, top_n=None)
    assert len(recs) > 0 and set(picks) == set(recs["order_id"])
    assert (recs["expected_saving"] > 0).all() and (recs["risk_after"] <= recs["risk_before"] + 1e-12).all()
    assert recs["value_at_risk"].is_monotonic_decreasing  # same order as the board
    first = recs.iloc[0]
    rec_row, keep_row = picks[first["order_id"]]
    assert rec_row["option"] == first["recommendation"] and keep_row["kind"] == "keep"
    assert first["expected_saving"] == pytest.approx(keep_row["expected_cost"] - rec_row["expected_cost"])
    assert isinstance(first["main_reason"], str) and first["main_reason"]
    # Limits and exclusions.
    assert len(ct.recommendations(board, open_orders, ctx, top_n=2)[0]) == 2
    skipped, _ = ct.recommendations(board, open_orders, ctx, exclude=[first["order_id"]], top_n=None)
    assert first["order_id"] not in set(skipped["order_id"]) and len(skipped) == len(recs) - 1
    assert ct.recommendations(board, open_orders, ctx, min_saving=1e9)[0].empty


def test_whats_new_and_weekly_on_time(world, snapshot):
    history = snapshot[0]
    news = ct.whats_new(world.orders, AS_OF, days=7)
    week = world.orders[(world.orders["order_date"] > AS_OF - pd.Timedelta(days=7)) & (world.orders["order_date"] <= AS_OF)]
    assert news["orders_placed"] == len(week) and 0 <= news["late_deliveries"] <= news["deliveries"]
    trend = ct.weekly_on_time(history, weeks=8)
    assert len(trend) == 8 and trend["on_time_rate"].between(0, 1).all() and trend["week"].is_monotonic_increasing


def test_autopilot_runs_the_whole_loop(world):
    store = ct.ActionStore(":memory:")
    empty = ct.results_to_date(store.ledger())
    assert empty["outcomes"] == 0 and empty["realized_saving"] == 0
    out = ct.run_autopilot(world, store, AS_OF, weeks=3, late_cost_pct=LATE_COST)
    assert out["as_of"] == AS_OF + pd.Timedelta(days=21) and store.get_meta("as_of") == str(out["as_of"].date())
    actions, ledger = store.actions(as_of=out["as_of"]), store.ledger()
    assert out["actions_created"] == len(actions) > 0
    assert (actions["decision"] == "approved").all() and (actions["status"] == "done").all() and actions["order_id"].is_unique
    done = ledger[ledger["final_cost"].notna()]
    assert out["outcomes_recorded"] == len(done) > 0
    # Only orders that have really arrived by the clock have an outcome, and each matches the simulated world.
    assert (done["delivery_date"] <= out["as_of"]).all()
    row = done.iloc[0]
    option = {k: row[k] for k in ("vendor", "carrier", "expedited", "split_vendor")}
    truth = world.realize_option(row["order_id"], option, LATE_COST)
    base = world.realize_option(row["order_id"], world.baseline_option(row["order_id"]), LATE_COST)
    assert row["final_cost"] == pytest.approx(truth["total_cost"]) and row["baseline_cost"] == pytest.approx(base["total_cost"])

    res = ct.results_to_date(ledger)
    assert res["outcomes"] == res["adopted"] == len(done)
    assert res["realized_saving"] == pytest.approx((done["baseline_cost"] - done["final_cost"]).sum())
    assert res["late_deliveries_avoided"] == int((done["baseline_late"] - done["late"]).sum())
    # Nothing new arrives if the clock does not move; a second pass records nothing twice.
    assert ct.record_simulated_outcomes(world, store, out["as_of"], LATE_COST) == 0

    board_counts = ct.pipeline_counts(pd.DataFrame(), actions, ledger)
    assert board_counts["outcomes_recorded"] == len(done) and board_counts["awaiting_outcome"] == len(actions) - len(done)
    assert board_counts["awaiting_decision"] == 0 and board_counts["needs_attention"] == 0


def test_rejected_actions_are_recorded_against_the_original_plan(world, snapshot):
    _, open_orders, _, ctx, board = snapshot
    recs, picks = ct.recommendations(board, open_orders, ctx, top_n=1)
    oid = recs.iloc[0]["order_id"]
    rec_row, keep_row = picks[oid]
    store = ct.ActionStore(":memory:")
    store.add_action(oid, rec_row, keep_row, "Dana", AS_OF, AS_OF, decision="rejected")
    assert ct.record_simulated_outcomes(world, store, AS_OF, LATE_COST) == 0  # not delivered yet
    assert ct.record_simulated_outcomes(world, store, AS_OF + pd.Timedelta(days=90), LATE_COST) == 1
    row = store.ledger().iloc[0]
    assert row["adopted"] == 0 and row["final_cost"] == pytest.approx(row["baseline_cost"]) and row["realized_saving"] == 0
    counts = ct.pipeline_counts(board, store.actions(as_of=AS_OF), store.ledger())
    urgent = board[board["priority"].isin(["Critical", "High"])]
    assert counts["needs_attention"] == len(urgent) - int(oid in set(urgent["order_id"]))


def test_reports_contain_the_headline_numbers(world, snapshot, replay):
    history, open_orders, _, ctx, board = snapshot
    recs, _ = ct.recommendations(board, open_orders, ctx, top_n=None)
    store = ct.ActionStore(":memory:")
    counts = ct.pipeline_counts(board, store.actions(as_of=AS_OF), store.ledger())
    headline = ct.briefing_headline(board, recs, counts)
    assert f"{len(board)} orders are open" in headline and f"{len(recs)} orders have a response" in headline
    html = ct.build_briefing(AS_OF, board, recs, store.actions(as_of=AS_OF), counts, ct.whats_new(world.orders, AS_OF),
                             ct.results_to_date(store.ledger()))
    assert headline in html and recs.iloc[0]["order_id"] in html and "No outcomes have been recorded yet" in html
    assert "There are no open orders today" in ct.briefing_headline(board.iloc[0:0], recs.iloc[0:0], counts)

    s = ct.summarize_replay(replay)
    perf = ct.build_performance_report(replay)
    assert f"${s['realized_saving']:,.0f}" in perf and f"{s['policy_on_time']:.1%}" in perf
    assert "only in a simulation" in perf


def test_cli_outputs(tmp_path, capsys):
    from controltower.cli import main

    db, brief, recs = tmp_path / "ct.db", tmp_path / "b.html", tmp_path / "r.csv"
    assert main(["briefing", "--db", str(db), "--date", "2025-06-30", "--out", str(brief), "--csv", str(recs)]) == 0
    assert brief.stat().st_size > 3000 and len(pd.read_csv(recs)) > 0
    assert main(["autopilot", "--weeks", "1", "--db", str(db)]) == 0
    assert "outcomes recorded" in capsys.readouterr().out
    perf, dec = tmp_path / "p.html", tmp_path / "d.csv"
    assert main(["replay", "--start", "2025-06-01", "--end", "2025-06-29", "--out", str(perf), "--csv", str(dec)]) == 0
    assert perf.exists() and {"est_saving", "realized_saving"} <= set(pd.read_csv(dec).columns)
    assert main(["replay", "--start", "2025-06-01", "--end", "2025-06-02"]) == 1
    assert "error:" in capsys.readouterr().err
