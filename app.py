"""SupplyLens Control Tower dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Let the app run straight from a checkout (for example on Streamlit Community
# Cloud) without the package being installed first.
_SRC = _Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import controltower as ct

st.set_page_config(page_title="SupplyLens Control Tower", layout="wide")

RED, AMBER, GREEN, BLUE, GREY = "#d1495b", "#e9a23b", "#2a9d8f", "#3a6ea5", "#9aa5b1"
PRIORITY_COLORS = {"Critical": RED, "High": AMBER, "Medium": BLUE, "Low": GREY}
DB_PATH = os.environ.get("CONTROL_TOWER_DB", str(Path(__file__).with_name("control_tower.db")))
DEFAULT_AS_OF = "2025-06-30"
PCT = st.column_config.NumberColumn(format="percent")
USD = st.column_config.NumberColumn(format="dollar")


# ------------------------------------------------------------------- caching
@st.cache_resource
def get_world() -> ct.SyntheticWorld:
    return ct.SyntheticWorld()


@st.cache_resource
def get_store() -> ct.ActionStore:
    return ct.ActionStore(DB_PATH)


@st.cache_resource(max_entries=6)
def get_snapshot(as_of: str, late_cost: float):
    """History, open orders, model and option context as known on one date."""
    history, open_orders = ct.split_as_of(get_world().orders, as_of)
    model = ct.fit_model(history)
    return history, open_orders, model, ct.OptionContext.build(model, history, late_cost_pct=late_cost)


@st.cache_data(max_entries=6, show_spinner=False)
def get_replay(start: str, end: str, late_cost: float, min_saving: float) -> pd.DataFrame:
    bar = st.progress(0.0, text="Replaying decisions week by week...")
    result = ct.run_policy_replay(get_world(), start, end, late_cost_pct=late_cost, min_saving=min_saving,
                                  progress=lambda f: bar.progress(f, text="Replaying decisions week by week..."))
    bar.empty()
    return result


def show(fig: go.Figure, height: int = 360, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return st.plotly_chart(fig, width="stretch", **kwargs)


def option_from_action(action: pd.Series) -> dict:
    return {"vendor": action["vendor"], "carrier": action["carrier"], "expedited": int(action["expedited"]),
            "split_vendor": action["split_vendor"]}


world, store = get_world(), get_store()
first_day = (world.orders["order_date"].min() + pd.Timedelta(days=365)).date()
last_day = world.orders["order_date"].max().date()

# ------------------------------------------------------------------- sidebar
st.sidebar.title("Control Tower")
st.sidebar.caption("Simulated supply chain. Not real company data.")
page = st.sidebar.radio("View", ["Briefing", "Monitor", "Actions", "Outcomes", "Performance", "How it works"])

st.sidebar.divider()
as_of = pd.Timestamp(store.get_meta("as_of", DEFAULT_AS_OF))
st.sidebar.markdown(f"**Today in the replay: {as_of:%a %d %b %Y}**")
b1, b2 = st.sidebar.columns(2)
target = as_of
if b1.button("Next day", width="stretch"):
    target = as_of + pd.Timedelta(days=1)
elif b2.button("Next week", width="stretch"):
    target = as_of + pd.Timedelta(days=7)
# The key includes the current date so the picker resets whenever the clock moves.
picked = st.sidebar.date_input("Jump to date", as_of.date(), min_value=first_day, max_value=last_day, key=f"jump_{as_of.date()}")
if target == as_of and pd.Timestamp(picked) != as_of:
    target = pd.Timestamp(picked)
target = min(max(target, pd.Timestamp(first_day)), pd.Timestamp(last_day))
if target != as_of:
    store.set_meta("as_of", str(target.date()))
    st.rerun()

st.sidebar.divider()
analyst = st.sidebar.text_input("Your name (default action owner)", "Analyst")
late_cost = st.sidebar.slider("Cost of a late delivery (share of goods value)", 0.02, 0.50, 0.10, 0.01)
stockout_cost = st.sidebar.slider("Extra cost if late causes a stockout", 0.0, 1.0, 0.25, 0.05)
risk_threshold = st.sidebar.slider("High-risk threshold (late probability)", 0.10, 0.80, 0.30, 0.05)

history, open_orders, model, ctx = get_snapshot(str(as_of.date()), late_cost)
board = ct.open_order_board(model, history, open_orders, as_of, on_hand=world.stock_snapshot(as_of, history),
                            late_cost_pct=late_cost, stockout_cost_pct=stockout_cost, risk_threshold=risk_threshold)
actions = store.actions(as_of=as_of)
ledger = store.ledger()
counts = ct.pipeline_counts(board, actions, ledger)


def order_row(order_id: str) -> pd.Series:
    return open_orders[open_orders["order_id"] == order_id].iloc[0]


def decide(order_id: str, option: pd.Series, keep: pd.Series, decision: str, note: str = "") -> None:
    """Create an action for an order with the decision already made."""
    aid = store.add_action(order_id, option, keep, analyst or "Analyst", as_of + pd.Timedelta(days=2), as_of, note, decision)
    if decision == "rejected":
        store.update_action(aid, status="done")


# The loop, with how much work sits at each step. The current view's step is highlighted.
STEPS = [
    ("Monitor", "1 Monitor", f"{counts['needs_attention']} need attention"),
    ("Monitor", "2 Decide", f"{counts['awaiting_decision']} awaiting decision"),
    ("Actions", "3 Track", f"{counts['in_progress']} in progress" + (f", {counts['overdue']} overdue" if counts["overdue"] else "")),
    ("Outcomes", "4 Record", f"{counts['awaiting_outcome']} awaiting outcome"),
    ("Performance", "5 Review", f"{counts['outcomes_recorded']} outcomes recorded"),
]
chips = "".join(
    f'<div style="flex:1;min-width:130px;padding:8px 12px;border-radius:8px;border:1px solid rgba(128,128,128,.35);'
    f'background:{"rgba(58,110,165,.18)" if view_name == page else "rgba(128,128,128,.07)"}">'
    f'<div style="font-size:12px;opacity:.75;font-weight:600">{label}</div><div style="font-size:14px">{text}</div></div>'
    for view_name, label, text in STEPS
)
st.markdown(f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">{chips}</div>', unsafe_allow_html=True)

# ================================================================== BRIEFING
if page == "Briefing":
    st.title(f"Daily briefing: {as_of:%A %d %B %Y}")
    taken = actions["order_id"] if len(actions) else ()
    recs, picks = ct.recommendations(board, open_orders, ctx, exclude=taken, top_n=None)
    results = ct.results_to_date(ledger)
    news = ct.whats_new(world.orders, as_of)
    st.info(ct.briefing_headline(board, recs, counts))

    k = st.columns(4)
    k[0].metric("Value at risk today", f"${board['value_at_risk'].sum():,.0f}" if len(board) else "$0")
    k[1].metric("Saving available today", f"${recs['expected_saving'].sum():,.0f}" if len(recs) else "$0",
                f"{len(recs)} order(s) to decide", delta_color="off", help="Estimated by the model")
    k[2].metric("Realized saving to date", f"${results['realized_saving']:,.0f}",
                f"${results['estimated_saving']:,.0f} was estimated", delta_color="off",
                help="Measured against the simulated baseline for outcomes recorded so far")
    k[3].metric("Late deliveries avoided to date", results["late_deliveries_avoided"],
                f"{results['outcomes']} outcome(s) recorded", delta_color="off")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Decide these first")
        if recs.empty:
            st.success("Nothing to decide: no response beats the current plan for any open order without an action.")
        if len(recs):
            st.caption("Largest estimated saving first.")
        for _, r in recs.sort_values("expected_saving", ascending=False).head(5).iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{r['order_id']}** · {r['priority']} priority · {r['quantity']:,.0f} × {r['sku']} from {r['vendor']} "
                            f"(${r['goods_value']:,.0f})")
                c1.markdown(f"**{r['recommendation']}**: saves an estimated **${r['expected_saving']:,.0f}**, late risk "
                            f"{r['risk_before']:.0%} → {r['risk_after']:.0%}. Main risk driver: {r['main_reason']}.")
                rec_row, keep_row = picks[r["order_id"]]
                if c2.button("Approve", key=f"approve_{r['order_id']}", type="primary", width="stretch"):
                    decide(r["order_id"], rec_row, keep_row, "approved", "approved from briefing")
                    st.rerun()
                if c2.button("Keep plan", key=f"keep_{r['order_id']}", width="stretch"):
                    decide(r["order_id"], rec_row, keep_row, "rejected", "kept original plan")
                    st.rerun()
        if len(recs) > 5:
            more = len(recs) - 5
            c1, c2 = st.columns([3, 2])
            c1.caption(f"{more} more order(s) have a recommended response. Open the Monitor view to review them one by one.")
            if c2.button(f"Approve all {len(recs)} recommendations", width="stretch"):
                for oid, (rec_row, keep_row) in picks.items():
                    decide(oid, rec_row, keep_row, "approved", "approved in bulk from briefing")
                st.rerun()
    with right:
        st.subheader("Since yesterday")
        a, b = st.columns(2)
        a.metric("Orders placed", news["orders_placed"])
        b.metric("Deliveries", news["deliveries"], f"{news['late_deliveries']} late", delta_color="off")
        trend = ct.weekly_on_time(history)
        fig = px.line(trend, x="week", y="on_time_rate", markers=True, labels={"week": "", "on_time_rate": "On-time rate, last 12 weeks"})
        fig.update_traces(line_color=BLUE)
        fig.update_yaxes(tickformat=".0%")
        show(fig, 220)
        due = actions[actions["status"].isin(["open", "in progress"]) & (actions["due_date"] <= as_of)] if len(actions) else actions
        st.markdown(f"**Actions due or overdue: {len(due)}**")
        if len(due):
            st.dataframe(due[["order_id", "option", "owner", "due_date"]], hide_index=True,
                         column_config={"due_date": st.column_config.DateColumn("Due", format="DD MMM")})

    st.subheader("Results so far")
    done = ledger[ledger["final_cost"].notna()] if len(ledger) else ledger
    if done.empty:
        st.caption("No outcomes yet. Approve some recommendations and move time forward, or let the autopilot run a few weeks below.")
    else:
        daily = done.sort_values("delivery_date").assign(
            estimated=lambda d: d["est_saving"].where(d["adopted"] == 1, 0.0).cumsum(),
            realized=lambda d: d["realized_saving"].fillna(0.0).cumsum(),
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=daily["delivery_date"], y=daily["estimated"], name="Estimated (model)", line=dict(color=GREY, dash="dash"))
            fig.add_scatter(x=daily["delivery_date"], y=daily["realized"], name="Realized (simulation)", line_color=GREEN)
            fig.update_layout(yaxis_title="Cumulative saving ($)", legend=dict(orientation="h", y=-0.25))
            show(fig, 300)
        with c2:
            st.metric("On time when adopted", "n/a" if np.isnan(results["on_time_adopted"]) else f"{results['on_time_adopted']:.0%}",
                      None if np.isnan(results["on_time_baseline"]) else f"baseline {results['on_time_baseline']:.0%}", delta_color="off")
            st.metric("Observed cost of recorded orders", f"${results['observed_cost']:,.0f}")
            ratio = results["realized_saving"] / results["estimated_saving"] if results["estimated_saving"] else float("nan")
            st.metric("Realized per $1 estimated", "n/a" if np.isnan(ratio) else f"${ratio:,.2f}")
        st.caption("Realized saving compares each order with its simulated baseline; it can be measured only because the data is simulated.")

    st.subheader("Fast-forward")
    c1, c2 = st.columns([2, 1])
    weeks = c1.slider("Let the autopilot follow every recommendation for this many weeks", 1, 12, 4)
    c2.write("")
    if c2.button(f"Run {weeks} week(s)", type="primary", width="stretch"):
        bar = st.progress(0.0, text="Deciding, moving time forward and recording outcomes...")
        out = ct.run_autopilot(world, store, as_of, weeks, owner="Autopilot", late_cost_pct=late_cost,
                               progress=lambda f: bar.progress(f, text="Deciding, moving time forward and recording outcomes..."))
        st.toast(f"{out['actions_created']} actions approved, {out['outcomes_recorded']} outcomes recorded")
        st.rerun()
    st.caption("Each week the autopilot approves the recommended response for new orders, advances the clock and records what "
               "happened. Use it to see results quickly; then review them under Performance.")

    st.subheader("Take it with you")
    d = st.columns(3)
    d[0].download_button("Today's briefing (HTML)", ct.build_briefing(as_of, board, recs, actions, counts, news, results),
                         file_name=f"briefing_{as_of:%Y-%m-%d}.html", mime="text/html", width="stretch")
    d[1].download_button("Recommended responses (CSV)", recs.to_csv(index=False), file_name=f"recommendations_{as_of:%Y-%m-%d}.csv",
                         mime="text/csv", width="stretch", disabled=recs.empty)
    d[2].download_button("Action list (CSV)", actions.to_csv(index=False), file_name="actions.csv", mime="text/csv",
                         width="stretch", disabled=actions.empty)

# =================================================================== MONITOR
elif page == "Monitor":
    st.title("Open orders at risk")
    active = actions[actions["status"].isin(["open", "in progress"])] if len(actions) else actions
    k = st.columns(5)
    k[0].metric("Open orders", f"{len(board)}")
    k[1].metric("Value at risk", f"${board['value_at_risk'].sum():,.0f}")
    k[2].metric("Critical or high", f"{board['priority'].isin(['Critical', 'High']).sum()}")
    k[3].metric("Active actions", f"{len(active)}")
    k[4].metric("Overdue actions", f"{int(active['overdue'].sum()) if len(active) else 0}")

    f1, f2, f3 = st.columns([2, 2, 1])
    pri = f1.multiselect("Priority", ct.PRIORITY_ORDER, default=ct.PRIORITY_ORDER)
    stg = f2.multiselect("Stage", ["pending release", "in transit", "overdue"], default=["pending release", "in transit", "overdue"])
    hide_done = f3.checkbox("Hide orders with an action", value=False)
    view = board[board["priority"].isin(pri) & board["stage"].isin(stg)]
    has_action = view["order_id"].isin(actions["order_id"]) if len(actions) else pd.Series(False, index=view.index)
    if hide_done:
        view = view[~has_action]
    view = view.assign(action=np.where(view["order_id"].isin(actions["order_id"]) if len(actions) else False, "yes", ""))

    if view.empty:
        st.info("No open orders match the filters.")
        st.stop()

    fig = px.scatter(view, x="late_probability", y="stockout_probability", size="goods_value", color="priority",
                     hover_name="order_id", hover_data={"vendor": True, "sku": True, "stage": True, "value_at_risk": ":,.0f"},
                     color_discrete_map=PRIORITY_COLORS, category_orders={"priority": ct.PRIORITY_ORDER},
                     labels={"late_probability": "Late probability", "stockout_probability": "Stockout probability if late"})
    fig.add_vline(x=risk_threshold, line_dash="dash", line_color=GREY)
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    show(fig, 320)
    st.caption("Each bubble is an open order, sized by goods value. Top-right orders are both likely to be late and "
               "likely to cause a stockout if they are.")

    st.subheader("Priority queue")
    st.caption("Sorted by value at risk. Click a row to compare response options and assign an action.")
    cols = ["order_id", "priority", "stage", "action", "days_to_due", "vendor", "carrier", "sku", "quantity", "goods_value",
            "late_probability", "stockout_probability", "value_at_risk"]
    event = st.dataframe(
        view[cols], hide_index=True, on_select="rerun", selection_mode="single-row", key="queue",
        column_config={
            "late_probability": st.column_config.ProgressColumn("Late prob.", format="percent", min_value=0, max_value=1),
            "stockout_probability": st.column_config.ProgressColumn("Stockout prob.", format="percent", min_value=0, max_value=1),
            "goods_value": USD, "value_at_risk": USD, "days_to_due": st.column_config.NumberColumn("Days to due"),
        },
    )
    rows = event.selection.rows
    chosen = view.iloc[rows[0] if rows else 0]
    order = order_row(chosen["order_id"])

    st.divider()
    st.subheader(f"{chosen['order_id']}: {chosen['quantity']:,.0f} units of {chosen['sku']} from {chosen['vendor']}")
    st.markdown(
        f"**{chosen['priority']}** priority, {chosen['stage']}. Promised {chosen['promised_date']:%d %b} "
        f"({chosen['days_to_due']:+d} days). Late probability **{chosen['late_probability']:.0%}**; stock on hand "
        f"{chosen['on_hand']:,.0f} units, stockout probability if late **{chosen['stockout_probability']:.0%}**."
    )
    opts = ctx.options(order, chosen["stage"])
    rec = ct.recommend(opts)
    keep = opts[opts["kind"] == "keep"].iloc[0]

    left, right = st.columns(2)
    with left:
        st.markdown("**Why this order is at risk**")
        contrib = model.contributions(open_orders[open_orders["order_id"] == chosen["order_id"]]).iloc[0].sort_values()
        fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h",
                               marker_color=[RED if v > 0 else GREEN for v in contrib.values],
                               customdata=np.exp(contrib.values), hovertemplate="%{y}: odds x%{customdata:.2f}<extra></extra>"))
        fig.update_layout(xaxis_title="Effect on late risk (log-odds vs. an average order)")
        show(fig, 300)
        if chosen["stage"] == "overdue":
            st.caption("This order has already missed its promised date; the bars show what made it risky when it was placed.")
    with right:
        st.markdown("**Response options**")
        plot = opts.assign(marker=np.where(opts["option"] == rec["option"], "recommended",
                                           np.where(opts["kind"] == "keep", "current plan", "other option")))
        fig = px.scatter(plot, x="late_probability", y="expected_cost", color="marker", hover_name="option", symbol="kind",
                         color_discrete_map={"recommended": GREEN, "current plan": RED, "other option": GREY},
                         labels={"late_probability": "Late probability", "expected_cost": "Expected cost ($)"})
        fig.update_traces(marker_size=12)
        fig.update_xaxes(tickformat=".0%")
        show(fig, 300)
        st.caption("Bottom-left is cheaper and safer. Supplier, carrier and split changes are only possible while an "
                   "order is pending release.")

    if rec["kind"] == "keep":
        st.info("Recommendation: **keep the order as planned**. No option is both cheaper in expected cost and no riskier.")
    else:
        st.success(f"Recommendation: **{rec['option']}**. Expected saving ${rec['saving_vs_keep']:,.0f}; late probability "
                   f"{keep['late_probability']:.0%} to {rec['late_probability']:.0%}.")
    st.dataframe(
        opts.drop(columns=["vendor", "carrier", "expedited", "split_vendor"]), hide_index=True,
        column_config={"late_probability": PCT, "risk_change": PCT, "goods_cost": USD, "freight_cost": USD,
                       "expected_late_cost": USD, "expected_cost": USD, "saving_vs_keep": USD},
    )

    existing = actions[actions["order_id"] == chosen["order_id"]] if len(actions) else actions
    if len(existing):
        e = existing.iloc[-1]
        st.warning(f"An action already exists for this order: **{e['option']}**, owner {e['owner']}, due "
                   f"{e['due_date']:%d %b}, decision {e['decision']}, status {e['status']}.")
    elif rec["kind"] != "keep":
        q1, q2, _ = st.columns([1, 1, 2])
        if q1.button("Approve recommendation", type="primary", width="stretch"):
            decide(chosen["order_id"], rec, keep, "approved", "approved from monitor")
            st.rerun()
        if q2.button("Keep the plan", width="stretch"):
            decide(chosen["order_id"], rec, keep, "rejected", "kept original plan")
            st.rerun()
    with st.form("assign"):
        st.markdown("**Or assign it to someone**")
        c = st.columns([3, 2, 2])
        names = opts["option"].tolist()
        option_name = c[0].selectbox("Response", names, index=names.index(rec["option"]))
        owner = c[1].text_input("Owner", analyst)
        due = c[2].date_input("Due date", (as_of + pd.Timedelta(days=2)).date(), min_value=as_of.date())
        note = st.text_input("Note", "")
        if st.form_submit_button("Create action", type="primary"):
            try:
                store.add_action(chosen["order_id"], opts[opts["option"] == option_name].iloc[0], keep, owner, due, as_of, note)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()

    with st.expander("Create actions in bulk"):
        pending = board[(board["stage"] == "pending release") & ~board["order_id"].isin(actions["order_id"] if len(actions) else [])]
        st.write(f"{len(pending)} order(s) pending release have no action yet. This creates an action for each one where "
                 "a response is recommended over the plan, owned by you and due in two days.")
        if st.button("Create recommended actions", disabled=pending.empty):
            made = 0
            for oid in pending["order_id"]:
                o = ctx.options(order_row(oid), "pending release")
                r = ct.recommend(o)
                if r["kind"] != "keep":
                    store.add_action(oid, r, o[o["kind"] == "keep"].iloc[0], analyst or "Analyst",
                                     as_of + pd.Timedelta(days=2), as_of, "created in bulk")
                    made += 1
            st.toast(f"Created {made} action(s)")
            st.rerun()

# =================================================================== ACTIONS
elif page == "Actions":
    st.title("Actions")
    if actions.empty:
        st.info("No actions yet. Open the Monitor view, pick an order and assign a response.")
        st.stop()
    k = st.columns(5)
    k[0].metric("All actions", len(actions))
    k[1].metric("Awaiting decision", int((actions["decision"] == "pending").sum()))
    k[2].metric("Approved", int((actions["decision"] == "approved").sum()))
    k[3].metric("Overdue", int(actions["overdue"].sum()))
    k[4].metric("Estimated saving (approved)", f"${actions.loc[actions['decision'] == 'approved', 'est_saving'].sum():,.0f}")

    f1, f2, f3 = st.columns(3)
    owners = f1.multiselect("Owner", sorted(actions["owner"].unique()), placeholder="All owners")
    statuses = f2.multiselect("Status", ct.STATUSES, default=["open", "in progress"])
    only_overdue = f3.checkbox("Overdue only")
    view = actions[actions["status"].isin(statuses)] if statuses else actions
    if owners:
        view = view[view["owner"].isin(owners)]
    if only_overdue:
        view = view[view["overdue"]]

    by_status = actions.groupby(["status", "decision"]).size().reset_index(name="actions")
    left, right = st.columns([1, 2])
    with left:
        show(px.bar(by_status, x="status", y="actions", color="decision",
                    color_discrete_map={"pending": GREY, "approved": GREEN, "rejected": RED},
                    category_orders={"status": ct.STATUSES}, labels={"status": "", "actions": "Actions"}), 300)
    with right:
        st.caption("Edit owner, due date, decision, status or note directly in the table, then save.")
        cols = ["action_id", "order_id", "option", "owner", "due_date", "decision", "status", "overdue", "est_saving",
                "est_keep_risk", "est_option_risk", "note"]
        edited = st.data_editor(
            view[cols], hide_index=True, key="actions_editor",
            disabled=["action_id", "order_id", "option", "overdue", "est_saving", "est_keep_risk", "est_option_risk"],
            column_config={
                "decision": st.column_config.SelectboxColumn(options=ct.DECISIONS, required=True),
                "status": st.column_config.SelectboxColumn(options=ct.STATUSES, required=True),
                "due_date": st.column_config.DateColumn("Due", format="DD MMM YYYY"),
                "est_saving": USD, "est_keep_risk": st.column_config.NumberColumn("Risk before", format="percent"),
                "est_option_risk": st.column_config.NumberColumn("Risk after", format="percent"),
            },
        )
        if st.button("Save changes", type="primary"):
            saved = 0
            for (_, old), (_, new) in zip(view[cols].iterrows(), edited.iterrows()):
                changes = {f: new[f] for f in ["owner", "due_date", "decision", "status", "note"]
                           if str(new[f]) != str(old[f]) and not (pd.isna(new[f]) and pd.isna(old[f]))}
                if changes:
                    store.update_action(int(old["action_id"]), **changes)
                    saved += 1
            st.toast(f"Saved {saved} action(s)")
            st.rerun()

    st.download_button("Download action list (CSV)", actions.to_csv(index=False), file_name="actions.csv", mime="text/csv")
    with st.expander("Delete all actions and outcomes"):
        if st.checkbox("I want to permanently delete every action and outcome") and st.button("Delete everything"):
            store.reset()
            st.rerun()

# ================================================================== OUTCOMES
elif page == "Outcomes":
    st.title("Outcomes")
    ledger = store.ledger()
    if ledger.empty:
        st.info("No actions yet, so there are no outcomes to record.")
        st.stop()
    waiting = ledger[ledger["final_cost"].isna() & (ledger["decision"] != "pending")].copy()
    undecided = int((ledger["final_cost"].isna() & (ledger["decision"] == "pending")).sum())

    # In the simulation the result of each decided action can be looked up once the order has arrived.
    ready = []
    for _, a in waiting.iterrows():
        adopted = a["decision"] == "approved"
        followed = option_from_action(a) if adopted else world.baseline_option(a["order_id"])
        result = world.realize_option(a["order_id"], followed, late_cost)
        if result["delivery_date"] <= as_of:
            base = world.realize_option(a["order_id"], world.baseline_option(a["order_id"]), late_cost)
            ready.append((a, adopted, result, base))

    k = st.columns(4)
    k[0].metric("Outcomes recorded", int(ledger["final_cost"].notna().sum()))
    k[1].metric("Delivered, ready to record", len(ready))
    k[2].metric("Decided, still in transit", len(waiting) - len(ready))
    k[3].metric("Awaiting a decision", undecided)
    if undecided:
        st.caption("Actions still marked 'pending' need an approve or reject decision on the Actions view before an outcome can be recorded.")

    st.subheader("Record from the simulation")
    st.write("For orders that have arrived by today's replay date, this records the delivery date, final cost and "
             "lateness that the simulated world produced for the decision that was followed. Because the world is "
             "simulated, it also records what the original plan would have produced.")
    if st.button(f"Record {len(ready)} outcome(s)", type="primary", disabled=not ready):
        for a, adopted, result, base in ready:
            store.record_outcome(int(a["action_id"]), adopted, result["delivery_date"], result["late"], result["total_cost"],
                                 as_of, baseline_cost=base["total_cost"], baseline_late=base["late"], source="simulation")
        st.rerun()

    with st.expander("Enter an outcome by hand"):
        st.caption("Use this as you would with real orders: you know what happened, not what would have happened otherwise, "
                   "so no baseline is stored.")
        open_ids = ledger.loc[ledger["final_cost"].isna(), "action_id"].tolist()
        if not open_ids:
            st.write("Every action already has an outcome.")
        else:
            with st.form("manual_outcome"):
                c = st.columns(4)
                aid = c[0].selectbox("Action", open_ids, format_func=lambda i: f"#{i} {ledger.set_index('action_id').loc[i, 'order_id']}")
                adopted = c[1].checkbox("Recommendation adopted", value=True)
                delivered = c[2].date_input("Delivery date", as_of.date(), max_value=as_of.date())
                late = c[3].checkbox("Arrived late")
                final_cost = st.number_input("Final cost ($, goods + freight + cost of lateness)", min_value=0.0, step=100.0)
                if st.form_submit_button("Record outcome"):
                    store.record_outcome(int(aid), adopted, delivered, late, final_cost, as_of, source="entered")
                    st.rerun()

    done = ledger[ledger["final_cost"].notna()]
    st.subheader("Recorded outcomes")
    if done.empty:
        st.info("Nothing recorded yet. Approve or reject some actions, move the replay date forward until the orders arrive, then record.")
    else:
        st.dataframe(
            done[["action_id", "order_id", "option", "owner", "adopted", "delivery_date", "late", "est_option_cost",
                  "final_cost", "est_error", "baseline_cost", "realized_saving", "source"]],
            hide_index=True,
            column_config={"adopted": st.column_config.CheckboxColumn(), "late": st.column_config.CheckboxColumn(),
                           "est_option_cost": st.column_config.NumberColumn("Estimated cost", format="dollar"),
                           "final_cost": USD, "est_error": st.column_config.NumberColumn("Final minus estimate", format="dollar"),
                           "baseline_cost": USD, "realized_saving": USD,
                           "delivery_date": st.column_config.DateColumn("Delivered", format="DD MMM YYYY")},
        )
        st.download_button("Download outcomes.csv", done.to_csv(index=False), file_name="outcomes.csv", mime="text/csv")

# =============================================================== PERFORMANCE
elif page == "Performance":
    st.title("Performance against the baseline")
    st.caption("Baseline: keep the original supplier and carrier for every order. Estimated savings (from the model, at "
               "decision time) are reported separately from costs that were actually observed.")
    tab_policy, tab_actions = st.tabs(["Policy replay", "Your recorded actions"])

    with tab_policy:
        st.write("Replays a period of history. Each week the model is retrained on what had been delivered by then, the "
                 "recommendation is followed for every new order, and both the decision and the baseline are played out.")
        c = st.columns(3)
        end_default = min(as_of, pd.Timestamp(last_day)).date()
        start = c[0].date_input("From", (pd.Timestamp(end_default) - pd.Timedelta(weeks=20)).date(), min_value=first_day, max_value=last_day)
        end = c[1].date_input("To", end_default, min_value=first_day, max_value=last_day)
        min_saving = c[2].number_input("Only act if expected saving exceeds ($)", 0, 5000, 0, 25)
        if pd.Timestamp(end) - pd.Timestamp(start) < pd.Timedelta(days=14):
            st.warning("Choose a period of at least two weeks.")
            st.stop()
        if st.button("Run replay", type="primary"):
            st.session_state["replay_key"] = (str(start), str(end), late_cost, float(min_saving))
        key = st.session_state.get("replay_key")
        if key is None:
            st.info("Run the replay to compare the policy with the baseline. A 20-week period takes about 20 seconds.")
        else:
            rp = get_replay(*key)
            if rp.empty:
                st.warning("No orders were decided in this period.")
                st.stop()
            s = ct.summarize_replay(rp)
            st.caption(f"Showing the replay of {key[0]} to {key[1]}, late cost {key[2]:.0%}, minimum saving ${key[3]:,.0f}.")
            d1, d2, _ = st.columns([1, 1, 2])
            d1.download_button("Performance report (HTML)", ct.build_performance_report(rp),
                               file_name=f"performance_{key[0]}_{key[1]}.html", mime="text/html", width="stretch")
            d2.download_button("Every decision (CSV)", rp.to_csv(index=False), file_name="replay_decisions.csv",
                               mime="text/csv", width="stretch")
            m = st.columns(4)
            m[0].metric("Orders decided", f"{s['orders']:,}", f"{s['share_changed']:.0%} changed", delta_color="off")
            m[1].metric("Estimated saving", f"${s['estimated_saving']:,.0f}", help="What the model expected at decision time")
            m[2].metric("Realized saving", f"${s['realized_saving']:,.0f}", f"{s['realized_saving_pct']:.2%} of baseline cost",
                        help="Baseline cost minus policy cost, as played out in the simulation")
            m[3].metric("On-time delivery", f"{s['policy_on_time']:.1%}",
                        f"{(s['policy_on_time'] - s['baseline_on_time']) * 100:+.1f} pts vs baseline {s['baseline_on_time']:.1%}")
            ratio = s["estimate_ratio"]
            st.markdown(
                f"For every $1 of saving the model estimated, **${ratio:,.2f}** was realized. "
                + ("The model over-promised; treat its estimates as optimistic." if ratio < 0.9 else
                   "The estimates held up in this period." if ratio <= 1.1 else "The model under-estimated the benefit in this period.")
            )

            weekly = rp.groupby("decided_on")[["est_saving", "realized_saving"]].sum().cumsum().reset_index()
            left, right = st.columns(2)
            with left:
                fig = go.Figure()
                fig.add_scatter(x=weekly["decided_on"], y=weekly["est_saving"], name="Estimated (model)", line=dict(color=GREY, dash="dash"))
                fig.add_scatter(x=weekly["decided_on"], y=weekly["realized_saving"], name="Realized (simulation)", line_color=GREEN)
                fig.update_layout(yaxis_title="Cumulative saving ($)", legend=dict(orientation="h", y=-0.2))
                show(fig, 340)
            with right:
                by = ct.replay_by_option(rp)
                by = by[by["kind"] != "keep"].melt(id_vars="kind", value_vars=["estimated_saving", "realized_saving"])
                show(px.bar(by, x="kind", y="value", color="variable", barmode="group",
                            color_discrete_map={"estimated_saving": GREY, "realized_saving": GREEN},
                            labels={"kind": "Type of change", "value": "Saving ($)"}), 340)

            changed = rp[rp["changed"]]
            fig = px.scatter(changed, x="est_saving", y="realized_saving", color="kind", hover_name="order_id", opacity=0.7,
                             labels={"est_saving": "Estimated saving ($)", "realized_saving": "Realized saving ($)"})
            lim = float(max(changed["est_saving"].max(), 1))
            fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=GREY, dash="dash"), name="estimate = realized")
            show(fig, 380)
            st.caption("Each point is one changed order. Individual orders scatter widely around the estimate because a single "
                       "order is either late or not; the estimate is only meaningful across many decisions.")
            st.dataframe(ct.replay_by_option(rp), hide_index=True,
                         column_config={"estimated_saving": USD, "realized_saving": USD, "est_risk_before": PCT,
                                        "est_risk_after": PCT, "late_share_baseline": PCT, "late_share_policy": PCT})
            st.warning("The realized saving can be measured here only because the supply chain is simulated: the same order "
                       "is played out twice, once per decision. With real orders the baseline outcome of a changed order is "
                       "never observed, so proving savings needs a controlled comparison, such as holding back a random "
                       "share of orders on the original plan.")

    with tab_actions:
        ledger = store.ledger()
        done = ledger[ledger["final_cost"].notna()] if len(ledger) else ledger
        if done.empty:
            st.info("No outcomes recorded yet. Record some on the Outcomes view.")
        else:
            r = ct.review_actions(ledger)
            m = st.columns(4)
            m[0].metric("Outcomes recorded", r["with_outcome"], f"{r['adopted']} adopted", delta_color="off")
            m[1].metric("Estimated saving (adopted)", f"${r['estimated_saving_adopted']:,.0f}", help="From the model, at decision time")
            m[2].metric("Observed cost", f"${r['observed_cost']:,.0f}", help="Sum of final costs actually recorded")
            m[3].metric("Realized saving", f"${r['realized_saving_known']:,.0f}",
                        f"known for {r['outcomes_with_baseline']} of {r['with_outcome']}", delta_color="off",
                        help="Only where the baseline outcome was recorded (simulation)")
            c = st.columns(3)
            c[0].metric("On time, adopted", "n/a" if np.isnan(r["on_time_adopted"]) else f"{r['on_time_adopted']:.0%}")
            c[1].metric("On time, not adopted", "n/a" if np.isnan(r["on_time_not_adopted"]) else f"{r['on_time_not_adopted']:.0%}")
            c[2].metric("Final cost minus estimate (mean)", f"{'-' if r['mean_estimate_error'] < 0 else '+'}${abs(r['mean_estimate_error']):,.0f}")
            st.caption("Comparing adopted with not-adopted actions describes what happened; it does not prove the decisions "
                       "caused the difference, because the two groups of orders can differ in other ways.")
            known = done[done["baseline_cost"].notna()]
            if len(known):
                fig = px.scatter(known, x="est_saving", y="realized_saving", color=known["adopted"].map({1: "adopted", 0: "not adopted"}),
                                 hover_name="order_id", color_discrete_map={"adopted": GREEN, "not adopted": GREY},
                                 labels={"est_saving": "Estimated saving ($)", "realized_saving": "Realized saving ($)", "color": ""})
                show(fig, 340)
            owners = done.groupby("owner").agg(actions=("action_id", "size"), adopted=("adopted", "sum"),
                                               on_time=("late", lambda s: 1 - s.mean()), observed_cost=("final_cost", "sum"),
                                               estimated_saving=("est_saving", "sum")).reset_index()
            st.dataframe(owners, hide_index=True, column_config={"on_time": PCT, "observed_cost": USD, "estimated_saving": USD})

# ============================================================== HOW IT WORKS
else:
    st.title("How it works")
    st.markdown(
        f"""
**The loop.** Monitor open orders → compare response options → assign an action → record the outcome →
review against the baseline.

**Replay.** The dashboard sits at one date in a simulated history (today in the replay: **{as_of:%d %b %Y}**).
It knows only orders already delivered by that date. Use *Next day* or *Next week* in the sidebar to move forward:
new orders appear, actions fall due, and delivered orders become ready for an outcome.

**Risk.** A late-delivery model (logistic regression, from the SupplyLens package) is trained on delivered orders
only. Stockout probability is the chance the item runs out before the order arrives if it is late by the typical
delay, given stock on hand and earlier inbound orders. Orders are ranked by
`goods value x late probability x (late cost + stockout cost x stockout probability)`.

**Options.** Keep as planned, expedite, change carrier, change supplier, or split 50/50 with another supplier.
Supplier, carrier and split changes are only offered while an order is pending release (placed within the last
three days). Expected cost = goods + freight + late probability x cost of a late delivery. The recommendation is
the cheapest option that is no more likely to be late than the plan.

**What is estimated and what is observed.**
- *Estimated saving* comes from the model at decision time.
- *Observed cost* is the final cost recorded once the order arrives.
- *Realized saving* compares the decision with the baseline outcome of the same order. That outcome exists only
  because the world is simulated. With real orders it is never observed, and claiming savings would need a
  controlled comparison.

**Limits.** The data is simulated and is not any real company's. Each decision is evaluated on its own: changing
one order does not affect later orders or supplier capacity. The cost of a late delivery and of a stockout are
assumptions you set in the sidebar, and the recommendations depend on them.
"""
    )
    with st.expander("Model check on the current date"):
        st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in model.metrics.items()})
        st.dataframe(model.drivers().head(12), hide_index=True)
