# SupplyLens Control Tower

**A decision workflow for supply chain analysts: see which open orders are at risk, compare what can be
done about each, assign the action, record what happened, and check whether it worked.**

**[Try the live dashboard](https://supplylens-control-tower.streamlit.app)** — runs in your browser on simulated data, nothing to install.

Built on the [SupplyLens](https://github.com/MDABDURRAUF/supplylens) analytics package. SupplyLens
predicts and explains disruptions; the Control Tower turns those predictions into tracked decisions and
measures the result against a baseline.

## What an analyst can do

| Step | View | What it does |
|---|---|---|
| Start here | **Briefing** | One paragraph on what matters today, the decisions worth the most with one-click Approve / Keep plan, what changed since yesterday, and results so far |
| 1. Monitor | **Monitor** | Ranks every open order by money at risk, combining the probability of late delivery with the probability that a late delivery causes a stockout |
| 2. Compare | **Monitor → order detail** | Prices each response (keep as planned, expedite, change carrier, change supplier, split the order) with its expected cost and late probability, and recommends one |
| 3. Assign | **Actions** | Records the chosen response with an owner, due date, decision (pending / approved / rejected) and status; flags overdue actions |
| 4. Record | **Outcomes** | Stores the actual delivery date, final cost, lateness, and whether the recommendation was adopted |
| 5. Review | **Performance** | Compares a decision policy with the baseline of keeping the original supplier and carrier, reporting estimated and realized savings separately |

A strip at the top of every view shows how much work sits at each step of the loop (need attention, awaiting
decision, in progress, awaiting outcome, outcomes recorded).

## Outputs

The Control Tower is built to produce things you can hand to someone:

| Output | Where | Format |
|---|---|---|
| Daily briefing | Briefing view, or `controltower briefing` | HTML page, printable |
| Recommended responses | Briefing view, or `controltower briefing --csv` | CSV |
| Action list with owners, due dates and status | Briefing and Actions views | CSV |
| Recorded outcomes | Outcomes view | CSV |
| Performance report against the baseline | Performance view, or `controltower replay` | HTML page, printable |
| Every replayed decision | Performance view, or `controltower replay --csv` | CSV |

```bash
controltower briefing  --date 2025-06-30 --out briefing.html --csv recommendations.csv
controltower autopilot --weeks 4          # follow every recommendation for four weeks, record outcomes
controltower replay    --start 2025-01-01 --end 2025-06-30 --out performance.html --csv decisions.csv
```

**Autopilot.** To see results quickly, the Briefing view (or `controltower autopilot`) can follow every
recommendation for a number of weeks: each week it approves the recommended response for new orders, moves
the clock forward, and records what happened.

## Estimated, observed and realized: three different numbers

The dashboard never mixes these:

- **Estimated saving** is what the model expected when the decision was made.
- **Observed cost** is the final cost recorded after the order arrived.
- **Realized saving** is the baseline outcome minus the actual outcome *for the same order*.

The third can only be measured here because the supply chain is simulated: each order is played out
twice with the same random luck, once under the decision and once under the baseline. With real
orders the baseline outcome of a changed order is never observed. Showing that decisions caused
savings in a real operation would need a controlled comparison, for example holding back a random
share of orders on the original plan.

## Example result

On the default simulated world (seed 7), replaying January to June 2025 week by week:

| | Baseline | Policy |
|---|---|---|
| Orders decided | 913 | 913 (61% changed) |
| On-time delivery (by value) | 83.5% | 86.8% |
| Total cost | $2,534,259 | $2,516,623 |

The model estimated a saving of $23,203; $17,636 was realized, 0.7% of baseline cost. For every $1 of
estimated saving, $0.76 was realized, so the model's estimates were optimistic. These figures describe
a simulation, not a real supply chain.

## Quick start

```bash
git clone https://github.com/MDABDURRAUF/supplylens-control-tower.git
cd supplylens-control-tower
pip install -e ".[app,dev]"
streamlit run app.py
```

The dashboard opens at a date inside the simulated history. Use **Next day** and **Next week** in the
sidebar to move time forward: new orders appear, actions fall due, and delivered orders become ready
for an outcome. Actions and outcomes are saved in `control_tower.db` (SQLite) next to the app; set the
`CONTROL_TOWER_DB` environment variable to keep it elsewhere.

A typical session:

1. On **Briefing**, approve or keep the plan for the top decisions; or on **Monitor**, click an order, read why it is at risk, and assign an action.
2. On **Actions**, approve or reject each action and set its status.
3. Click **Next week** a few times.
4. On **Outcomes**, record the results of orders that have arrived.
5. On **Performance**, run the policy replay and review your own recorded actions.

From Python:

```python
import controltower as ct

world = ct.SyntheticWorld()
history, open_orders = ct.split_as_of(world.orders, "2025-06-30")
model = ct.fit_model(history)

board = ct.open_order_board(model, history, open_orders, "2025-06-30",
                            on_hand=world.stock_snapshot("2025-06-30", history))

ctx = ct.OptionContext.build(model, history, late_cost_pct=0.10)
options = ctx.options(open_orders.iloc[0], stage="pending release")
print(ct.recommend(options))

replay = ct.run_policy_replay(world, "2025-01-01", "2025-06-30")
print(ct.summarize_replay(replay))
```

## How it works

**Replay without leakage.** At any date, the tool splits orders into *history* (already delivered) and
*open* (placed, not yet arrived). Delivery dates of open orders are removed before anything else
runs, and the model is trained on history only.

**Risk and priority.** Late probability comes from the SupplyLens logistic regression, which also
learns the effect of expediting from past expedited shipments. Stockout probability is the chance
the item runs out before the order arrives if it is late by the typical delay, given stock on hand and
earlier inbound orders. Orders are ranked by

    goods value × late probability × (late cost + stockout cost × stockout probability)

**Options.** Supplier, carrier and split changes are offered only while an order is *pending release*
(placed within the last three days); afterwards only expediting remains. Alternative suppliers are
priced from their past prices for the same item, freight from a per-lane fit of charge against
quantity, and the expedite premium from past expedited shipments.

    expected cost = goods + freight + late probability × cost of a late delivery

The recommendation is the cheapest option that is no more likely to be late than the plan.

**The simulated world.** `SyntheticWorld` holds hidden "true" behaviour for each vendor and carrier,
generates the order history from it, and can replay any order under a different decision. The model
never sees the hidden parameters. Each order keeps its own random draws, so two replays of the same
order differ only because of the decision.

## Limitations

- The data is simulated. Nothing here is a finding about a real company.
- Each decision is evaluated on its own: a changed order does not affect later orders, supplier
  capacity, or the history the model learns from.
- The cost of a late delivery and of a stockout are assumptions set by the user, and the
  recommendations depend on them.
- Splits are always 50/50, and demand for an item is approximated by its purchase history.
- With real orders the "Record from the simulation" step is replaced by entering outcomes by hand, and
  realized savings are then unknown without a controlled comparison.

## Tests

```bash
pytest
```

Twenty-two tests cover the simulation (including that a replayed order reproduces its history), the
no-leakage split, priority rules, option costing, the action store, the policy replay, the autopilot,
the reports and the command line.

## Author

Md Abdur Rauf — ORCID [0000-0002-5105-9892](https://orcid.org/0000-0002-5105-9892)

## License

MIT. See [LICENSE](LICENSE).
