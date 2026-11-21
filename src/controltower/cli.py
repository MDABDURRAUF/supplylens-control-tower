"""Command line: produce the Control Tower's outputs without opening the dashboard.

    controltower briefing  --date 2025-06-30 --out briefing.html
    controltower replay    --start 2025-01-01 --end 2025-06-30 --out performance.html
    controltower autopilot --weeks 4
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .briefing import pipeline_counts, recommendations, results_to_date, run_autopilot, whats_new
from .monitor import fit_model, open_order_board, split_as_of
from .options import OptionContext
from .replay import run_policy_replay, summarize_replay
from .report import briefing_headline, build_briefing, build_performance_report, write_html
from .store import ActionStore
from .world import SyntheticWorld

DEFAULT_AS_OF = "2025-06-30"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="controltower", description="SupplyLens Control Tower outputs.")
    p.add_argument("--version", action="version", version=f"controltower {__version__}")
    p.add_argument("--seed", type=int, default=7, help="seed of the simulated supply chain (default: 7)")
    p.add_argument("--late-cost-pct", type=float, default=0.10, help="cost of a late delivery as a share of goods value")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("briefing", help="write the daily briefing as an HTML file")
    b.add_argument("--date", help="replay date, YYYY-MM-DD (default: the date saved in the database)")
    b.add_argument("--db", default="control_tower.db", help="actions database (default: control_tower.db)")
    b.add_argument("--out", default="briefing.html")
    b.add_argument("--csv", help="also write the recommended responses to this CSV file")

    r = sub.add_parser("replay", help="replay a period against the baseline and write a performance report")
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    r.add_argument("--min-saving", type=float, default=0.0)
    r.add_argument("--out", default="performance.html")
    r.add_argument("--csv", help="also write every decision to this CSV file")

    a = sub.add_parser("autopilot", help="follow every recommendation for some weeks and record the outcomes")
    a.add_argument("--weeks", type=int, default=4)
    a.add_argument("--db", default="control_tower.db")
    a.add_argument("--owner", default="Autopilot")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        world = SyntheticWorld(seed=args.seed)
        if args.command == "replay":
            replay = run_policy_replay(world, args.start, args.end, late_cost_pct=args.late_cost_pct, min_saving=args.min_saving)
            s = summarize_replay(replay)
            print(f"Orders decided:    {s['orders']:,} ({s['share_changed']:.0%} changed)")
            print(f"Estimated saving:  ${s['estimated_saving']:,.0f}")
            print(f"Realized saving:   ${s['realized_saving']:,.0f} ({s['realized_saving_pct']:.2%} of baseline cost)")
            print(f"On-time delivery:  {s['baseline_on_time']:.1%} baseline, {s['policy_on_time']:.1%} policy")
            print(f"Wrote {write_html(build_performance_report(replay), args.out)}")
            if args.csv:
                replay.to_csv(args.csv, index=False)
                print(f"Wrote {args.csv}")
            return 0

        store = ActionStore(args.db)
        if args.command == "autopilot":
            start = store.get_meta("as_of", DEFAULT_AS_OF)
            out = run_autopilot(world, store, start, args.weeks, owner=args.owner, late_cost_pct=args.late_cost_pct)
            res = results_to_date(store.ledger())
            print(f"Ran {start} to {out['as_of']:%Y-%m-%d}: {out['actions_created']} actions approved, "
                  f"{out['outcomes_recorded']} outcomes recorded.")
            print(f"To date: estimated saving ${res['estimated_saving']:,.0f}, realized saving ${res['realized_saving']:,.0f}, "
                  f"{res['late_deliveries_avoided']} late deliveries avoided.")
            return 0

        as_of = pd.Timestamp(args.date or store.get_meta("as_of", DEFAULT_AS_OF))
        history, open_orders = split_as_of(world.orders, as_of)
        model = fit_model(history)
        board = open_order_board(model, history, open_orders, as_of, on_hand=world.stock_snapshot(as_of, history),
                                 late_cost_pct=args.late_cost_pct)
        actions, ledger = store.actions(as_of=as_of), store.ledger()
        ctx = OptionContext.build(model, history, late_cost_pct=args.late_cost_pct)
        recs, _ = recommendations(board, open_orders, ctx, exclude=actions["order_id"] if len(actions) else (), top_n=None)
        counts = pipeline_counts(board, actions, ledger)
        print(briefing_headline(board, recs, counts))
        html = build_briefing(as_of, board, recs, actions, counts, whats_new(world.orders, as_of), results_to_date(ledger))
        print(f"Wrote {write_html(html, args.out)}")
        if args.csv:
            recs.to_csv(args.csv, index=False)
            print(f"Wrote {args.csv}")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
