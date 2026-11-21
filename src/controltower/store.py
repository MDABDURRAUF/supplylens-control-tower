"""Persistent record of actions and outcomes (SQLite).

An *action* is a decision about one order: which response option was chosen,
who owns it, when it is due, whether it was approved, and how far along it is.
It also freezes the estimates made at decision time, so they can later be
compared with what actually happened.

An *outcome* is what was observed once the order arrived: delivery date, final
cost, whether it was late, and whether the recommendation was really adopted.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

DECISIONS = ["pending", "approved", "rejected"]
STATUSES = ["open", "in progress", "done", "cancelled"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    action_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id       TEXT NOT NULL,
    option         TEXT NOT NULL,
    kind           TEXT NOT NULL,
    vendor         TEXT, carrier TEXT, expedited INTEGER, split_vendor TEXT,
    owner          TEXT NOT NULL,
    due_date       TEXT NOT NULL,
    decision       TEXT NOT NULL DEFAULT 'pending',
    status         TEXT NOT NULL DEFAULT 'open',
    note           TEXT DEFAULT '',
    created_on     TEXT NOT NULL,
    est_keep_cost  REAL, est_option_cost REAL,
    est_keep_risk  REAL, est_option_risk REAL
);
CREATE TABLE IF NOT EXISTS outcomes (
    action_id        INTEGER PRIMARY KEY REFERENCES actions(action_id),
    adopted          INTEGER NOT NULL,
    delivery_date    TEXT NOT NULL,
    late             INTEGER NOT NULL,
    final_cost       REAL NOT NULL,
    baseline_cost    REAL,
    baseline_late    INTEGER,
    source           TEXT NOT NULL,
    recorded_on      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

_EDITABLE = {"owner", "due_date", "decision", "status", "note"}


class ActionStore:
    """Actions and outcomes in one SQLite file. Use ``":memory:"`` for tests."""

    def __init__(self, path: str | Path = "control_tower.db"):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ---------------------------------------------------------------- actions
    def add_action(self, order_id: str, option: pd.Series | dict, keep: pd.Series | dict, owner: str, due_date,
                   created_on, note: str = "", decision: str = "pending") -> int:
        """Create an action for ``order_id`` choosing ``option`` over ``keep``.

        ``option`` and ``keep`` are rows from ``OptionContext.options``; their
        expected cost and risk are stored as the estimates made at decision time.
        """
        if not owner or not str(owner).strip():
            raise ValueError("An action needs an owner")
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        split = option.get("split_vendor")
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO actions (order_id, option, kind, vendor, carrier, expedited, split_vendor, owner, due_date,"
                " decision, note, created_on, est_keep_cost, est_option_cost, est_keep_risk, est_option_risk)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (order_id, option["option"], option["kind"], option["vendor"], option["carrier"], int(option["expedited"]),
                 split if isinstance(split, str) and split else None, str(owner).strip(),
                 str(pd.Timestamp(due_date).date()), decision, note, str(pd.Timestamp(created_on).date()),
                 float(keep["expected_cost"]), float(option["expected_cost"]),
                 float(keep["late_probability"]), float(option["late_probability"])),
            )
        return int(cur.lastrowid)

    def update_action(self, action_id: int, **changes) -> None:
        """Change owner, due_date, decision, status or note of an action."""
        unknown = set(changes) - _EDITABLE
        if unknown:
            raise ValueError(f"Cannot change: {', '.join(sorted(unknown))}")
        if "decision" in changes and changes["decision"] not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if "status" in changes and changes["status"] not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if "due_date" in changes:
            changes["due_date"] = str(pd.Timestamp(changes["due_date"]).date())
        if not changes:
            return
        sets = ", ".join(f"{k} = ?" for k in changes)
        with self._conn:
            cur = self._conn.execute(f"UPDATE actions SET {sets} WHERE action_id = ?", (*changes.values(), action_id))
        if cur.rowcount == 0:
            raise KeyError(f"No action {action_id}")

    def actions(self, as_of=None) -> pd.DataFrame:
        """All actions, with the estimated saving and whether each is overdue on ``as_of``."""
        df = pd.read_sql_query("SELECT * FROM actions ORDER BY action_id", self._conn, parse_dates=["due_date", "created_on"])
        df["est_saving"] = df["est_keep_cost"] - df["est_option_cost"]
        if as_of is not None:
            active = df["status"].isin(["open", "in progress"])
            df["overdue"] = active & (df["due_date"] < pd.Timestamp(as_of).normalize())
        return df

    # --------------------------------------------------------------- outcomes
    def record_outcome(self, action_id: int, adopted: bool, delivery_date, late: bool, final_cost: float, recorded_on,
                       baseline_cost: float | None = None, baseline_late: bool | None = None,
                       source: str = "entered") -> None:
        """Record what actually happened to the order behind an action.

        ``baseline_cost`` and ``baseline_late`` are what keeping the original
        plan would have produced. They are only knowable in a simulation;
        leave them empty for real orders. Recording again replaces the
        earlier outcome for the same action.
        """
        if final_cost < 0:
            raise ValueError("final_cost cannot be negative")
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?,?,?,?,?,?)",
                (action_id, int(adopted), str(pd.Timestamp(delivery_date).date()), int(late), float(final_cost),
                 None if baseline_cost is None else float(baseline_cost),
                 None if baseline_late is None else int(baseline_late), source, str(pd.Timestamp(recorded_on).date())),
            )

    def ledger(self) -> pd.DataFrame:
        """Actions joined to their outcomes (one row per action; outcome columns empty until recorded).

        Adds three separate measures that must not be confused:

        ``est_saving``         estimated at decision time, from the model
        ``est_error``          final cost minus the estimated cost of the
                               option that was actually followed
        ``realized_saving``    baseline cost minus final cost, only where the
                               baseline outcome is known (simulation)
        """
        df = pd.read_sql_query(
            "SELECT a.*, o.adopted, o.delivery_date, o.late, o.final_cost, o.baseline_cost, o.baseline_late, o.source,"
            " o.recorded_on FROM actions a LEFT JOIN outcomes o USING (action_id) ORDER BY a.action_id",
            self._conn, parse_dates=["due_date", "created_on", "delivery_date", "recorded_on"],
        )
        df["est_saving"] = df["est_keep_cost"] - df["est_option_cost"]
        followed = df["est_option_cost"].where(df["adopted"] == 1, df["est_keep_cost"])
        df["est_error"] = df["final_cost"] - followed
        df["realized_saving"] = df["baseline_cost"] - df["final_cost"]
        return df

    # ------------------------------------------------------------------- meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with closing(self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,))) as cur:
            row = cur.fetchone()
        return default if row is None else row[0]

    def set_meta(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))

    def reset(self) -> None:
        """Delete every action and outcome."""
        with self._conn:
            self._conn.execute("DELETE FROM outcomes")
            self._conn.execute("DELETE FROM actions")
