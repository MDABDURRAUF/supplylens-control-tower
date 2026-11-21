"""A simulated supply chain whose outcomes can be replayed under different decisions.

The Control Tower needs something real data cannot give: the outcome of the
decision that was *not* taken. ``SyntheticWorld`` holds the hidden "true"
behaviour of every vendor and carrier, generates a history of purchase orders
from it, and can then answer "what would have happened to this order if it had
gone to another vendor, another carrier, or been expedited?".

Every order carries its own fixed random draws, so the same order replayed
under two decisions differs only because of the decision (common random
numbers). That makes simulated before/after comparisons fair.

Nothing here is real company data, and the forecasting model never sees the
hidden parameters: it only sees the order history, as it would in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATEGORIES = {
    # name: (base unit price, seasonal amplitude, peak month, yearly growth)
    "Electronics": (85.0, 0.45, 11, 0.10),
    "Packaging": (2.4, 0.15, 10, 0.04),
    "Raw Materials": (14.0, 0.10, 3, 0.06),
    "Apparel": (22.0, 0.35, 8, 0.02),
    "Home Goods": (31.0, 0.25, 12, 0.08),
}
REGIONS = {"Northeast": 1.15, "Southeast": 1.00, "Midwest": 0.95, "West": 1.25, "Southwest": 1.05}
# carrier: (freight cost multiplier, effect on the log-odds of late delivery)
CARRIERS = {"RoadLink": (1.00, 0.00), "BlueFreight": (0.90, 0.35), "Summit Logistics": (1.20, -0.40)}

EXPEDITE_COST_MULTIPLIER = 2.5
EXPEDITE_LOGIT_EFFECT = -1.4
EXPEDITE_SHARE_IN_HISTORY = 0.07


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SyntheticWorld:
    """Generate order history and replay any order under an alternative decision."""

    seed: int = 7
    n_orders: int = 6000
    n_vendors: int = 10
    start: str = "2023-01-01"
    months: int = 36
    orders: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        nv = self.n_vendors
        self.vendors = [f"Vendor {chr(65 + i)}" for i in range(nv)]
        cats = list(CATEGORIES)

        # Hidden truth. Reliable vendors tend to charge a little more, so
        # there is a real trade-off between price and risk.
        late_z = rng.normal(0, 1, nv)
        self._late_logit = dict(zip(self.vendors, -1.7 + 0.75 * late_z))
        self._price_premium = dict(zip(self.vendors, 0.01 - 0.025 * late_z + rng.normal(0, 0.012, nv)))
        self._defect = dict(zip(self.vendors, rng.beta(1.5, 60, nv)))
        self._lead = dict(zip(self.vendors, rng.uniform(7, 22, nv)))

        # Each vendor serves three categories; Electronics has a single supplier.
        serves = {v: set(rng.choice(cats[1:], size=3, replace=False)) for v in self.vendors}
        serves[self.vendors[0]].add(cats[0])
        self._serves = serves
        share = rng.dirichlet(np.full(nv, 1.5))

        month_starts = pd.date_range(self.start, periods=self.months, freq="MS")
        m_idx = np.arange(self.months)
        rows, draws = [], []
        for i in range(self.n_orders):
            vendor = self.vendors[int(rng.choice(nv, p=share))]
            category = str(rng.choice(sorted(serves[vendor])))
            _, amp, peak, growth = CATEGORIES[category]
            season = 1 + amp * np.cos(2 * np.pi * (month_starts.month.to_numpy() - peak) / 12)
            weight = season * (1 + growth) ** (m_idx / 12)
            m = int(rng.choice(self.months, p=weight / weight.sum()))
            order_date = month_starts[m] + pd.Timedelta(days=int(rng.integers(0, month_starts[m].days_in_month)))
            rows.append(
                {
                    "order_id": f"PO-{100000 + i}",
                    "order_date": order_date,
                    "vendor": vendor,
                    "category": category,
                    "sku": f"{category[:3].upper()}-{int(rng.integers(1, 7)):03d}",
                    "quantity": int(np.clip(rng.lognormal(4.6, 0.8), 5, 5000)),
                    "region": str(rng.choice(list(REGIONS))),
                    "carrier": str(rng.choice(list(CARRIERS))),
                    "expedited": int(rng.random() < EXPEDITE_SHARE_IN_HISTORY),
                }
            )
            # The order's own luck, fixed once and reused for every replay of it.
            draws.append(
                {
                    "u_late": rng.random(), "u_late2": rng.random(),
                    "delay": int(rng.geometric(0.22)), "delay2": int(rng.geometric(0.22)),
                    "early": int(rng.integers(0, 3)), "early2": int(rng.integers(0, 3)),
                    "lead_noise": rng.normal(0, 1.5), "lead_noise2": rng.normal(0, 1.5),
                    "price_noise": rng.normal(0, 0.008), "price_noise2": rng.normal(0, 0.008),
                    "freight_noise": rng.lognormal(0, 0.15),
                    "u_defect": rng.random(),
                }
            )
        base = pd.DataFrame(rows)
        self._draws = pd.DataFrame(draws, index=base["order_id"])
        self._base = base.set_index("order_id")

        outcomes = pd.DataFrame([self.realize(oid) for oid in base["order_id"]])
        cols = ["order_id", "order_date", "vendor", "category", "sku", "quantity", "unit_price", "contract_price",
                "promised_date", "delivery_date", "units_rejected", "region", "carrier", "freight_cost", "expedited"]
        full = pd.concat([base, outcomes.drop(columns=["order_id", "vendor", "carrier", "expedited", "quantity"])], axis=1)
        self.orders = full[cols].sort_values("order_date").reset_index(drop=True)

    # ------------------------------------------------------------------ truth
    def vendors_for(self, category: str) -> list[str]:
        """Vendors able to supply a category."""
        return [v for v in self.vendors if category in self._serves[v]]

    def contract_price(self, sku: str) -> float:
        category = next(c for c in CATEGORIES if c[:3].upper() == sku[:3])
        return round(CATEGORIES[category][0] * (1 + 0.03 * (int(sku[-3:]) - 3)), 2)

    def true_late_probability(self, order_id: str, vendor=None, carrier=None, expedited=None, quantity=None) -> float:
        """The hidden probability of late delivery. For tests and diagnostics only."""
        o = self._base.loc[order_id]
        vendor = o["vendor"] if vendor is None else vendor
        carrier = o["carrier"] if carrier is None else carrier
        expedited = o["expedited"] if expedited is None else int(expedited)
        quantity = o["quantity"] if quantity is None else quantity
        logit = (
            self._late_logit[vendor]
            + CARRIERS[carrier][1]
            + (0.35 if o["region"] == "West" else 0.0)
            + 0.30 * (np.log(quantity) - 4.6)
            + (0.55 if o["order_date"].month in (10, 11, 12) else 0.0)
            + EXPEDITE_LOGIT_EFFECT * expedited
        )
        return float(_sigmoid(logit))

    def realize(self, order_id: str, vendor=None, carrier=None, expedited=None, quantity=None, leg: int = 1) -> dict:
        """What happens to an order under a given decision.

        With no overrides this is the order as it happened in the history.
        ``vendor``, ``carrier``, ``expedited`` and ``quantity`` replace the
        original decision. ``leg=2`` uses the order's second set of random
        draws, for the second half of a split order.
        """
        o = self._base.loc[order_id]
        d = self._draws.loc[order_id]
        sfx = "" if leg == 1 else "2"
        vendor = o["vendor"] if vendor is None else vendor
        carrier = o["carrier"] if carrier is None else carrier
        expedited = int(o["expedited"] if expedited is None else expedited)
        quantity = float(o["quantity"] if quantity is None else quantity)
        if o["category"] not in self._serves[vendor]:
            raise ValueError(f"{vendor} does not supply {o['category']}")
        if carrier not in CARRIERS:
            raise ValueError(f"Unknown carrier '{carrier}'")

        contract = self.contract_price(o["sku"])
        unit_price = round(contract * (1 + self._price_premium[vendor] + d["price_noise" + sfx]), 2)
        promised_lead = int(max(3, round(self._lead[vendor] + d["lead_noise" + sfx])))
        late = bool(d["u_late" + sfx] < self.true_late_probability(order_id, vendor, carrier, expedited, quantity))
        actual_lead = max(1, promised_lead + (int(d["delay" + sfx]) if late else -int(d["early" + sfx])))
        freight = quantity * 0.38 * REGIONS[o["region"]] * CARRIERS[carrier][0] * d["freight_noise"] + 35
        if expedited:
            freight *= EXPEDITE_COST_MULTIPLIER
        rejected = int(quantity * min(0.5, self._defect[vendor] * (0.5 + d["u_defect"])))
        return {
            "order_id": order_id,
            "vendor": vendor,
            "carrier": carrier,
            "expedited": expedited,
            "quantity": quantity,
            "unit_price": unit_price,
            "contract_price": contract,
            "promised_date": o["order_date"] + pd.Timedelta(days=promised_lead),
            "delivery_date": o["order_date"] + pd.Timedelta(days=actual_lead),
            "late": late,
            "units_rejected": rejected,
            "goods_cost": quantity * unit_price,
            "freight_cost": round(float(freight), 2),
        }

    def realize_option(self, order_id: str, option: dict | pd.Series, late_cost_pct: float) -> dict:
        """Outcome and total cost of an order under one response option.

        ``option`` needs ``vendor``, ``carrier``, ``expedited`` and
        ``split_vendor`` (empty/None unless the order is split 50/50).
        Total cost = goods + freight + ``late_cost_pct`` x goods value of any
        part that arrives late. ``late_share`` is the share of goods value
        delivered late (0 or 1 unless the order is split).
        """
        split = option.get("split_vendor")
        qty = float(self._base.loc[order_id, "quantity"])
        if isinstance(split, str) and split:
            legs = [
                self.realize(order_id, option["vendor"], option["carrier"], option["expedited"], qty / 2, leg=1),
                self.realize(order_id, split, option["carrier"], option["expedited"], qty / 2, leg=2),
            ]
        else:
            legs = [self.realize(order_id, option["vendor"], option["carrier"], option["expedited"], qty)]
        goods = sum(l["goods_cost"] for l in legs)
        freight = sum(l["freight_cost"] for l in legs)
        late_goods = sum(l["goods_cost"] for l in legs if l["late"])
        return {
            "delivery_date": max(l["delivery_date"] for l in legs),
            "late": any(l["late"] for l in legs),
            "late_share": late_goods / goods,
            "goods_cost": goods,
            "freight_cost": freight,
            "late_cost": late_cost_pct * late_goods,
            "total_cost": goods + freight + late_cost_pct * late_goods,
        }

    def baseline_option(self, order_id: str) -> dict:
        """The decision actually recorded in the history for this order."""
        o = self._base.loc[order_id]
        return {"vendor": o["vendor"], "carrier": o["carrier"], "expedited": int(o["expedited"]), "split_vendor": None}

    # ------------------------------------------------------------------ stock
    def stock_snapshot(self, as_of, history: pd.DataFrame | None = None) -> pd.Series:
        """Simulated units on hand per SKU on a given date.

        Each SKU holds between 4 and 40 days of its average daily demand. The
        level changes week to week and is reproducible for a given seed.
        """
        as_of = pd.Timestamp(as_of)
        source = self.orders if history is None else history
        span_days = max((source["order_date"].max() - source["order_date"].min()).days, 1)
        daily = source.groupby("sku")["quantity"].sum() / span_days
        week = as_of.toordinal() // 7
        cover = {
            sku: np.random.default_rng([self.seed, week, i]).uniform(4, 40) for i, sku in enumerate(sorted(daily.index))
        }
        return (daily * pd.Series(cover)).round().rename("on_hand")
