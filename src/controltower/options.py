"""Response options for an open order, each with an expected cost and risk.

Options considered:

* keep the order as planned (the baseline every other option is compared to);
* expedite the shipment;
* change the carrier;
* change the supplier;
* split the order 50/50 between the current supplier and another one.

Supplier, carrier and split changes are only offered while an order is
``pending release``. Once it is in transit or overdue, only expediting remains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import supplylens as sl

OPTION_COLUMNS = [
    "option", "kind", "vendor", "carrier", "expedited", "split_vendor", "late_probability", "goods_cost",
    "freight_cost", "expected_late_cost", "expected_cost", "saving_vs_keep", "risk_change",
]


@dataclass
class OptionContext:
    """Everything needed to price options, estimated once from the order history.

    Build it with ``OptionContext.build(model, history)`` and reuse it for
    every open order on the same date.
    """

    model: sl.LateDeliveryModel
    price: pd.Series  # (sku, vendor) -> average unit price
    lead_days: pd.Series  # vendor -> median promised lead time
    freight_fit: dict  # (region, carrier) -> (fixed charge, charge per unit)
    expedite_multiplier: float | None  # None when history has no expedited shipments
    uses_expedited: bool
    late_cost_pct: float

    @classmethod
    def build(cls, model, history: pd.DataFrame, late_cost_pct: float = 0.10, min_history_lines: int = 5):
        hist = sl.data.add_derived_columns(history)
        g = hist.groupby(["sku", "vendor"])
        enough = g.size() >= min_history_lines
        price = (g["spend"].sum() / g["quantity"].sum())[enough]
        lead_days = hist.groupby("vendor")["promised_lead_days"].median()

        has_flag = "expedited" in hist.columns
        normal = hist[hist["expedited"] == 0] if has_flag else hist
        freight_fit = {}
        for lane, grp in normal.groupby(["region", "carrier"]):
            if len(grp) >= 10 and grp["quantity"].nunique() > 1:
                per_unit, fixed = np.polyfit(grp["quantity"], grp["freight_cost"], 1)
                freight_fit[lane] = (float(fixed), float(per_unit))

        multiplier = None
        if has_flag and (hist["expedited"] == 1).sum() >= 10:
            fast = hist[hist["expedited"] == 1]
            predicted = [cls._lane_cost(freight_fit, r, c, q) for r, c, q in zip(fast["region"], fast["carrier"], fast["quantity"])]
            ratio = (fast["freight_cost"] / pd.Series(predicted, index=fast.index)).dropna()
            multiplier = float(ratio.median()) if len(ratio) else None
        return cls(model, price, lead_days, freight_fit, multiplier, "expedited" in model.extra_numeric, late_cost_pct)

    @staticmethod
    def _lane_cost(freight_fit: dict, region: str, carrier: str, quantity: float) -> float:
        if (region, carrier) not in freight_fit:
            return np.nan
        fixed, per_unit = freight_fit[(region, carrier)]
        return fixed + per_unit * quantity

    # ------------------------------------------------------------ candidates
    def _leg(self, order: pd.Series, vendor: str, carrier: str, expedited: int, quantity: float) -> dict | None:
        """Estimated cost inputs for one shipment; None if it cannot be priced."""
        same_vendor = vendor == order["vendor"]
        if same_vendor:
            unit_price, promised = float(order["unit_price"]), order["promised_date"]
        else:
            if (order["sku"], vendor) not in self.price.index or vendor not in self.lead_days.index:
                return None
            unit_price = float(self.price[(order["sku"], vendor)])
            promised = order["order_date"] + pd.Timedelta(days=round(float(self.lead_days[vendor])))

        unchanged = same_vendor and carrier == order["carrier"] and quantity == order["quantity"]
        was_expedited = int(order.get("expedited", 0))
        if unchanged:
            # The freight charge on the order is known; only an expedite decision changes it.
            freight = float(order["freight_cost"])
            if expedited and not was_expedited:
                freight *= self.expedite_multiplier
        else:
            freight = self._lane_cost(self.freight_fit, order["region"], carrier, quantity)
            if np.isnan(freight):
                return None
            if expedited:
                freight *= self.expedite_multiplier
        return {
            "order_date": order["order_date"], "promised_date": promised, "vendor": vendor, "carrier": carrier,
            "region": order["region"], "category": order["category"], "quantity": quantity, "expedited": expedited,
            "goods_cost": quantity * unit_price, "freight_cost": freight,
        }

    def options(self, order: pd.Series, stage: str = "pending release") -> pd.DataFrame:
        """All response options for one open order, cheapest expected cost first.

        ``late_probability`` is the model's estimate; for a split order it is
        the expected share of goods value arriving late.

            expected_cost = goods_cost + freight_cost + expected_late_cost
            expected_late_cost = late probability x late_cost_pct x goods value

        ``saving_vs_keep`` and ``risk_change`` compare each option with
        keeping the order as planned.
        """
        vendor, carrier = order["vendor"], order["carrier"]
        qty = float(order["quantity"])
        expedited = int(order.get("expedited", 0))
        can_expedite = self.uses_expedited and self.expedite_multiplier is not None and not expedited

        plans = [("Keep as planned", "keep", [(vendor, carrier, expedited, qty)], None)]
        if can_expedite:
            plans.append(("Expedite shipping", "expedite", [(vendor, carrier, 1, qty)], None))
        if stage == "pending release":
            for c in sorted({lane[1] for lane in self.freight_fit if lane[0] == order["region"]} - {carrier}):
                plans.append((f"Change carrier to {c}", "carrier", [(vendor, c, expedited, qty)], None))
            alternatives = [v for v in self.price.loc[order["sku"]].index if v != vendor] if order["sku"] in self.price.index.get_level_values(0) else []
            for v in sorted(alternatives):
                plans.append((f"Change supplier to {v}", "supplier", [(v, carrier, expedited, qty)], None))
                plans.append((f"Split 50/50 with {v}", "split", [(vendor, carrier, expedited, qty / 2), (v, carrier, expedited, qty / 2)], v))

        rows, legs = [], []
        for name, kind, shipments, split_vendor in plans:
            priced = [self._leg(order, *s) for s in shipments]
            if any(p is None for p in priced):
                continue
            for p in priced:
                legs.append({**p, "plan": len(rows)})
            first = shipments[0]
            rows.append({"option": name, "kind": kind, "vendor": first[0], "carrier": first[1], "expedited": first[2],
                         "split_vendor": split_vendor})

        legs = pd.DataFrame(legs)
        legs["p_late"] = self.model.predict_proba(legs).to_numpy()
        legs["late_goods"] = legs["p_late"] * legs["goods_cost"]
        totals = legs.groupby("plan")[["goods_cost", "freight_cost", "late_goods"]].sum()

        out = pd.DataFrame(rows)
        out["goods_cost"] = totals["goods_cost"]
        out["freight_cost"] = totals["freight_cost"]
        # Expected share of goods value that arrives late.
        out["late_probability"] = totals["late_goods"] / totals["goods_cost"]
        out["expected_late_cost"] = self.late_cost_pct * totals["late_goods"]
        out["expected_cost"] = out["goods_cost"] + out["freight_cost"] + out["expected_late_cost"]
        keep = out.iloc[0]
        out["saving_vs_keep"] = keep["expected_cost"] - out["expected_cost"]
        out["risk_change"] = out["late_probability"] - keep["late_probability"]
        return out[OPTION_COLUMNS].sort_values("expected_cost").reset_index(drop=True)


def recommend(options: pd.DataFrame, min_saving: float = 0.0) -> pd.Series:
    """Pick the option to act on.

    The recommendation is the lowest expected cost among options that are no
    more likely to be late than keeping the order as planned. If nothing beats
    the plan by more than ``min_saving``, the recommendation is to keep it.
    """
    keep = options[options["kind"] == "keep"].iloc[0]
    eligible = options[(options["risk_change"] <= 0) & (options["saving_vs_keep"] > min_saving)]
    return keep if eligible.empty else eligible.sort_values("expected_cost").iloc[0]
