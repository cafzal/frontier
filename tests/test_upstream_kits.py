"""Upstream-kit sufficiency: BRIEF.md + data.csv must reconstruct the canonical bundle.

Two examples ship an upstream kit (a user-voiced brief plus raw CSV) so a session can
start at FRAME the way a real user would. The kit's promise is that framing that input
lands on exactly the shipped problem.json + scores.json — these tests reconstruct the
model from the CSV plus the brief's stated rules and diff it against the bundle, so the
kit can't silently drift from the canonical model (or vice versa).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _canon(constraints):
    return sorted(json.dumps(c, sort_keys=True) for c in constraints)


def _rows(example, fname="data.csv"):
    with open(EXAMPLES / example / fname, newline="") as f:
        return list(csv.DictReader(f))


def _norm_exclusion(c):
    if c.get("type") == "exclusion_pair" and c["option_a"] > c["option_b"]:
        c = dict(c)
        c["option_a"], c["option_b"] = c["option_b"], c["option_a"]
    return c


def test_capital_kit_reconstructs_canonical_model():
    rows = _rows("capital_project_selection_120")
    # The brief's stated rules: $610M budget, 18-40 projects, category caps
    # Growth 8 / Digital 6 / R&D 6 / Maintenance 7; CSV carries requires /
    # mutually_exclusive_with / committed.
    built = [{"type": "objective_bound", "objective": "Cost", "operator": "max", "value": 610.0},
             {"type": "cardinality", "min": 18, "max": 40}]
    built += [{"type": "force_include", "option": r["project"]} for r in rows if r["committed"] == "yes"]
    built += [{"type": "dependency", "if_option": r["project"], "then_option": r["requires"]}
              for r in rows if r["requires"]]
    seen = set()
    for r in rows:
        for other in filter(None, r["mutually_exclusive_with"].split("|")):
            pair = tuple(sorted([r["project"], other]))
            if pair not in seen:
                seen.add(pair)
                built.append({"type": "exclusion_pair", "option_a": pair[0], "option_b": pair[1]})
    bycat = defaultdict(list)
    for r in rows:
        bycat[r["category"]].append(r["project"])
    for cat, mx in {"Growth": 8, "Digital": 6, "R&D": 6, "Maintenance": 7}.items():
        built.append({"type": "group_limit", "options": bycat[cat], "min": 0, "max": mx})
    built_scores = [{"option": r["project"], "objective": obj, "value": float(r[col])}
                    for r in rows
                    for obj, col in [("NPV", "npv_musd"), ("Cost", "cost_musd"),
                                     ("Risk", "risk_score"), ("StrategicFit", "strategic_fit")]]

    p = json.load(open(EXAMPLES / "capital_project_selection_120" / "problem.json"))
    s = json.load(open(EXAMPLES / "capital_project_selection_120" / "scores.json"))
    canon_cs = [_norm_exclusion(dict(c)) for c in p["constraints"]]
    assert _canon(canon_cs) == _canon(built)
    assert sorted(json.dumps(r, sort_keys=True) for r in s["scores"]) == \
           sorted(json.dumps(r, sort_keys=True) for r in built_scores)
    assert [o["name"] for o in s["options"]] == [r["project"] for r in rows]


def test_rationing_kit_reconstructs_canonical_model_and_scenarios():
    rows = _rows("scarce_supply_rationing")
    # The brief's stated rules: 8% cap, mandate StrategicValue >= 4.8; CSV carries
    # contract floors and credit caps; outage restates all rules with floors
    # 8/7/5/5/4; surge = DST revenue x1.35 (2dp).
    built = [{"type": "max_allocation", "max": 8},
             {"type": "objective_bound", "objective": "StrategicValue", "operator": "min", "value": 4.8}]
    for r in rows:
        if r["contract_floor_pct"]:
            built.append({"type": "allocation_bound", "option": r["customer"],
                          "min": int(r["contract_floor_pct"]), "max": 100})
        if r["credit_cap_pct"]:
            built.append({"type": "allocation_bound", "option": r["customer"],
                          "min": 0, "max": int(r["credit_cap_pct"])})
    built_scores = [{"option": r["customer"], "objective": obj, "value": float(r[col])}
                    for r in rows
                    for obj, col in [("Revenue", "revenue_per_1pct_musd"),
                                     ("StrategicValue", "strategic_value_per_1pct"),
                                     ("DemandFragility", "demand_fragility_per_1pct")]]
    raised = {"HYP-01": 8, "HYP-02": 7, "HYP-03": 5, "IND-01": 5, "IND-02": 4}
    outage = [{"type": "max_allocation", "max": 8},
              {"type": "objective_bound", "objective": "StrategicValue", "operator": "min", "value": 4.8}]
    outage += [{"type": "allocation_bound", "option": k, "min": v, "max": 100} for k, v in raised.items()]
    outage += [{"type": "allocation_bound", "option": "DST-01", "min": 0, "max": 4},
               {"type": "allocation_bound", "option": "DST-02", "min": 0, "max": 5}]
    surge = [{"option": r["customer"], "objective": "Revenue",
              "value": round(float(r["revenue_per_1pct_musd"]) * 1.35, 2)}
             for r in rows if r["segment"] == "Distributor"]

    p = json.load(open(EXAMPLES / "scarce_supply_rationing" / "problem.json"))
    s = json.load(open(EXAMPLES / "scarce_supply_rationing" / "scores.json"))
    assert _canon(p["constraints"]) == _canon(built)
    assert sorted(json.dumps(r, sort_keys=True) for r in s["scores"]) == \
           sorted(json.dumps(r, sort_keys=True) for r in built_scores)
    scn = {x["name"]: x for x in p["scenarios"]}
    assert _canon(scn["fab_outage"]["constraint_overrides"]) == _canon(outage)
    assert sorted(json.dumps(r, sort_keys=True) for r in scn["spot_surge"]["score_overrides"]) == \
           sorted(json.dumps(r, sort_keys=True) for r in surge)
