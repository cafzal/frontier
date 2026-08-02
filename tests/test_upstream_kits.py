"""Upstream-kit sufficiency: each README's Step-1 ask + data.csv (+ matrix CSVs) must reconstruct the
canonical bundle.

Every example ships an upstream kit (a user-voiced ask in the README plus raw CSVs) so a session can
start at FRAME the way a real user would. The kit's promise is that framing that input
lands on exactly the shipped problem.json + scores.json — these tests reconstruct each
model from the CSVs plus the ask's stated rules and diff it against the bundle, so the
kits can't silently drift from the canonical models (or vice versa).
"""
import csv
import functools
import json
from collections import defaultdict
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# ─── helpers ───

@functools.lru_cache(maxsize=None)
def _rows(example, fname="data.csv"):
    with open(EXAMPLES / example / fname, newline="") as f:
        return tuple(csv.DictReader(f))


def _bundle(example):
    p = json.load(open(EXAMPLES / example / "problem.json"))
    s = json.load(open(EXAMPLES / example / "scores.json"))
    return p, s


def _norm(c):
    """Order-insensitive normal form for a constraint dict."""
    c = dict(c)
    if c.get("type") == "exclusion_pair" and c["option_a"] > c["option_b"]:
        c["option_a"], c["option_b"] = c["option_b"], c["option_a"]
    if c.get("type") == "group_limit":
        c["options"] = sorted(c["options"])
        c.setdefault("min", 0)
    return c


def _canon(constraints):
    return sorted(json.dumps(_norm(c), sort_keys=True) for c in constraints)


def _canon_scores(scores):
    return sorted(json.dumps({"option": r["option"], "objective": r["objective"],
                              "value": float(r["value"])}, sort_keys=True) for r in scores)


def _assert_model(example, built_constraints, built_scores, objectives, option_key):
    p, s = _bundle(example)
    assert [(o["name"], o["direction"], o["aggregation"]) for o in p["objectives"]] == objectives
    assert _canon(p["constraints"]) == _canon(built_constraints)
    assert _canon_scores(s["scores"]) == _canon_scores(built_scores)
    assert [o["name"] for o in s["options"]] == [r[option_key] for r in _rows(example)]
    return p, s


def _scores_from(rows, option_key, colmap):
    return [{"option": r[option_key], "objective": obj, "value": float(r[col])}
            for r in rows for obj, col in colmap]


def _matrix_from_csv(example, fname):
    with open(EXAMPLES / example / fname, newline="") as f:
        raw = list(csv.reader(f))
    names = raw[0][1:]
    entries = {}
    for row in raw[1:]:
        vals = {b: float(v) for b, v in zip(names, row[1:]) if v != ""}
        if vals:
            entries[row[0]] = vals
    return entries


def _coerce_matrix(entries):
    return {a: {b: float(v) for b, v in row.items()} for a, row in entries.items() if row}


def _assert_matrix(example, fname, canonical_entries):
    parsed = _matrix_from_csv(example, fname)
    # An empty CSV compared against empty canonical entries passes while proving
    # nothing — and it hides the real defect: a kit file shipping headers with no
    # data, which the runbook still tells the user to paste. Fail loudly instead.
    assert parsed, (
        f"{example}/{fname} parsed to an empty matrix — a kit CSV must carry values, "
        "or the bundle should not claim that file reconstructs the matrix")
    assert parsed == _coerce_matrix(canonical_entries)


def _groups(rows, option_key, group_key):
    g = defaultdict(list)
    for r in rows:
        if r[group_key]:
            g[r[group_key]].append(r[option_key])
    return g


def _exclusions(rows, option_key, col):
    seen, out = set(), []
    for r in rows:
        for other in filter(None, r[col].split("|")):
            pair = tuple(sorted([r[option_key], other]))
            if pair not in seen:
                seen.add(pair)
                out.append({"type": "exclusion_pair", "option_a": pair[0], "option_b": pair[1]})
    return out


def _norm_adjustments(adjs):
    return sorted((a["objective"], a.get("multiply"), a.get("add")) for a in adjs)


def _scenario(p, name):
    return next(x for x in p["scenarios"] if x["name"] == name)


# ─── the two demo kits (validated first) ───

def test_capital_kit_reconstructs_canonical_model():
    rows = _rows("capital_project_selection_300")
    built = [{"type": "objective_bound", "objective": "Cost", "operator": "max", "value": 1550.0},
             {"type": "cardinality", "min": 45, "max": 100}]
    built += [{"type": "force_include", "option": r["project"]} for r in rows if r["committed"] == "yes"]
    built += [{"type": "dependency", "if_option": r["project"], "then_option": r["requires"]}
              for r in rows if r["requires"]]
    built += _exclusions(rows, "project", "mutually_exclusive_with")
    groups = _groups(rows, "project", "category")
    for cat, mx in {"Growth": 20, "Digital": 15, "R&D": 15, "Maintenance": 18}.items():
        built.append({"type": "group_limit", "options": groups[cat], "min": 0, "max": mx})
    scores = _scores_from(rows, "project", [("NPV", "npv_musd"), ("Cost", "cost_musd"),
                                            ("Risk", "risk_score"), ("StrategicFit", "strategic_fit")])
    p, _ = _assert_model("capital_project_selection_300", built, scores,
                         [("NPV", "maximize", "sum"), ("Cost", "minimize", "sum"),
                          ("Risk", "minimize", "sum"), ("StrategicFit", "maximize", "sum")], "project")
    assert _norm_adjustments(_scenario(p, "cost_inflation")["score_adjustments"]) == \
           _norm_adjustments([{"objective": "Cost", "multiply": 1.12}])
    crunch = [dict(c) for c in built]
    crunch[1] = {"type": "cardinality", "min": 45, "max": 70}
    assert _canon(_scenario(p, "delivery_crunch")["constraint_overrides"]) == _canon(crunch)


def test_rationing_kit_reconstructs_canonical_model_and_scenarios():
    rows = _rows("scarce_supply_rationing")
    built = [{"type": "max_allocation", "max": 8},
             {"type": "objective_bound", "objective": "StrategicValue", "operator": "min", "value": 4.8}]
    for r in rows:
        if r["contract_floor_pct"]:
            built.append({"type": "allocation_bound", "option": r["customer"],
                          "min": int(r["contract_floor_pct"]), "max": 100})
        if r["credit_cap_pct"]:
            built.append({"type": "allocation_bound", "option": r["customer"],
                          "min": 0, "max": int(r["credit_cap_pct"])})
    scores = _scores_from(rows, "customer", [("Revenue", "revenue_per_1pct_musd"),
                                             ("StrategicValue", "strategic_value_per_1pct"),
                                             ("DemandFragility", "demand_fragility_per_1pct")])
    p, _ = _assert_model("scarce_supply_rationing", built, scores,
                         [("Revenue", "maximize", "sum"), ("StrategicValue", "maximize", "sum"),
                          ("DemandFragility", "minimize", "sum")], "customer")
    raised = {"HYP-01": 8, "HYP-02": 7, "HYP-03": 5, "IND-01": 5, "IND-02": 4}
    outage = [{"type": "max_allocation", "max": 8},
              {"type": "objective_bound", "objective": "StrategicValue", "operator": "min", "value": 4.8}]
    outage += [{"type": "allocation_bound", "option": k, "min": v, "max": 100} for k, v in raised.items()]
    outage += [{"type": "allocation_bound", "option": "DST-01", "min": 0, "max": 4},
               {"type": "allocation_bound", "option": "DST-02", "min": 0, "max": 5}]
    assert _canon(_scenario(p, "fab_outage")["constraint_overrides"]) == _canon(outage)
    surge = [{"option": r["customer"], "objective": "Revenue",
              "value": round(float(r["revenue_per_1pct_musd"]) * 1.35, 2)}
             for r in rows if r["segment"] == "Distributor"]
    assert _canon_scores(_scenario(p, "spot_surge")["score_overrides"]) == _canon_scores(surge)


# ─── the remaining ten ───

def test_budget_allocation_kit():
    rows = _rows("budget_allocation")
    built = [{"type": "max_allocation", "max": 20},
             {"type": "objective_bound", "objective": "TimeToImpact", "operator": "max", "value": 9.0}]
    built += [{"type": "group_limit", "options": opts, "min": 0, "max": 3}
              for opts in _groups(rows, "initiative", "department").values()]
    scores = _scores_from(rows, "initiative", [("ROI", "roi_pct"), ("Strategic Reach", "strategic_reach"),
                                               ("TimeToImpact", "time_to_impact_months")])
    p, _ = _assert_model("budget_allocation", built, scores,
                         [("ROI", "maximize", "avg"), ("Strategic Reach", "maximize", "avg"),
                          ("TimeToImpact", "minimize", "avg")], "initiative")
    haircut = [{"option": r["initiative"], "objective": "ROI", "value": float(r["roi_under_downturn"])}
               for r in rows if r["roi_under_downturn"]]
    assert _canon_scores(_scenario(p, "downturn")["score_overrides"]) == _canon_scores(haircut)


def test_production_mix_kit():
    rows = _rows("production_mix")
    built = [{"type": "max_allocation", "max": 30}]
    built += [{"type": "group_limit", "options": opts, "min": 0, "max": 2}
              for opts in _groups(rows, "product", "line").values()]
    scores = _scores_from(rows, "product", [("Margin", "margin_usd_unit"),
                                            ("Throughput", "throughput_kunits_wk"),
                                            ("Sustainability", "sustainability")])
    p, _ = _assert_model("production_mix", built, scores,
                         [("Margin", "maximize", "avg"), ("Throughput", "maximize", "avg"),
                          ("Sustainability", "maximize", "avg")], "product")
    spike = [{"option": r["product"], "objective": "Margin", "value": float(r["margin_under_input_cost_spike"])}
             for r in rows if r["margin_under_input_cost_spike"]]
    assert _canon_scores(_scenario(p, "input_cost_spike")["score_overrides"]) == _canon_scores(spike)
    crunch = [dict(c) for c in built]
    crunch[0] = {"type": "max_allocation", "max": 25}
    assert _canon(_scenario(p, "capacity_crunch")["constraint_overrides"]) == _canon(crunch)


def test_channel_budget_kit():
    rows = _rows("channel_budget")
    built = [{"type": "max_allocation", "max": 15},
             {"type": "objective_bound", "objective": "ROAS", "operator": "min", "value": 2.0}]
    built += [{"type": "group_limit", "options": opts, "min": 0, "max": 2}
              for opts in _groups(rows, "channel", "platform_group").values()]
    scores = _scores_from(rows, "channel", [("Conversions", "conversions"), ("Reach", "reach"),
                                            ("ROAS", "roas"), ("BrandLift", "brand_lift")])
    p, s = _assert_model("channel_budget", built, scores,
                         [("Conversions", "maximize", "avg"), ("Reach", "maximize", "quadratic"),
                          ("ROAS", "maximize", "avg"), ("BrandLift", "maximize", "avg")], "channel")
    _assert_matrix("channel_budget", "reach_overlap.csv", s["interaction_matrices"][0]["entries"])
    assert _norm_adjustments(_scenario(p, "signal_loss")["score_adjustments"]) == \
           _norm_adjustments([{"objective": "Conversions", "multiply": 0.8}])
    assert _norm_adjustments(_scenario(p, "demand_pullback")["score_adjustments"]) == \
           _norm_adjustments([{"objective": "Conversions", "multiply": 0.85},
                              {"objective": "ROAS", "multiply": 0.9}])


def test_supplier_selection_kit():
    rows = _rows("supplier_selection")
    base = [{"type": "max_allocation", "max": 15},
            {"type": "objective_bound", "objective": "Reliability", "operator": "min", "value": 78.0}]
    base += [{"type": "group_limit", "options": opts, "min": 0, "max": 3}
             for opts in _groups(rows, "supplier", "region").values()]
    floors = [{"type": "allocation_bound", "option": r["supplier"],
               "min": int(r["contract_floor_pct"]), "max": 100}
              for r in rows if r["contract_floor_pct"]]
    base += floors
    scores = _scores_from(rows, "supplier", [("Cost", "cost_usd_unit"), ("Reliability", "reliability"),
                                             ("LeadTime", "lead_time_days"), ("ESGRisk", "esg_risk"),
                                             ("ConcentrationRisk", "concentration_risk")])
    p, s = _assert_model("supplier_selection", base, scores,
                         [("Cost", "minimize", "avg"), ("Reliability", "maximize", "avg"),
                          ("LeadTime", "minimize", "avg"), ("ESGRisk", "minimize", "avg"),
                          ("ConcentrationRisk", "minimize", "quadratic")], "supplier")
    _assert_matrix("supplier_selection", "concentration_interactions.csv",
                   s["interaction_matrices"][0]["entries"])
    # china_disruption: CN throttled to ≤5%; the CN01 contract floor survives as one MERGED
    # bound (min 4, max 5) — allocation_bound is last-wins per option, so the scenario ships
    # the intersection explicitly rather than a floor entry plus a cap entry.
    china = [c for c in base if not (c["type"] == "allocation_bound" and c["option"] == "CN01")]
    china += [{"type": "allocation_bound", "option": "CN01", "min": 4, "max": 5}]
    china += [{"type": "allocation_bound", "option": r["supplier"], "min": 0, "max": 5}
              for r in rows if r["region"] == "CN" and r["supplier"] != "CN01"]
    assert _canon(_scenario(p, "china_disruption")["constraint_overrides"]) == _canon(china)
    surge = [dict(c) for c in base]
    surge[0] = {"type": "max_allocation", "max": 10}
    assert _canon(_scenario(p, "demand_surge")["constraint_overrides"]) == _canon(surge)


def test_capacity_planning_kit():
    rows = _rows("capacity_planning")
    built = [{"type": "max_allocation", "max": 25},
             {"type": "objective_bound", "objective": "CO2", "operator": "max", "value": 0.2},
             {"type": "objective_bound", "objective": "Firmness", "operator": "min", "value": 50.0}]
    scores = _scores_from(rows, "project", [("LCOE", "lcoe_usd_mwh"), ("CO2", "co2_t_mwh"),
                                            ("Firmness", "firmness"), ("VariabilityRisk", "variability_risk"),
                                            ("LandUse", "land_use")])
    p, s = _assert_model("capacity_planning", built, scores,
                         [("LCOE", "minimize", "avg"), ("CO2", "minimize", "avg"),
                          ("Firmness", "maximize", "avg"), ("VariabilityRisk", "minimize", "quadratic"),
                          ("LandUse", "minimize", "avg")], "project")
    _assert_matrix("capacity_planning", "variability_interactions.csv",
                   s["interaction_matrices"][0]["entries"])
    assert _norm_adjustments(_scenario(p, "carbon_price")["score_adjustments"]) == \
           _norm_adjustments([{"objective": "LCOE", "multiply": 1.15}])
    mo = _scenario(p, "low_renewables_year")["interaction_matrix_overrides"][0]
    assert mo["objective"] == "VariabilityRisk"
    # The poor-renewable-year regime shift is a scale factor on the renewable block,
    # not a replacement matrix: co-movement rises 1.4x within solar+wind and nothing
    # else changes. Assert that mechanism directly — there is no kit CSV for it.
    assert mo["mode"] == "upsert" and not mo["entries"]
    assert [g["factor"] for g in mo["scale_groups"]] == [1.4]
    assert all(o.startswith(("Solar_", "Wind_")) for o in mo["scale_groups"][0]["options"])
    surge = [dict(c) for c in built]
    surge[2] = {"type": "objective_bound", "objective": "Firmness", "operator": "min", "value": 60.0}
    assert _canon(_scenario(p, "demand_surge")["constraint_overrides"]) == _canon(surge)


def test_investment_portfolio_kit():
    rows = _rows("investment_portfolio")
    built = [{"type": "max_allocation", "max": 30},
             {"type": "objective_bound", "objective": "Volatility", "operator": "max", "value": 20.0}]
    built += [{"type": "group_limit", "options": opts, "min": 0, "max": 3}
              for opts in _groups(rows, "fund", "asset_class_group").values()]
    scores = _scores_from(rows, "fund", [("Return", "expected_return"), ("Volatility", "volatility"),
                                         ("Yield", "dividend_yield")])
    p, s = _assert_model("investment_portfolio", built, scores,
                         [("Return", "maximize", "avg"), ("Volatility", "minimize", "quadratic"),
                          ("Yield", "maximize", "avg")], "fund")
    _assert_matrix("investment_portfolio", "covariance.csv", s["interaction_matrices"][0]["entries"])
    assert _norm_adjustments(_scenario(p, "rate_cuts")["score_adjustments"]) == \
           _norm_adjustments([{"objective": "Yield", "multiply": 0.85}])
    assert _norm_adjustments(_scenario(p, "inflation")["score_adjustments"]) == \
           _norm_adjustments([{"objective": "Return", "multiply": 0.9},
                              {"objective": "Yield", "multiply": 1.15}])
    mo = _scenario(p, "recession")["interaction_matrix_overrides"][0]
    assert mo["objective"] == "Volatility"
    # Recession is a regime shift expressed as a scale factor on the US-equity block
    # (they co-move 1.5x more), not a replacement covariance matrix — so there is no
    # kit CSV for it. Assert the mechanism itself.
    assert mo["mode"] == "upsert" and not mo["entries"]
    assert [g["factor"] for g in mo["scale_groups"]] == [1.5]


def test_claims_triage_kit():
    rows = _rows("claims_investigation_triage")
    built = [{"type": "objective_bound", "objective": "Hours", "operator": "max", "value": 1170.0},
             {"type": "objective_bound", "objective": "ExpectedRecovery", "operator": "min", "value": 4840.0},
             {"type": "cardinality", "min": 45, "max": 100}]
    built += [{"type": "force_include", "option": r["claim"]} for r in rows if r["mandated_referral"] == "yes"]
    groups = _groups(rows, "claim", "line")
    for line, mx in {"AUT": 38, "PRP": 32, "LIA": 28, "WC": 26}.items():
        built.append({"type": "group_limit", "options": groups[line], "min": 0, "max": mx})
    scores = _scores_from(rows, "claim", [("ExpectedRecovery", "expected_recovery_kusd"),
                                          ("Hours", "hours"), ("Friction", "friction")])
    p, _ = _assert_model("claims_investigation_triage", built, scores,
                         [("ExpectedRecovery", "maximize", "sum"), ("Hours", "minimize", "sum"),
                          ("Friction", "minimize", "sum")], "claim")
    cut = [dict(c) for c in built]
    cut[0] = {"type": "objective_bound", "objective": "Hours", "operator": "max", "value": 1140.0}
    assert _canon(_scenario(p, "capacity_cut")["constraint_overrides"]) == _canon(cut)


def test_charging_siting_kit():
    rows = _rows("charging_network_siting")
    built = [{"type": "objective_bound", "objective": "Cost", "operator": "max", "value": 34.0},
             {"type": "cardinality", "min": 16, "max": 24}]
    built += [{"type": "force_include", "option": r["site"]} for r in rows if r["committed"] == "yes"]
    built += _exclusions(rows, "site", "mutually_exclusive_with")
    groups = _groups(rows, "site", "region")
    metros = {"HAR", "CED", "EAS", "BRK", "KNG", "NOR", "WES"}
    for region, opts in groups.items():
        if region in metros:
            built.append({"type": "group_limit", "options": opts, "min": 1, "max": 4})
        else:
            built.append({"type": "group_limit", "options": opts, "min": 0, "max": 5})
    scores = _scores_from(rows, "site", [("DriversServed", "drivers_served_kday"), ("Cost", "cost_musd")])
    # Overlap is quadratic: per-option scores still ship (diagonal display values)
    p, s = _bundle("charging_network_siting")
    overlap = {r["option"]: r["value"] for r in s["scores"] if r["objective"] == "Overlap"}
    scores += [{"option": k, "objective": "Overlap", "value": v} for k, v in overlap.items()]
    assert [(o["name"], o["direction"], o["aggregation"]) for o in p["objectives"]] == \
           [("DriversServed", "maximize", "sum"), ("Overlap", "minimize", "quadratic"),
            ("Cost", "minimize", "sum")]
    assert _canon(p["constraints"]) == _canon(built)
    non_overlap = [r for r in s["scores"] if r["objective"] != "Overlap"]
    assert _canon_scores(non_overlap) == _canon_scores(
        _scores_from(rows, "site", [("DriversServed", "drivers_served_kday"), ("Cost", "cost_musd")]))
    assert [o["name"] for o in s["options"]] == [r["site"] for r in rows]
    _assert_matrix("charging_network_siting", "catchment_overlap.csv",
                   s["interaction_matrices"][0]["entries"])
    surge = [{"option": r["site"], "objective": "DriversServed",
              "value": float(r["drivers_under_adoption_surge"])}
             for r in rows if r["drivers_under_adoption_surge"]]
    assert _canon_scores(_scenario(p, "adoption_surge")["score_overrides"]) == _canon_scores(surge)
    infl = [{"option": r["site"], "objective": "Cost", "value": float(r["cost_under_grid_inflation"])}
            for r in rows if r["cost_under_grid_inflation"]]
    assert _canon_scores(_scenario(p, "grid_cost_inflation")["score_overrides"]) == _canon_scores(infl)


def test_research_cohort_kit():
    rows = _rows("research_cohort_selection")
    built = [{"type": "cardinality", "min": 24, "max": 24}]
    built += [{"type": "force_exclude", "option": r["volunteer"]} for r in rows if r["screen_failed"] == "yes"]
    built += _exclusions(rows, "volunteer", "same_household_as")
    floors = {"A": 4, "B": 4, "C": 4, "D": 3, "E": 3, "F": 2}
    for stratum, opts in _groups(rows, "volunteer", "stratum").items():
        built.append({"type": "group_limit", "options": opts, "min": floors[stratum], "max": 8})
    for site, opts in _groups(rows, "volunteer", "site").items():
        built.append({"type": "group_limit", "options": opts, "min": 0, "max": 4})
    scores = _scores_from(rows, "volunteer", [("SignalStrength", "signal_strength"),
                                              ("RetentionRisk", "retention_risk"),
                                              ("CostPerParticipant", "cost_per_participant_kusd")])
    _assert_model("research_cohort_selection", built, scores,
                  [("SignalStrength", "maximize", "sum"), ("RetentionRisk", "minimize", "sum"),
                   ("CostPerParticipant", "minimize", "sum")], "volunteer")


def test_interconnection_kit():
    rows = _rows("interconnection_approvals")
    base = [{"type": "objective_bound", "objective": "Cost", "operator": "max", "value": 400.0}]
    base += [{"type": "dependency", "if_option": r["option"], "then_option": r["requires"]}
             for r in rows if r["requires"]]
    base += _exclusions(rows, "option", "mutually_exclusive_with")
    base += [{"type": "group_limit", "options": opts, "min": 0, "max": 9}
             for opts in _groups(rows, "option", "zone").values()]
    scores = _scores_from(rows, "option", [("NetValue", "net_value_musd"), ("Cost", "capex_musd"),
                                           ("ReliabilityRisk", "reliability_risk")])
    p, _ = _assert_model("interconnection_approvals", base, scores,
                         [("NetValue", "maximize", "sum"), ("Cost", "minimize", "sum"),
                          ("ReliabilityRisk", "minimize", "sum")], "option")
    for name, cap in [("capex_low", 320.0), ("capex_base", 400.0),
                      ("capex_high", 480.0), ("capex_stretch", 560.0)]:
        envelope = [dict(c) for c in base]
        envelope[0] = {"type": "objective_bound", "objective": "Cost", "operator": "max", "value": cap}
        assert _canon(_scenario(p, name)["constraint_overrides"]) == _canon(envelope), name


def test_shift_coverage_staffing_kit():
    rows = _rows("shift_coverage_staffing")
    built = [{"type": "objective_bound", "objective": "Cost", "operator": "max", "value": 210.0},
             {"type": "cardinality", "min": 60, "max": 80}]
    units = _groups(rows, "offer", "unit")
    floors = {"ICU": 12, "ED": 14, "MedSurg": 12, "Oncology": 8, "LD": 8}
    built += [{"type": "group_limit", "options": units[u], "min": floors[u], "max": len(units[u])}
              for u in floors]
    built += [{"type": "group_limit", "options": offs, "min": 0, "max": 3}
              for offs in _groups(rows, "offer", "nurse").values()]
    # same-nurse same-block offers exclude each other — derived from the nurse+block columns
    by_nb = defaultdict(list)
    for r in rows:
        by_nb[(r["nurse"], r["block"])].append(r["offer"])
    for offs in by_nb.values():
        for i in range(len(offs)):
            for j in range(i + 1, len(offs)):
                built.append({"type": "exclusion_pair", "option_a": offs[i], "option_b": offs[j]})
    scores = _scores_from(rows, "offer", [("CoverageValue", "coverage_value"),
                                          ("Cost", "cost_kusd"), ("FatigueRisk", "fatigue_risk")])
    _assert_model("shift_coverage_staffing", built, scores,
                  [("CoverageValue", "maximize", "sum"), ("Cost", "minimize", "sum"),
                   ("FatigueRisk", "minimize", "sum")], "offer")


def test_community_program_funding_kit():
    rows = _rows("community_program_funding")
    built = [{"type": "max_allocation", "max": 12},
             {"type": "objective_bound", "objective": "DeliveryRisk", "operator": "max", "value": 2.6}]
    built += [{"type": "allocation_bound", "option": r["program"],
               "min": int(r["mandated_floor_pct"]), "max": 100}
              for r in rows if r["mandated_floor_pct"]]
    scores = _scores_from(rows, "program", [("CommunityImpact", "impact_per_1pct"),
                                            ("ResidentsServed", "residents_k_per_1pct"),
                                            ("DeliveryRisk", "delivery_risk_per_1pct")])
    p, _ = _assert_model("community_program_funding", built, scores,
                         [("CommunityImpact", "maximize", "sum"), ("ResidentsServed", "maximize", "sum"),
                          ("DeliveryRisk", "minimize", "sum")], "program")
    assert _norm_adjustments(_scenario(p, "overrun_wave")["score_adjustments"]) == \
           _norm_adjustments([{"objective": "DeliveryRisk", "multiply": 1.25}])


# ─── kit-coverage guards ───

# One entry per kit test above. A bundled example missing here (or here without a
# bundle) fails test_every_bundled_example_has_a_kit, so a new example can't land
# without its reconstruction test and ask-literal guard.
KIT_COVERED = [
    "budget_allocation",
    "capacity_planning",
    "capital_project_selection_300",
    "channel_budget",
    "charging_network_siting",
    "claims_investigation_triage",
    "community_program_funding",
    "interconnection_approvals",
    "investment_portfolio",
    "production_mix",
    "research_cohort_selection",
    "scarce_supply_rationing",
    "shift_coverage_staffing",
    "supplier_selection",
]

# The load-bearing numbers each kit test hardcodes, as they read in the README's
# step-1 ask (commas stripped). The kit tests prove CSV+rules == canonical model;
# this proves the ask PROSE still states those rules, so the quoted ask can't
# drift from the model the test certifies.
ASK_LITERALS = {
    "budget_allocation": ["20%", "at most 3", "9 months", "roi_under_downturn"],
    "capacity_planning": ["25%", "0.20", "at or above 50", "15% higher",
                          "1.4x", "from 50 to 60"],
    "capital_project_selection_300": ["$1550M", "between 45 and 100", "20 Growth",
                                      "15 Digital", "15 R&D", "18 Maintenance",
                                      "12% over", "at most 70"],
    "channel_budget": ["15%", "at most 2 active", "2.0x", "20% lower"],
    "charging_network_siting": ["$34M", "between 16 and 24", "at most 4", "at most 5",
                                "NCR-01"],
    "claims_investigation_triage": ["1170 hours", "$4840k", "between 45 and 100",
                                    "38 Auto", "32 Property", "28 Liability",
                                    "26 Workers", "1140 hours"],
    "interconnection_approvals": ["$400M", "$320M", "$480M", "$560M",
                                  "9 approvals per zone"],
    "investment_portfolio": ["30%", "at most 3", "20%", "15% lower", "1.5x"],
    "production_mix": ["30%", "25%", "at most 2 active"],
    "research_cohort_selection": ["exactly 24", "at least 4", "at least 3",
                                  "at least 2", "no more than 8", "at most 4", "V-118"],
    "scarce_supply_rationing": ["8%", "at least 4.8", "8/7/5/5/4", "≥6%", "≥3%",
                                "≤4%", "~35%"],
    "supplier_selection": ["15%", "at most 3", "at or above 78", "at most 5%",
                           "from 15% to 10%", "at least 5%", "at least 4%",
                           "between 4% and 5%"],
    "shift_coverage_staffing": ["$210k", "between 60 and 80", "12 ICU", "14 ED",
                                "12 MedSurg", "8 Oncology", "8 LD", "more than 3"],
    "community_program_funding": ["12%", "2.6", "5%", "6%", "3% each", "25% higher"],
}


def test_every_bundled_example_has_a_kit():
    from engine import problem_io
    assert problem_io.list_available()["examples"] == KIT_COVERED
    assert sorted(ASK_LITERALS) == KIT_COVERED


def _ask(example):
    text = (EXAMPLES / example / "README.md").read_text()
    quoted = [ln.lstrip().lstrip(">").strip()
              for ln in text.splitlines() if ln.lstrip().startswith(">")]
    return " ".join(quoted).replace(",", "")


@pytest.mark.parametrize("example", sorted(ASK_LITERALS))
def test_ask_prose_states_the_load_bearing_numbers(example):
    ask = _ask(example).lower()
    for lit in ASK_LITERALS[example]:
        assert lit.replace(",", "").lower() in ask, \
            f"{example}: the step-1 ask no longer states {lit!r}"
