#!/usr/bin/env python3
"""Sensitivity analysis excluding structurally invalid topology candidates."""

from __future__ import annotations

import json

import build_corrected_pareto as corrected
import build_pareto as bp
import run_sensitivity as base
from run_experiment import _requests


def main() -> int:
    result_dir = corrected.ROOT / "results"
    all_primitive = json.loads((result_dir / "primitive_results.json").read_text(encoding="utf-8"))
    primitive = [row for row in all_primitive if corrected.structurally_valid(row)]
    requests = [row for row in _requests(primitive) if corrected.structurally_valid(row)]
    baseline = bp.build_pareto_summary(
        bp.aggregate_scores(primitive, requests, allow_projection=True)
    )
    baseline_fronts = base._fronts(baseline)
    variants = {}
    for name, field, factor in base.VARIANTS:
        changed = base.perturb(primitive, field, factor)
        changed_requests = [row for row in _requests(changed) if corrected.structurally_valid(row)]
        summary = bp.build_pareto_summary(
            bp.aggregate_scores(changed, changed_requests, allow_projection=True)
        )
        variants[name] = {"field": field, "factor": factor,
                          "fronts": base._fronts(summary),
                          "argmax_switches": summary["argmax_switches"]}
    fragile = {}
    for group, front in baseline_fronts.items():
        retained = set(front)
        for variant in variants.values():
            retained &= set(variant["fronts"].get(group, ()))
        fragile[group] = sorted(set(front) - retained)
    document = {
        "schema_version": "exp4.corrected_sensitivity.v2",
        "scope": "capacity-only performance projection; structural invalidity excluded",
        "baseline_fronts": baseline_fronts, "variants": variants,
        "fragile_pareto": fragile,
        "limitation": "HBM first-byte uses a documented 2% fixed-service attribution in the compact sensitivity layer.",
    }
    (result_dir / "sensitivity_summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(variants)} corrected one-factor sensitivity variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
