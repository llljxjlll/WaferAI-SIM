#!/usr/bin/env python3
"""One-factor analytical robustness sweep for the exp4 result set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from build_pareto import aggregate_scores, build_pareto_summary
from run_experiment import _requests


ROOT = Path(__file__).resolve().parent

VARIANTS = (
    ("dte_eff_0.8", "dte_service_cycles", 1 / .8),
    ("dte_eff_0.9", "dte_service_cycles", 1 / .9),
    ("d2d_eff_0.8", "d2d_service_cycles", 1 / .8),
    ("d2d_eff_0.9", "d2d_service_cycles", 1 / .9),
    ("hbm_util_0.8", "hbm_service_cycles", .9 / .8),
    ("hbm_util_0.95", "hbm_service_cycles", .9 / .95),
    # First-byte latency is not separable in the exp2 ledger. Conservatively
    # attribute 2% of HBM service to its fixed-latency component.
    ("hbm_first_byte_0.5x", "hbm_service_cycles", 1 - .02 * .5),
    ("hbm_first_byte_2x", "hbm_service_cycles", 1 + .02),
    ("stripe_imbalance_1.1x", "d2d_service_cycles", 1.1),
    ("stripe_imbalance_1.25x", "d2d_service_cycles", 1.25),
)


def perturb(rows: list[dict[str, Any]], field: str, factor: float) -> list[dict[str, Any]]:
    output = []
    for original in rows:
        row = dict(original)
        old_service = float(row[field])
        new_service = old_service * factor
        old_cycles = float(row["estimate_cycles"])
        lower = max(float(row["theory_lower_cycles"]), new_service)
        row["estimate_cycles"] = max(lower, old_cycles + new_service - old_service)
        tokens = int(row["batch_size"]) if row["workload_type"] == "decode" else int(row["batch_size"]) * int(row["seq_len"])
        row["throughput"] = tokens * 500_000_000 / row["estimate_cycles"]
        row["latency"] = row["estimate_cycles"] / 500_000_000
        output.append(row)
    return output


def _fronts(summary: dict[str, Any]) -> dict[str, list[str]]:
    return {f"{g['model_id']}::{g['software_state']}":
            [point["candidate_id"] for point in g["pareto"]]
            for g in summary["groups"] + summary["average_groups"]}


def main() -> int:
    result_dir = ROOT / "results"
    primitive = json.loads((result_dir / "primitive_results.json").read_text(encoding="utf-8"))
    baseline = build_pareto_summary(aggregate_scores(primitive, _requests(primitive), allow_projection=True))
    baseline_fronts = _fronts(baseline)
    variants: dict[str, Any] = {}
    for name, field, factor in VARIANTS:
        changed = perturb(primitive, field, factor)
        summary = build_pareto_summary(aggregate_scores(changed, _requests(changed), allow_projection=True))
        variants[name] = {"field": field, "factor": factor, "fronts": _fronts(summary),
                          "argmax_switches": summary["argmax_switches"]}
    fragile: dict[str, list[str]] = {}
    for group, base in baseline_fronts.items():
        retained = set(base)
        for variant in variants.values():
            retained &= set(variant["fronts"].get(group, ()))
        fragile[group] = sorted(set(base) - retained)
    document = {"schema_version": "exp4.sensitivity.v1",
                "scope": "capacity-infeasible performance-projection front",
                "baseline_fronts": baseline_fronts, "variants": variants,
                "fragile_pareto": fragile,
                "limitation": "HBM first-byte uses a documented 2% fixed-service attribution because exp2 did not publish a separable counter."}
    (result_dir / "sensitivity_summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(variants)} one-factor sensitivity variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
