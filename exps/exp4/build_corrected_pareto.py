#!/usr/bin/env python3
"""Build strict and capacity-projection Pareto summaries with structural gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_pareto as bp


ROOT = Path(__file__).resolve().parent
BAD_STRUCTURAL = ("topology_capacity_infeasible", "semantic_mismatch", "invalid", "failed", "error")


def structurally_valid(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(key, "")).lower()
                    for key in ("status", "topology_status"))
    return not any(word in text for word in BAD_STRUCTURAL)


def validate_shapes(primitive: list[dict[str, Any]], request: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, str], set[tuple[int, int, int]]] = {}
    for row in primitive:
        key = (row["candidate_id"], row["model_id"], row["software_state"], row["workload_type"])
        groups.setdefault(key, set()).add((int(row["seq_len"]), int(row["batch_size"]), int(row.get("kv_len") or 0)))
    expected = {"training": {(2304, 1, 0), (36864, 1, 0)},
                "prefill": {(2304, 1, 0), (36864, 1, 0)},
                "decode": {(1, 64, 36864), (1, 512, 36864)}}
    if any(shapes != expected[kind] for (*_, kind), shapes in groups.items()):
        raise ValueError("primitive shape completeness gate failed")
    request_groups: dict[tuple[str, str, str], set[tuple[int, int, int]]] = {}
    for row in request:
        key = (row["candidate_id"], row["model_id"], row["software_state"])
        request_groups.setdefault(key, set()).add((int(row["seq_len"]), int(row["batch_size"]), int(row["output_tokens"])))
    if any(shapes != {(2304, 64, 512), (2304, 512, 512)} for shapes in request_groups.values()):
        raise ValueError("request shape completeness gate failed")


def selected_candidates(primitive: list[dict[str, Any]], exp2: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for kind in ("training", "prefill", "decode"):
        for source in exp2[kind]:
            for state in ("naive", "sw_opt"):
                matches = [row for row in primitive if row["workload_type"] == kind
                           and row["model_id"] == source["model_id"]
                           and row["software_state"] == state
                           and ((kind != "decode" and int(row["seq_len"]) == int(source["seq_len"]))
                                or (kind == "decode" and int(row["batch_size"]) == int(source["batch_size"])))]
                feasible = [row for row in matches if bp.is_feasible(row)]
                pool = feasible or matches
                best = min(pool, key=lambda row: (float(row["estimate_cycles"]), row["candidate_id"]))
                output.append({"workload_type": kind, "model_id": source["model_id"],
                               "seq_len": source["seq_len"], "batch_size": source["batch_size"],
                               "software_state": state, "candidate_id": best["candidate_id"],
                               "capacity_projection": not bool(feasible),
                               "source_result_digest": source["result_digest"]})
    return output



def evaluated_candidate_scatter(
    primitive: list[dict[str, Any]], request: list[dict[str, Any]],
    projected: dict[str, Any], exp2: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Normalize both software states to the same H_exp2 + naive baseline."""
    models = sorted({row["model_id"] for row in exp2["training"]})
    baselines = {}
    for model in models:
        training = [
            float(row["seq_len"]) * float(row.get("batch_size", 1)) * 500_000_000
            / float(row["T_base_cycles"])
            for row in exp2["training"] if row["model_id"] == model
        ]
        inference = [
            float(row["batch_size"]) * float(row["output_tokens"]) * 500_000_000
            / float(row["T_base_cycles"])
            for row in exp2["request"] if row["model_id"] == model
        ]
        if len(training) != 2 or len(inference) != 2:
            raise ValueError(f"incomplete H_exp2 baseline for {model}")
        baselines[model] = {
            "training_score": bp.geometric_mean(training),
            "inference_score": bp.geometric_mean(inference),
        }

    all_scores = bp.aggregate_scores(primitive, request, allow_projection=True)
    valid_groups = {
        (group["model_id"], group["software_state"]): group
        for group in projected["groups"]
    }
    valid_candidate_ids = {
        row["candidate_id"] for row in primitive if structurally_valid(row)
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_scores:
        grouped.setdefault((row["model_id"], row["software_state"]), []).append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        baseline = baselines[key[0]]
        front_ids = {point["candidate_id"] for point in valid_groups[key]["pareto"]}
        points = [{
            "candidate_id": row["candidate_id"],
            "training_norm": float(row["training_score"]) / baseline["training_score"],
            "inference_norm": float(row["inference_score"]) / baseline["inference_score"],
            "structurally_valid": row["candidate_id"] in valid_candidate_ids,
            "pareto_member": row["candidate_id"] in front_ids,
        } for row in sorted(rows, key=lambda item: item["candidate_id"])]
        if len(points) != 383:
            raise ValueError(f"expected 383 evaluated candidates for {key}, got {len(points)}")
        output.append({"model_id": key[0], "software_state": key[1],
                       "candidate_count": len(points), "points": points})

    group_lookup = {(group["model_id"], group["software_state"]): group for group in output}
    average_lookup = {group["software_state"]: group for group in projected["average_groups"]}
    for state in ("naive", "sw_opt"):
        maps = {
            model: {point["candidate_id"]: point
                    for point in group_lookup[(model, state)]["points"]
                    if point["structurally_valid"]}
            for model in models
        }
        common = set.intersection(*(set(points) for points in maps.values()))
        front_ids = {point["candidate_id"] for point in average_lookup[state]["pareto"]}
        points = [{
            "candidate_id": candidate_id,
            "training_norm": bp.geometric_mean(
                maps[model][candidate_id]["training_norm"] for model in models
            ),
            "inference_norm": bp.geometric_mean(
                maps[model][candidate_id]["inference_norm"] for model in models
            ),
            "structurally_valid": True,
            "pareto_member": candidate_id in front_ids,
        } for candidate_id in sorted(common)]
        if len(points) != 366:
            raise ValueError("expected 366 valid candidates for six-model geomean")
        output.append({"model_id": "geomean_6_models", "software_state": state,
                       "candidate_count": len(points), "points": points})
    return output, baselines

def main() -> int:
    result_dir = ROOT / "results"
    primitive = bp.load_rows(result_dir / "primitive_results.json")
    request = bp.load_rows(result_dir / "request_results.json")
    validate_shapes(primitive, request)
    strict = bp.build_pareto_summary(bp.aggregate_scores(primitive, request))
    valid_primitive = [row for row in primitive if structurally_valid(row)]
    valid_request = [row for row in request if structurally_valid(row)]
    projected = bp.build_pareto_summary(
        bp.aggregate_scores(valid_primitive, valid_request, allow_projection=True)
    )
    for group in projected["groups"] + projected["average_groups"]:
        group["projection_only"] = True
        for point in group["points"]:
            point["evidence"]["all_feasible"] = False
    strict["projection_groups"] = projected["groups"]
    strict["projection_average_groups"] = projected["average_groups"]
    strict["projection_argmax_switches"] = projected["argmax_switches"]
    b512_feasible = sum(bp.is_feasible(row) for row in request if int(row["batch_size"]) == 512)
    strict["feasible_pareto_empty_reason"] = (
        f"B512 capacity-feasible request row count is {b512_feasible}"
        if not strict["groups"] else None
    )
    exp2_dir = ROOT.parent / "exp2/exp2_1/results"
    exp2 = {"training": bp.load_rows(exp2_dir / "training_e2e.json"),
            "prefill": bp.load_rows(exp2_dir / "inference_prefill_pd_breakdown.json"),
            "decode": bp.load_rows(exp2_dir / "inference_decode_e2e.json"),
            "request": bp.load_rows(exp2_dir / "inference_request_e2e.json")}
    strict["speedup_figure"] = bp.build_speedup_summary(valid_primitive, exp2)
    strict["per_shape_selected_candidates"] = selected_candidates(valid_primitive, exp2)
    scatter, baselines = evaluated_candidate_scatter(
        primitive, request, projected, exp2
    )
    strict["evaluated_candidate_scatter"] = scatter
    strict["pareto_normalization"] = {
        "baseline": "H_exp2+naive", "baseline_point": [1.0, 1.0], "per_model": baselines}
    strict["result_digest"] = bp._digest(strict)
    output = result_dir / "pareto_summary.json"
    output.write_text(json.dumps(strict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote corrected Pareto: strict_groups={len(strict['groups'])}, projection_groups={len(projected['groups'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
