#!/usr/bin/env python3
"""Build Exp-4 Pareto metrics and Figure-1 speedup data.

Only feasible hardware rows enter the Pareto set.  Infeasible analytical
projections may be retained in the speedup table, but are explicitly marked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


INVALID_WORDS = ("infeasible", "mismatch", "invalid", "failed", "error")
MODEL_ORDER = (
    "llama2_7b", "gpt3_175b", "llama3_8b", "llama3_1_405b",
    "mixtral_8x7b", "deepseek_v3",
)


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    for key in ("rows", "results", "data"):
        if isinstance(obj.get(key), list):
            return obj[key]
    raise ValueError(f"No row list in {path}")


def geometric_mean(values: Iterable[float]) -> float:
    xs = [float(v) for v in values]
    if not xs or any(not math.isfinite(v) or v <= 0 for v in xs):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(v) for v in xs) / len(xs))


def _number(row: dict[str, Any], name: str, default: float | None = None) -> float | None:
    value = row.get(name, default)
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def workload_type(row: dict[str, Any]) -> str:
    kind = str(row.get("workload_type") or row.get("workload_kind") or row.get("workload") or "").lower()
    if "request" in kind:
        return "request"
    if "prefill" in kind:
        return "prefill"
    if "decode" in kind:
        return "decode"
    if "train" in kind:
        return "training"
    return kind


def is_feasible(row: dict[str, Any]) -> bool:
    fields = ("status", "capacity_status", "topology_status")
    text = " ".join(str(row.get(k, "")).lower() for k in fields)
    return not any(word in text for word in INVALID_WORDS)


def row_throughput(row: dict[str, Any]) -> float | None:
    value = _number(row, "throughput")
    if value is not None and value > 0:
        return value
    cycles = _number(row, "estimate_cycles")
    if cycles is not None and cycles > 0:
        return 1.0 / cycles
    return None


def _evidence(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    levels = sorted({str(r.get("evidence_level", "unspecified")) for r in rows})
    return {
        "levels": levels,
        "all_feasible": all(is_feasible(r) for r in rows),
        "has_direct_anchor": "cycle_accurate_direct" in levels,
        "has_analytical_extrapolation": any("analytical" in x for x in levels),
    }


def aggregate_scores(
    primitive_rows: list[dict[str, Any]], request_rows: list[dict[str, Any]], *,
    allow_projection: bool = False,
) -> list[dict[str, Any]]:
    """Return one training/request score per candidate/model/software state."""
    train: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    request: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in primitive_rows:
        if workload_type(row) != "training" or (not allow_projection and not is_feasible(row)) or row_throughput(row) is None:
            continue
        train[(str(row["candidate_id"]), str(row["model_id"]), str(row["software_state"]))].append(row)
    for row in request_rows:
        if workload_type(row) != "request" or (not allow_projection and not is_feasible(row)) or row_throughput(row) is None:
            continue
        # The primary inference metric is S=2304, G=512, geometrically averaged
        # over B64/B512.  Missing optional fields are accepted for compact fixtures.
        if row.get("seq_len") not in (None, "", 2304, "2304"):
            continue
        if row.get("output_tokens") not in (None, "", 512, "512"):
            continue
        request[(str(row["candidate_id"]), str(row["model_id"]), str(row["software_state"]))].append(row)

    output = []
    for key in sorted(set(train) & set(request)):
        tr, ir = train[key], request[key]
        # Main contract requires two shapes in each aggregate.  De-duplicate by
        # the defining shape so reruns cannot silently change the score.
        tr_by_shape = _best_unique(tr, lambda r: str(r.get("seq_len", r.get("workload_id", ""))))
        ir_by_shape = _best_unique(ir, lambda r: str(r.get("batch_size", r.get("workload_id", ""))))
        if len(tr_by_shape) < 2 or len(ir_by_shape) < 2:
            continue
        t = geometric_mean(row_throughput(r) for r in tr_by_shape)
        i = geometric_mean(row_throughput(r) for r in ir_by_shape)
        output.append({
            "candidate_id": key[0], "model_id": key[1], "software_state": key[2],
            "training_score": t, "inference_score": i,
            "balanced_score": math.sqrt(t * i),
            "evidence": _evidence(tr_by_shape + ir_by_shape),
        })
    return output


def _best_unique(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        old = by_key.get(key)
        if old is None or (row_throughput(row) or 0) > (row_throughput(old) or 0):
            by_key[key] = row
    return [by_key[k] for k in sorted(by_key)]


def pareto_front(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Max/max non-dominated front; exact ties are retained and deterministically sorted."""
    front = []
    for p in points:
        dominated = any(
            (q["training_norm"] >= p["training_norm"] and q["inference_norm"] >= p["inference_norm"])
            and (q["training_norm"] > p["training_norm"] or q["inference_norm"] > p["inference_norm"])
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: (p["training_norm"], -p["inference_norm"], p["candidate_id"]))


def summarize_group(rows: list[dict[str, Any]], model_id: str, state: str) -> dict[str, Any]:
    tmax = max(r["training_score"] for r in rows)
    imax = max(r["inference_score"] for r in rows)
    points = []
    for row in rows:
        p = dict(row)
        p["training_norm"] = row["training_score"] / tmax
        p["inference_norm"] = row["inference_score"] / imax
        p["balanced_norm"] = math.sqrt(p["training_norm"] * p["inference_norm"])
        points.append(p)
    choose = lambda key: max(points, key=lambda p: (p[key], p["training_norm"] + p["inference_norm"], p["candidate_id"]))
    tstar, istar, balanced = choose("training_norm"), choose("inference_norm"), choose("balanced_norm")
    angle = abs(math.degrees(math.atan2(tstar["inference_norm"], tstar["training_norm"]) -
                             math.atan2(istar["inference_norm"], istar["training_norm"])))
    front = pareto_front(points)
    return {
        "model_id": model_id, "software_state": state,
        "candidate_count": len(points), "points": points, "pareto": front,
        "training_best_candidate": tstar["candidate_id"],
        "inference_best_candidate": istar["candidate_id"],
        "balanced_candidate": balanced["candidate_id"],
        "G_split": 0.5 * (1.0 / balanced["training_norm"] + 1.0 / balanced["inference_norm"]) - 1.0,
        "split_recommended_5pct": 0.5 * (1.0 / balanced["training_norm"] + 1.0 / balanced["inference_norm"]) - 1.0 >= .05,
        "pareto_point_count": len(front), "front_angle_degrees": angle,
    }


def build_pareto_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[(row["model_id"], row["software_state"])].append(row)
    groups = [summarize_group(rows, *key) for key, rows in sorted(grouped.items()) if rows]

    averages = []
    for state in sorted({g["software_state"] for g in groups}):
        state_groups = [g for g in groups if g["software_state"] == state]
        models = sorted(g["model_id"] for g in state_groups)
        per_model = {g["model_id"]: {p["candidate_id"]: p for p in g["points"]} for g in state_groups}
        common = set.intersection(*(set(x) for x in per_model.values())) if per_model else set()
        rows = []
        for cid in sorted(common):
            ts = [per_model[m][cid]["training_norm"] for m in models]
            ins = [per_model[m][cid]["inference_norm"] for m in models]
            rows.append({"candidate_id": cid, "model_id": "geomean_6_models", "software_state": state,
                         "training_score": geometric_mean(ts), "inference_score": geometric_mean(ins),
                         "balanced_score": geometric_mean(ts + ins),
                         "evidence": {"levels": sorted(set(sum((per_model[m][cid]["evidence"]["levels"] for m in models), []))),
                                      "all_feasible": True}})
        if rows:
            average = summarize_group(rows, "geomean_6_models", state)
            average["constituent_models"] = models
            averages.append(average)

    lookup = {(g["model_id"], g["software_state"]): g for g in groups + averages}
    switches = []
    for model in sorted({m for m, _ in lookup}):
        a, b = lookup.get((model, "naive")), lookup.get((model, "sw_opt"))
        if not a or not b:
            continue
        item = {"model_id": model}
        for objective, field in (("training", "training_best_candidate"),
                                 ("inference", "inference_best_candidate"),
                                 ("balanced", "balanced_candidate")):
            item[objective + "_naive"] = a[field]
            item[objective + "_sw_opt"] = b[field]
            item[objective + "_argmax_switched"] = a[field] != b[field]
        switches.append(item)
    return {"schema_version": "exp4_pareto_v1", "groups": groups,
            "average_groups": averages, "argmax_switches": switches}


def _exp2_key(row: dict[str, Any], kind: str) -> tuple[Any, ...]:
    if kind in ("training", "prefill"):
        return (row.get("model_id"), int(row.get("seq_len", 0)))
    if kind == "decode":
        return (row.get("model_id"), int(row.get("batch_size", 0)))
    return (row.get("model_id"), int(row.get("batch_size", 0)))


def build_speedup_summary(primitive: list[dict[str, Any]], exp2_by_kind: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Build Figure-1 values with H_exp2+naive as the common denominator."""
    out = []
    for kind in ("training", "prefill", "decode"):
        exp_rows = exp2_by_kind.get(kind, [])
        candidates = [r for r in primitive if workload_type(r) == kind]
        for model in MODEL_ORDER:
            base_rows = [r for r in exp_rows if r.get("model_id") == model]
            values = {"sw_opt_only": [], "hw_opt_only": [], "sw_hw_opt": []}
            projections = {k: False for k in values}
            for er in base_rows:
                key = _exp2_key(er, kind)
                base = _number(er, "T_base_cycles")
                opt_name = "T_full_train_overlap_cycles" if kind == "training" else "T_overlap_cycles"
                opt = _number(er, opt_name)
                if not base or not opt:
                    continue
                values["sw_opt_only"].append(base / opt)
                for state, name in (("naive", "hw_opt_only"), ("sw_opt", "sw_hw_opt")):
                    matching = [r for r in candidates if r.get("model_id") == model and
                                r.get("software_state") == state and _exp2_key(r, kind) == key and
                                (_number(r, "estimate_cycles") or 0) > 0]
                    feasible = [r for r in matching if is_feasible(r)]
                    pool = feasible or matching
                    if pool:
                        values[name].append(base / min(float(r["estimate_cycles"]) for r in pool))
                        projections[name] |= not bool(feasible)
            if not all(values.values()):
                continue
            out.append({"workload_type": kind, "model_id": model,
                        **{k: geometric_mean(v) for k, v in values.items()},
                        "projection": projections,
                        "sw_opt_evidence_level": "inherited_exp2_analytical_only_target_binding_unclosed"})
    return {"schema_version": "exp4_speedup_v1", "rows": out}


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--primitive", default=root / "results/primitive_results.json")
    ap.add_argument("--request", default=root / "results/request_results.json")
    ap.add_argument("--output", default=root / "results/pareto_summary.json")
    args = ap.parse_args(argv)
    primitive, request = load_rows(args.primitive), load_rows(args.request)
    summary = build_pareto_summary(aggregate_scores(primitive, request))
    projected = build_pareto_summary(
        aggregate_scores(primitive, request, allow_projection=True)
    )
    for group in projected["groups"] + projected["average_groups"]:
        group["projection_only"] = True
    summary["projection_groups"] = projected["groups"]
    summary["projection_average_groups"] = projected["average_groups"]
    summary["projection_argmax_switches"] = projected["argmax_switches"]
    summary["feasible_pareto_empty_reason"] = (
        "B512 request has no capacity-feasible 36-module candidate"
        if not summary["groups"] else None
    )
    exp2 = root.parent / "exp2/exp2_1/results"
    exp2_data = {
        "training": load_rows(exp2 / "training_e2e.json"),
        "prefill": load_rows(exp2 / "inference_prefill_pd_breakdown.json"),
        "decode": load_rows(exp2 / "inference_decode_e2e.json"),
    }
    summary["speedup_figure"] = build_speedup_summary(primitive, exp2_data)
    summary["input_digest"] = _digest({"primitive": primitive, "request": request})
    summary["result_digest"] = _digest(summary)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}: {len(summary['groups'])} model/state groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
