import importlib.util
import json
import math
from pathlib import Path


EXP4 = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXP4 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bp = _load("build_pareto")
plot = _load("plot_results")


def _row(candidate, model, state, kind, shape, throughput, status="ok", evidence="cycle_anchor_calibrated_analytical_hardware_sweep"):
    row = {"candidate_id": candidate, "model_id": model, "software_state": state,
           "workload_type": kind, "throughput": throughput, "estimate_cycles": 1 / throughput,
           "status": status, "capacity_status": "feasible", "topology_status": "feasible",
           "evidence_level": evidence}
    if kind == "training":
        row["seq_len"] = shape
    elif kind == "request":
        row.update(batch_size=shape, seq_len=2304, output_tokens=512)
    return row


def test_aggregation_excludes_infeasible_and_computes_pareto_metrics():
    primitive, request = [], []
    # A is training-specialized, B inference-specialized, C balanced.
    values = {"A": (10, 2), "B": (2, 10), "C": (6, 6), "D": (1, 1)}
    for cid, (train, infer) in values.items():
        for seq in (2304, 36864):
            primitive.append(_row(cid, "m", "naive", "training", seq, train))
        for batch in (64, 512):
            request.append(_row(cid, "m", "naive", "request", batch, infer))
    # Despite huge numbers, an infeasible projection must not enter aggregation/Pareto.
    for seq in (2304, 36864):
        primitive.append(_row("X", "m", "naive", "training", seq, 100, "topology_capacity_infeasible"))
    for batch in (64, 512):
        request.append(_row("X", "m", "naive", "request", batch, 100, "capacity_infeasible_projection"))

    scores = bp.aggregate_scores(primitive, request)
    assert {r["candidate_id"] for r in scores} == {"A", "B", "C", "D"}
    group = bp.build_pareto_summary(scores)["groups"][0]
    assert [p["candidate_id"] for p in group["pareto"]] == ["B", "C", "A"]
    assert group["balanced_candidate"] == "C"
    assert group["pareto_point_count"] == 3
    assert math.isclose(group["G_split"], 2 / 3, rel_tol=1e-12)
    assert 0 < group["front_angle_degrees"] < 90
    assert group["split_recommended_5pct"] is True


def test_average_is_after_per_model_normalization_and_argmax_switches():
    rows = []
    for model, scale in (("m1", 1), ("m2", 100)):
        for state in ("naive", "sw_opt"):
            rows += [
                {"candidate_id": "A", "model_id": model, "software_state": state,
                 "training_score": 10 * scale, "inference_score": (2 if state == "naive" else 12) * scale,
                 "balanced_score": 1, "evidence": {"levels": ["analytical"]}},
                {"candidate_id": "B", "model_id": model, "software_state": state,
                 "training_score": 6 * scale, "inference_score": 10 * scale,
                 "balanced_score": 1, "evidence": {"levels": ["analytical"]}},
            ]
    summary = bp.build_pareto_summary(rows)
    avg = next(g for g in summary["average_groups"] if g["software_state"] == "naive")
    point_a = next(p for p in avg["points"] if p["candidate_id"] == "A")
    assert math.isclose(point_a["training_norm"], 1.0)
    switch = next(x for x in summary["argmax_switches"] if x["model_id"] == "m1")
    assert switch["inference_argmax_switched"] is True


def test_speedup_common_denominator_and_projection_marker():
    exp2 = [{"model_id": "llama2_7b", "seq_len": 2304, "T_base_cycles": 100,
             "T_full_train_overlap_cycles": 80},
            {"model_id": "llama2_7b", "seq_len": 36864, "T_base_cycles": 400,
             "T_full_train_overlap_cycles": 200}]
    primitive = []
    for seq, cycles in ((2304, 50), (36864, 100)):
        primitive += [
            {"candidate_id": "A", "model_id": "llama2_7b", "software_state": "naive",
             "workload_type": "training", "seq_len": seq, "estimate_cycles": cycles, "status": "ok"},
            {"candidate_id": "A", "model_id": "llama2_7b", "software_state": "sw_opt",
             "workload_type": "training", "seq_len": seq, "estimate_cycles": cycles / 2,
             "status": "capacity_infeasible_projection"},
        ]
    result = bp.build_speedup_summary(primitive, {"training": exp2})["rows"][0]
    assert math.isclose(result["sw_opt_only"], math.sqrt(2.5))
    assert math.isclose(result["hw_opt_only"], math.sqrt(8))
    assert math.isclose(result["sw_hw_opt"], math.sqrt(32))
    assert result["projection"]["sw_hw_opt"] is True


def test_svg_renderers_emit_both_figures(tmp_path):
    points = [{"candidate_id": "A", "training_norm": .5, "inference_norm": 1},
              {"candidate_id": "B", "training_norm": 1, "inference_norm": .5}]
    summary = {
        "groups": [{"model_id": "llama2_7b", "software_state": "naive", "pareto": points},
                   {"model_id": "llama2_7b", "software_state": "sw_opt", "pareto": points}],
        "average_groups": [],
        "speedup_figure": {"rows": [
            {"workload_type": kind, "model_id": "llama2_7b", "sw_opt_only": 1.1,
             "hw_opt_only": 1.2, "sw_hw_opt": 1.3, "projection": {"hw_opt_only": kind == "decode"}}
            for kind in ("training", "prefill", "decode")
        ]},
    }
    a, b = tmp_path / "a.svg", tmp_path / "b.svg"
    plot.render_speedup(summary, a)
    plot.render_pareto(summary, b)
    assert a.read_text().startswith("<svg") and "SW + HW" in a.read_text()
    assert b.read_text().startswith("<svg") and "geomean" in b.read_text()


def test_load_rows_accepts_wrapped_json(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"results": [{"candidate_id": "x"}]}))
    assert bp.load_rows(path) == [{"candidate_id": "x"}]
