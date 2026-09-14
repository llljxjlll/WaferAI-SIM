#!/usr/bin/env python3
"""Create the frozen 168-run calibration design and audit executable evidence.

This driver never labels an unexecuted analytical point as cycle accurate.  In
the current repository the exp2 direct target-hardware closure is unresolved,
so the design is emitted for reproducibility and the sweep inherits an honest
analytical-only evidence level until current-build logs are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from candidate_loader import load_candidates


MOTIFS = ("isolated_gemm", "sram_read_write", "single_channel_dte",
          "d2d_1_2_multihop", "hbm_read_write", "collective_broadcast_reduce")
WINDOWS = ("dense_train_short", "dense_prefill_long", "dense_decode_B512",
           "gqa_decode_B64", "moe_train_long", "moe_decode_B512")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _features(c: Any) -> tuple[float, ...]:
    router = {"base": 0.0, "broadcast": 1 / 3, "reduce": 2 / 3, "both": 1.0}[c.router]
    return (math.log2(c.N_PE), c.B_GBs, c.B_s_GBs, c.K_MiB, c.N,
            c.HBM_stack_count, c.t_hbm, c.d_d2d, router, c.n_ctrl)


def farthest_points(candidates: list[Any], count: int) -> list[Any]:
    raw = [_features(c) for c in candidates]
    dims = list(zip(*raw))
    normalized = [tuple(0.0 if max(col) == min(col) else (x - min(col)) / (max(col) - min(col))
                        for x, col in zip(row, dims)) for row in raw]
    chosen = [min(range(len(candidates)), key=lambda i: candidates[i].candidate_id)]
    while len(chosen) < count:
        def distance(i: int) -> tuple[float, str]:
            nearest = min(sum((a-b)**2 for a, b in zip(normalized[i], normalized[j])) for j in chosen)
            return nearest, candidates[i].candidate_id
        chosen.append(max((i for i in range(len(candidates)) if i not in chosen), key=distance))
    return [candidates[i] for i in chosen]


def build_design() -> list[dict[str, Any]]:
    selected = farthest_points(load_candidates(), 11)
    rows: list[dict[str, Any]] = []
    for c in selected[:8]:
        for motif in MOTIFS:
            for repeat in (1, 2):
                rows.append({"anchor_id": f"fit-{c.candidate_id}-{motif}-r{repeat}",
                             "split": "fit", "candidate_id": c.candidate_id,
                             "workload": motif, "software_state": "motif", "repeat": repeat,
                             "execution_status": "planned_not_executed"})
    for c in selected[8:]:
        for window in WINDOWS:
            for state in ("naive", "sw_opt"):
                for repeat in (1, 2):
                    rows.append({"anchor_id": f"holdout-{c.candidate_id}-{window}-{state}-r{repeat}",
                                 "split": "holdout", "candidate_id": c.candidate_id,
                                 "workload": window, "software_state": state, "repeat": repeat,
                                 "execution_status": "planned_not_executed"})
    assert len(rows) == 168
    return rows


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=root / "calibration")
    args = ap.parse_args(argv)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    design = build_design()
    design_doc = {"schema_version": "exp4.anchor_design.v1", "planned_run_count": len(design),
                  "completed_run_count": 0, "runs": design}
    design_doc["design_digest"] = digest(design_doc)
    exp2_summary_path = root.parent / "exp2/exp2_1/results/calibration_summary.json"
    exp2_summary = json.loads(exp2_summary_path.read_text(encoding="utf-8"))
    summary = {
        "schema_version": "exp4.calibration_summary.v1",
        "planned_run_count": 168, "completed_run_count": 0,
        "fit_motif_run_count": 96, "holdout_window_run_count": 72,
        "current_build_direct_gate_passed": False,
        "publish_status": "analytical_only_current_build_anchor_pending",
        "evidence_level": "resource_explicit_analytical_extrapolation",
        "reason": "No 168 current-build direct shadow-run logs are present; exp2 source is inherited only as structural evidence.",
        "inherited_exp2_publish_status": exp2_summary.get("publish_status"),
        "inherited_exp2_calibration_digest": exp2_summary.get("calibration_summary_digest"),
        "anchor_design_digest": design_doc["design_digest"],
    }
    summary["calibration_digest"] = digest(summary)
    (output / "anchor_design.json").write_text(json.dumps(design_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "calibration_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned {len(design)} anchors; completed 0; status={summary['publish_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
