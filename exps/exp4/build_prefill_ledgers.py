#!/usr/bin/env python3
"""Extract exact prefill+handoff resource ledgers from the frozen exp2 DAG."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
EXP2_ROOT = ROOT.parent / "exp2" / "exp2_1"
if str(EXP2_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP2_ROOT))

from e2e_replay import assert_same_work, build_inference_replay, replay  # noqa: E402
from model_manifests import load_model_manifests  # noqa: E402


PHASES = frozenset(("prefill", "handoff", "handoff_wait"))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def action_digest(actions: tuple[Any, ...]) -> str:
    return digest([action.manifest_dict() for action in actions])


def build() -> dict[str, Any]:
    published_path = EXP2_ROOT / "results" / "inference_prefill_pd_breakdown.json"
    published = json.loads(published_path.read_text(encoding="utf-8"))
    by_key = {(row["model_id"], int(row["seq_len"])): row for row in published}
    rows = []
    for model_id, manifest in load_model_manifests().items():
        for seq_len in (2304, 36864):
            pair = build_inference_replay(
                manifest, 64, prefill_seq=seq_len, kv_length=36864,
                routing_skew=1.0, include_shared_experts=False,
            )
            base_actions = tuple(action for action in pair.base_actions if action.phase in PHASES)
            opt_actions = tuple(action for action in pair.overlap_actions if action.phase in PHASES)
            assert_same_work(base_actions, opt_actions)
            base = replay(base_actions)
            opt = replay(opt_actions)
            source = by_key[(model_id, seq_len)]
            if base.makespan_cycles != float(source["T_base_cycles"]):
                raise AssertionError(f"base prefill mismatch: {model_id}/{seq_len}")
            if opt.makespan_cycles != float(source["T_overlap_cycles"]):
                raise AssertionError(f"optimized prefill mismatch: {model_id}/{seq_len}")
            if max(base.resource_service_cycles.values()) > base.makespan_cycles:
                raise AssertionError("base resource lower bound exceeds makespan")
            if max(opt.resource_service_cycles.values()) > opt.makespan_cycles:
                raise AssertionError("optimized resource lower bound exceeds makespan")
            row = {
                "case_id": source["case_id"], "model_id": model_id, "seq_len": seq_len,
                "phases": sorted(PHASES), "action_count": len(base_actions),
                "action_digest_base": action_digest(base_actions),
                "action_digest_sw_opt": action_digest(opt_actions),
                "same_work_invariant": True,
                "base_makespan_cycles": base.makespan_cycles,
                "sw_opt_makespan_cycles": opt.makespan_cycles,
                "base_theory_lower_cycles": base.theory_lower_cycles,
                "sw_opt_theory_lower_cycles": opt.theory_lower_cycles,
                "base_resource_service_cycles": dict(base.resource_service_cycles),
                "sw_opt_resource_service_cycles": dict(opt.resource_service_cycles),
                "source_result_digest": source["result_digest"],
            }
            row["ledger_digest"] = digest(row)
            rows.append(row)
    document = {
        "schema_version": "exp4.prefill_resource_ledgers.v1",
        "generator": "exp2 build_inference_replay filtered to prefill/handoff/handoff_wait",
        "exp2_source_sha256": hashlib.sha256(published_path.read_bytes()).hexdigest(),
        "row_count": len(rows), "rows": rows,
    }
    document["document_digest"] = digest(document)
    return document


def main() -> int:
    output = ROOT / "inputs" / "prefill_resource_ledgers.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    document = build()
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {document['row_count']} exact prefill ledgers: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
