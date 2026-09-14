#!/usr/bin/env python3
"""Corrected exp4 driver using Action FLOP/byte profiles for every workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import action_replay
import run_experiment as driver


ROOT = Path(__file__).resolve().parent


def _source_rows(_exp2_root: Path) -> dict[str, dict]:
    result_dir = ROOT.parent / "exp2/exp2_1/results"
    rows = {}
    for name in ("training_e2e.json", "inference_prefill_pd_breakdown.json",
                 "inference_decode_e2e.json"):
        for row in json.loads((result_dir / name).read_text(encoding="utf-8")):
            rows[row["case_id"]] = row
    if len(rows) != 36:
        raise ValueError("expected 36 frozen exp2 primitive rows")
    return rows


def _estimate(source: dict, kind: str, state: str, candidate: object) -> SimpleNamespace:
    value = action_replay.estimate(source, kind, state, candidate)
    return SimpleNamespace(
        estimate_cycles=value.estimate_cycles,
        theory_lower_cycles=value.theory_lower_cycles,
        attainment=value.attainment,
        critical_resource=value.critical_resource,
        class_service_cycles=value.class_service_cycles,
        resource_service_cycles=value.resource_service_cycles,
        calibration_factor=(1.0 / value.attainment),
        result_digest=value.result_digest,
    )


def main(argv: list[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", default=ROOT / "results")
    known, _ = parser.parse_known_args(args_in)
    audit = action_replay.audit_reference_reproduction()
    if audit["failure_count"]:
        raise SystemExit(f"reference reproduction gate failed: {audit['failures']}")
    driver.estimate = _estimate
    driver.load_exp2_resource_rows = _source_rows
    status = driver.main(args_in)
    if status:
        return status
    output = Path(known.output_dir)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles = json.loads((ROOT / "inputs/action_profiles.json").read_text(encoding="utf-8"))
    manifest.update({
        "schema_version": "exp4.corrected_action_replay_run.v2",
        "analytical_model": "action_flops_bytes_dependency_resource_lower_bound",
        "action_profile_digest": profiles["document_digest"],
        "reference_reproduction_audit": audit,
        "invalidated_model": "duration-ledger-scaling-v1",
    })
    manifest.pop("run_digest", None)
    manifest["run_digest"] = driver.canonical_digest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"corrected action replay gate passed: {audit['row_count']} reference rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
