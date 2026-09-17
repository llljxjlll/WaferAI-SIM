"""Collect measured Dense mesh costs from already audited native Fresh cases.

This is a performance report, not a release completion receipt. Missing cases
remain explicit and never count as verified executions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .run_dense_native_mesh_matrix import audit_cached_case


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(roots: tuple[Path, ...]) -> dict[str, object]:
    cases: dict[str, object] = {}
    missing: list[str] = []
    bindings: list[dict[str, object]] = []
    profiles: set[str] = set()
    for root in roots:
        root = root.resolve()
        binding_path = root / "matrix_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        profile = binding.get("profile", "mixed")
        if profile not in ("mixed", "all_dies_scaled"):
            raise ValueError(f"unknown matrix profile in {root}")
        profiles.add(profile)
        bindings.append({"root": str(root), "sha256": _sha(binding_path),
                         "profile": profile, "shapes": binding["shapes"]})
        for shape in binding["shapes"]:
            if shape in cases or shape in missing:
                raise ValueError(f"duplicate shape across cost roots: {shape}")
            case = root / shape
            if not (case / "case_evidence.json").is_file():
                missing.append(shape)
                continue
            evidence = audit_cached_case(case, shape,
                                         all_dies_scaled=profile == "all_dies_scaled")
            observations = evidence["fresh"]
            samples = []
            for index, observation in enumerate(observations):
                fresh = case / f"fresh{index}"
                receipt = json.loads((fresh / "compiled_receipt.json").read_text(
                    encoding="utf-8"))
                stages = receipt["segment_metrics"]
                if receipt["runtime_status"] != "verified" or len(stages) != 3:
                    raise ValueError(f"incomplete cost receipt: {fresh}")
                samples.append({
                    "makespan_cycles": observation["makespan_cycles"],
                    "total_wall_seconds": receipt["total_wall_seconds"],
                    "production_compile_wall_seconds": receipt["phase_wall_seconds"]["production_compile"],
                    "native_npusim_wall_seconds": receipt["phase_wall_seconds"]["native_npusim"],
                    "frontend_peak_rss_kib": receipt["frontend_peak_rss_kib"],
                    "children_max_rss_kib": receipt["children_max_rss_kib"],
                    "linked_bytes": sum(item["linked_manifest_bytes"] for item in stages),
                    "npup_bytes": sum(item["npup_bytes"] for item in stages),
                    "program_io_bytes": sum(item["program_io_bytes"] for item in stages),
                    "finalizer_wall_seconds": round(sum(item["finalizer_wall_seconds"] for item in stages), 3),
                    "program_io_wall_seconds": round(sum(item["program_io_wall_seconds"] for item in stages), 3),
                    "resolver_wall_seconds": round(sum(item["resolver_wall_seconds"] for item in stages), 3),
                })
            cases[shape] = {
                "active_die_ids": observations[0]["active_die_ids"],
                "case_evidence_sha256": _sha(case / "case_evidence.json"),
                "fresh": samples,
            }
    return {
        "schema_version": "dense-native-mesh-measured-costs-v1",
        "status": "measured_partial" if missing else "measured_complete_for_supplied_bindings",
        "release_completion_claim": False,
        "profiles": sorted(profiles),
        "bindings": bindings,
        "measured_case_count": len(cases),
        "measured_fresh_count": 2 * len(cases),
        "missing_shapes": missing,
        "cases": dict(sorted(cases.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("cost report output already exists")
    report = collect(tuple(args.roots))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"Dense measured costs {report['measured_case_count']} cases / "
          f"{report['measured_fresh_count']} Fresh; "
          f"{len(report['missing_shapes'])} missing")


if __name__ == "__main__":
    main()
