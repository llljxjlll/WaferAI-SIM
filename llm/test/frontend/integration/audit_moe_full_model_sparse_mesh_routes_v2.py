"""Independently reopen every signed remote flow's sparse-mesh X-first path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run_moe_full_model_sparse_mesh_v2 as sparse


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected object: {path}")
    return value


def audit(root: Path) -> dict:
    run_binding = _read(root / "v2_binding.json")
    physical_shape = run_binding["physical_mesh"]
    logical_shape = run_binding["logical_mesh"]
    if sparse._PAIRS.get(physical_shape) != logical_shape:
        raise ValueError("unsupported logical-to-physical mesh binding")
    canonical_source = sparse._physical_source(logical_shape, physical_shape)
    lr, lc = sparse._size(logical_shape)
    pr, pc = sparse._size(physical_shape)
    fresh = []
    for index in range(2):
        directory = root / f"fresh{index}"
        physical_source = _read(directory / "physical_source.json")
        evidence = _read(directory / "v2_evidence.json")
        flow_binding_path = directory / "compiled" / "source_tool_binding.json"
        flow_binding = _read(flow_binding_path)
        if physical_source != canonical_source or evidence["source"] != physical_source:
            raise ValueError("physical request or manifest no longer materializes exactly")
        flows = flow_binding["expected_remote_flows"]
        logical_links, _ = sparse.canonical._recompute_flow_evidence(
            flows, rows=lr, columns=lc,
        )
        if logical_links != flow_binding["expected_d2d_links"]:
            raise ValueError("logical signed directed links drifted")
        per_flow = []
        for flow in flows:
            logical_path, _ = sparse.canonical._recompute_flow_evidence(
                [flow], rows=lr, columns=lc,
            )
            physical_path, _ = sparse.canonical._recompute_flow_evidence(
                [flow], rows=pr, columns=pc,
            )
            if logical_path != physical_path:
                raise ValueError(f"X-first Die/route identity changed: {flow['id']}")
            per_flow.append(logical_path)
        physical_links, _ = sparse.canonical._recompute_flow_evidence(
            flows, rows=pr, columns=pc,
        )
        if physical_links != evidence["d2d_links"]:
            raise ValueError("physical native directed links differ from each signed path")
        if (evidence["physical_die_count"] != pr * pc
                or physical_source["active_die_ids"] != list(range(lr * lc))
                or physical_source["idle_die_ids"] != list(range(lr * lc, pr * pc))):
            raise ValueError("active or idle Die identity drifted")
        fresh.append({
            "index": index,
            "flow_binding_sha256": _sha(flow_binding_path),
            "physical_source_sha256": _sha(directory / "physical_source.json"),
            "v2_evidence_sha256": _sha(directory / "v2_evidence.json"),
            "signed_flow_count": len(flows),
            "endpoints": sorted({rank for flow in flows for rank in (
                flow["source_rank"], flow["destination_rank"]
            )}),
            "per_flow_paths_sha256": hashlib.sha256(json.dumps(
                per_flow, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
        })
    if (fresh[0]["flow_binding_sha256"] != fresh[1]["flow_binding_sha256"]
            or fresh[0]["per_flow_paths_sha256"] != fresh[1]["per_flow_paths_sha256"]):
        raise ValueError("Fresh route or source binding differs")
    return {
        "schema_version": "moe-sparse-v2-per-flow-route-audit",
        "status": "verified",
        "physical_mesh": physical_shape,
        "logical_mesh": logical_shape,
        "physical_die_count": pr * pc,
        "active_die_count": lr * lc,
        "idle_die_count": pr * pc - lr * lc,
        "v2_case_evidence_sha256": _sha(root / "v2_case_evidence.json"),
        "fresh": fresh,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
