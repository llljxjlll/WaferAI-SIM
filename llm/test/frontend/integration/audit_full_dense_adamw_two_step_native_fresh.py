"""Reopen and cross-check two independently materialized Dense AdamW Fresh roots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_adamw_ir0 import build_full_dense_training_two_step_adamw_ir0
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data, physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest, RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.full_training_physical_dag import FullTrainingPhysicalDAG
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragments
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.program_io import ProgramHbmTarget, ProgramIoContract
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import load_json_dataclass, canonical_digest
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_source():
    plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
    graph = build_full_dense_training_two_step_adamw_ir0(plan)
    hardware = _hardware(1, 1)
    placed = place_train_forward_ir0(graph, PlacementContext.create(
        producer_pass="full_dense_training_two_step_adamw_native",
        fabric=physical_fabric_from_data(hardware),
        placement=plan.source_experiment.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(hardware)))
    return placed.replicas[0].graph


def _audit(root: Path, source, expected_commit: str) -> dict:
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    files = {
        "linked": root / "full_adamw_two_step.linked.json",
        "physical_dag": root / "full_adamw_two_step.physical_dag.json",
        "npup": root / "full_adamw_two_step.npup",
        "program_io": root / "full_adamw_two_step.program_io.json",
        "hardware": root / "hardware.json",
        "native_stdout": root / "npusim.stdout.txt",
    }
    if (evidence["source_commit"] != expected_commit
            or any(_sha(path) != evidence["files_sha256"][name]
                   for name, path in files.items())):
        raise RuntimeError(f"{root}: source commit or materialized bytes changed")
    manifest = load_json_dataclass(LinkedProgramManifest, files["linked"])
    physical = load_json_dataclass(FullTrainingPhysicalDAG, files["physical_dag"])
    io = load_json_dataclass(ProgramIoContract, files["program_io"])
    manifest.validate()
    physical.validate_against(
        _leaf_fragments(manifest.fragments), manifest.core_streams,
        required_operation_ids=tuple(sorted(node.id for node in source.nodes)))
    io.validate_against(manifest)
    anchors = {item.kind.value: item.artifact_id for item in manifest.input_digests
               if item.kind.value in ("ir1", "global_action_dag")}
    if (anchors.get("ir1") != source.id
            or physical.source_artifact_ids != tuple(sorted((
                source.id, anchors.get("global_action_dag", ""))))
            or io.program_artifact_sha256 != _sha(files["npup"])):
        raise RuntimeError(f"{root}: physical/program source digest or NPUP differs")
    leaves = _leaf_fragments(manifest.fragments)
    opcodes = [record.opcode for leaf in leaves for stream in leaf.core_streams
               for record in stream.records]
    wgrad = sum(opcode in (
        RecordOpcode.EMBEDDING_TABLE_WGRAD_TIMING,
        RecordOpcode.NORM_GAMMA_WGRAD_TIMING,
        RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING) for opcode in opcodes)
    physical_state_abis = {abi.id for leaf in leaves for abi in leaf.state_abi}
    hbm_seeds = [item.target.state_abi_id for item in io.initializations
                 if isinstance(item.target, ProgramHbmTarget)]
    source_wgrad = sum(node.kind in (
        OpKind.EMBEDDING_TABLE_WGRAD, OpKind.NORM_GAMMA_WGRAD,
        OpKind.GEMM_WEIGHT_WGRAD) for node in source.nodes)
    stdout = files["native_stdout"].read_text(encoding="utf-8")
    cycles = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout)
    if (len(manifest.fragments) != 520 or len(physical.actions) != 520
            or len(physical.state_version_edges) != 75
            or source_wgrad != wgrad or wgrad != 30
            or opcodes.count(RecordOpcode.ADAMW_UPDATE) != 30
            or len(physical_state_abis) != 75
            or len(hbm_seeds) != 75 or set(hbm_seeds) != physical_state_abis
            or len(io.initializations) != 433 or len(io.output_probes) != 62
            or stdout.count("[TRAIN_CE] core=") != 2
            or stdout.count("[TRAIN_CE_BACKWARD] core=") != 2
            or stdout.count("[TRAIN_ADAMW] core=") != 30
            or len(cycles) != 1 or int(cycles[0]) != evidence["makespan_cycles"]
            or "[DRAIN] router_residual=0" not in stdout
            or "[DRAIN] d2d_link_residual=0" not in stdout):
        raise RuntimeError(f"{root}: native gradients, StateABI versions, ProgramIO or drain differ")
    if (evidence["numeric_mode"] != "timing" or evidence["offload"] is not False):
        raise RuntimeError(f"{root}: evidence misstates AdamW numeric/offload status")
    return evidence


def audit(first: Path, second: Path, expected_commit: str) -> dict:
    if first.resolve() == second.resolve():
        raise RuntimeError("two Fresh roots must be distinct")
    source = _expected_source()
    a = _audit(first, source, expected_commit)
    b = _audit(second, source, expected_commit)
    stable = ("linked", "physical_dag", "npup", "program_io", "hardware",
              "npusim", "finalizer", "resolver", "simulation")
    if (any(a["files_sha256"][name] != b["files_sha256"][name] for name in stable)
            or a["makespan_cycles"] != b["makespan_cycles"]
            or a["files_sha256"]["native_stdout"] == b["files_sha256"]["native_stdout"]):
        raise RuntimeError("independent AdamW Fresh materializations are not repeatable")
    result = {"source_commit": expected_commit, "fresh_roots": [str(first), str(second)],
              "makespan_cycles": a["makespan_cycles"], "states": 75,
              "wgrad_paths": 30, "adamw_updates": 30,
              "linked_sha256": a["files_sha256"]["linked"],
              "npup_sha256": a["files_sha256"]["npup"],
              "program_io_sha256": a["files_sha256"]["program_io"],
              "physical_dag_sha256": a["files_sha256"]["physical_dag"],
              "native_stdout_sha256": [a["files_sha256"]["native_stdout"],
                                        b["files_sha256"]["native_stdout"]],
              "numeric_mode": "timing", "offload": False}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    audit(args.first, args.second, args.source_commit)


if __name__ == "__main__":
    main()
