"""Real L2 Dense CE-forward/native seeded-backward *scoped* physical canary.

The older forward IR1/projection/schedule does not declare CE backward:
production .validate_against must reject.  This runner records that rejection
first, then attempts C++ finalization of the schema-valid physical carrier.
It cannot be used as a full-model training success oracle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.full_training_ce_tape_graft import (
    graft_seeded_ce_backward_onto_forward_tape,
)
from llm.frontend.wafer_frontend.lowering.full_training_ce_tape_program import (
    build_bounded_seeded_ce_physical_manifest,
)
from llm.frontend.wafer_frontend.passes.full_training_ce_seed_program_io import (
    build_bounded_seeded_ce_program_io,
)
from llm.frontend.wafer_frontend.passes.dense_training_ce_seeded_phase import (
    build_dense_training_ce_seeded_phase,
)
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.schema.artifact_manifest import LinkedProgramManifest
from llm.frontend.wafer_frontend.schema.train_n6 import TrainLinkedProgram
from llm.frontend.wafer_frontend.lowering.full_training_ce_tape_graft import (
    CeGraftedPhysicalForward,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.test.frontend.unit.test_dense_training_ce_seeded_phase import (
    DenseSeededCePhaseTest,
)
from llm.test.frontend.unit.test_full_training_timeline_linker import (
    FullTrainingTimelineLinkerTest,
)
from llm.test.frontend.integration.flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)


_SOURCE_GATE = "actions must exactly preserve projection DAG/task order"
_ROOT = Path(__file__).resolve().parents[4]
_SIM = _ROOT / "llm/test/program/p5_behavioral_simulation.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_SOURCE_FILES = (
    _ROOT / "llm/frontend/wafer_frontend/passes/dense_training_ce_backward.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/dense_training_ce_seeded_phase.py",
    _ROOT / "llm/frontend/wafer_frontend/lowering/full_training_ce_tape_graft.py",
    _ROOT / "llm/frontend/wafer_frontend/lowering/full_training_ce_tape_program.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/full_training_ce_seed_program_io.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/flexible_dense_train.py",
)


@dataclass(frozen=True, slots=True)
class BoundedSeededCeCase:
    manifest: LinkedProgramManifest
    profile: TrainLinkedProgram
    graft: CeGraftedPhysicalForward
    manifest_id: str
    manifest_json: str
    source_gate: str
    physical_records: int
    loss_gradient_seed: bytes


def build_case() -> BoundedSeededCeCase:
    fixture = DenseSeededCePhaseTest
    fixture.setUpClass()
    source = FullTrainingTimelineLinkerTest.forward.linked_forward.source.replicas[0]
    old_global = source.lowering_context.global_dag
    global_args = old_global._semantic_key()
    global_args["actions"] = old_global.actions + (fixture.backward,)
    new_global = GlobalActionDAG.create(
        producer_pass="bounded_dense_native_seeded_ce_global_dag", **global_args,
    )
    new_global.validate("bounded_ce_real_global")
    carrier = build_dense_training_ce_seeded_phase(
        fixture.forward, fixture.tape,
        backward_action=fixture.backward, source_global_dag_id=new_global.id,
        loss_gradient=fixture.loss_gradient,
        loss_gradient_address=fixture.loss_address,
        loss_gradient_label=fixture.loss_label,
        logits_gradient=fixture.logits_gradient,
        logits_gradient_address=fixture.grad_address,
        logits_gradient_label=fixture.grad_label,
        region=fixture.region,
    )
    graft = graft_seeded_ce_backward_onto_forward_tape(
        fixture.forward, fixture.tape, carrier,
        source_global_dag_id=new_global.id,
    )
    linked = build_bounded_seeded_ce_physical_manifest(
        fixture.forward, graft=graft, new_global_dag=new_global,
    )
    context = source.lowering_context
    try:
        linked.validate_against(
            context.ir1, context.fusion_plans, context.standalone_plans,
            context.projection, context.schedule_set,
            new_global, linked.fragments,
        )
    except SchemaError as error:
        if _SOURCE_GATE not in str(error):
            raise RuntimeError("bounded case was rejected by an unrelated source gate") from error
        source_gate = str(error)
    else:
        raise RuntimeError("old forward IR1/schedule unexpectedly accepted CE backward")
    return BoundedSeededCeCase(
        linked, FullTrainingTimelineLinkerTest.forward.linked_forward, graft,
        linked.id, canonical_json(linked), source_gate,
        sum(len(stream.records) for stream in linked.core_streams),
        graft.loss_gradient_seed,
    )


def run_finalizer(case: BoundedSeededCeCase, *,
                  executable: Path, output: Path,
                  resolver: Path | None = None,
                  npusim: Path | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    source = output / "bounded_seeded_ce.manifest.json"
    artifact = output / "bounded_seeded_ce.program.bin"
    report = output / "bounded_seeded_ce.finalizer.json"
    transcript = output / "bounded_seeded_ce.finalizer.stdout.txt"
    source.write_text(case.manifest_json, encoding="utf-8")
    finalization = subprocess.run([
        str(executable.resolve()), "--input", str(source.resolve()),
        "--output", str(artifact.resolve()), "--report", str(report.resolve()),
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120, check=False)
    transcript.write_text(finalization.stdout, encoding="utf-8")
    evidence = {
        "scope": "L2 Dense forward plus native seeded CE backward only",
        "deep_source_status": "REJECTED",
        "deep_source_reason": case.source_gate,
        "manifest_id": case.manifest_id,
        "physical_records": case.physical_records,
        "seed_bytes_hex": case.loss_gradient_seed.hex(),
        "manifest_sha256": hashlib.sha256(case.manifest_json.encode("utf-8")).hexdigest(),
        "source_sha256": {
            str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in _SOURCE_FILES
        },
        "finalizer_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "finalizer_exit": finalization.returncode,
        "finalizer_stdout_sha256": hashlib.sha256(
            finalization.stdout.encode("utf-8")).hexdigest(),
    }
    if finalization.returncode == 0:
        evidence["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        evidence["report"] = json.loads(report.read_text(encoding="utf-8"))
        if resolver is not None:
            contract = build_bounded_seeded_ce_program_io(
                case.profile, case.manifest, case.graft,
                evidence["artifact_sha256"],
            )
            sidecar = output / "bounded_seeded_ce.program_io.json"
            sidecar.write_text(canonical_json(contract), encoding="utf-8")
            resolved = subprocess.run([
                str(resolver.resolve()), "--resolve", str(source.resolve()),
                str(artifact.resolve()), str(sidecar.resolve()),
            ], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False, timeout=120)
            (output / "bounded_seeded_ce.program_io.stdout.txt").write_text(
                resolved.stdout, encoding="utf-8",
            )
            evidence["program_io_exit"] = resolved.returncode
            evidence["program_io_sha256"] = hashlib.sha256(
                resolver.read_bytes()).hexdigest()
            evidence["program_io_stdout_sha256"] = hashlib.sha256(
                resolved.stdout.encode("utf-8")).hexdigest()
            evidence["program_io_contract_id"] = contract.id
            evidence["program_io_initializations"] = len(contract.initializations)
            evidence["program_io_probes"] = len(contract.output_probes)
            evidence["program_io_dloss_fp32_ones_bytes"] = len(case.loss_gradient_seed)
            evidence["program_io_sidecar_sha256"] = hashlib.sha256(
                sidecar.read_bytes()).hexdigest()
            if resolved.returncode == 0 and npusim is not None:
                hardware = json.loads(specialize_p5_large_release_hardware(1, 2))
                sram = hardware["memory"]["sram"]
                hardware["memory"]["sram_size"] = 65536
                sram["capacity_bytes"] = 65536
                sram["regions"] = [{
                    "name": "sram", "base_bytes": 0, "size_bytes": 65536,
                    "allocator": "block", "spillable": False,
                    "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
                }]
                hw = output / "hardware.runtime.json"
                hw.write_text(json.dumps(hardware, sort_keys=True,
                                         separators=(",", ":")), encoding="utf-8")
                runtime = subprocess.run([
                    str(npusim.resolve()), "--program", str(artifact.resolve()),
                    "--linked-manifest", str(source.resolve()),
                    "--program-io", str(sidecar.resolve()),
                    "--hardware-config", str(hw.resolve()),
                    "--simulation-config", str(_SIM.resolve()),
                    "--mapping-config", str(_MAPPING.resolve()),
                    "--trace-window", "1000000",
                ], cwd=_ROOT / "llm", stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True,
                    check=False, timeout=600)
                stdout = runtime.stdout
                (output / "bounded_seeded_ce.npusim.stdout.txt").write_text(
                    stdout, encoding="utf-8",
                )
                ce_forward = re.findall(
                    r"\[TRAIN_CE\] core=(\d+) invocations=(\d+) rank_rows=(\d+)"
                    r" label_read_bytes=(\d+) loss_write_bytes=(\d+)", stdout,
                )
                ce_backward = re.findall(
                    r"\[TRAIN_CE_BACKWARD\] core=(\d+) invocations=(\d+)"
                    r" rank_rows=(\d+) upstream_elements=(\d+)"
                    r" logits_read_bytes=(\d+) label_read_bytes=(\d+)"
                    r" upstream_read_bytes=(\d+) logits_grad_write_bytes=(\d+)",
                    stdout,
                )
                phases = re.findall(
                    r"\[PROGRAM_IO\] phase=(\w+) mode=timing "
                    r"initializations=(\d+) probes=(\d+) "
                    r"checksum=([0-9a-f]{64}) pass=(\d+)", stdout,
                )
                probes = re.findall(
                    r"\[PROGRAM_IO_PROBE\].* address=(\d+) bytes=(\d+)"
                    r".* valid=(\d+) exact=(\d+) pass=(\d+)", stdout,
                )
                makespan = re.findall(
                    r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout,
                )
                memory = re.findall(
                    r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+)"
                    r" lsu_completed=(\d+) lsu_hbm_read_bytes=(\d+)"
                    r".* lsu_residual=(\d+) dte_residual=(\d+)", stdout,
                )
                drains = (
                    "[DRAIN] router_residual=0" in stdout
                    and "[DRAIN] d2d_link_residual=0" in stdout
                )
                passed = (
                    runtime.returncode == 0
                    and ce_forward == [("0", "1", "4", "16", "16")]
                    and ce_backward ==
                    [("0", "1", "4", "4", "128", "16", "16", "128")]
                    and [x[0] for x in phases] ==
                    ["resolved", "applied", "verify"]
                    and all((x[1], x[2], x[4]) == ("59", "1", "1")
                            for x in phases)
                    and probes == [("3520", "16", "1", "1", "1")]
                    and len(makespan) == 1 and int(makespan[0]) > 0
                    and memory == [("0", "15", "15", "936", "0", "0")]
                    and drains
                )
                evidence.update({
                    "npusim_exit": runtime.returncode,
                    "npusim_sha256": hashlib.sha256(
                        npusim.read_bytes()).hexdigest(),
                    "npusim_stdout_sha256": hashlib.sha256(
                        stdout.encode("utf-8")).hexdigest(),
                    "hardware_sha256": hashlib.sha256(hw.read_bytes()).hexdigest(),
                    "simulation_sha256": hashlib.sha256(_SIM.read_bytes()).hexdigest(),
                    "mapping_sha256": hashlib.sha256(_MAPPING.read_bytes()).hexdigest(),
                    "runtime_root": str(_ROOT / "llm"),
                    "ce_forward": ce_forward,
                    "ce_backward": ce_backward,
                    "program_io_phases": phases,
                    "loss_probe_timing_only": probes,
                    "memory": memory,
                    "makespan_cycles": int(makespan[0]) if makespan else None,
                    "drain_zero": drains,
                    "scoped_runtime_status": "PASS" if passed else "FAIL",
                    "functional_logits_gradient": False,
                })
    (output / "bounded_seeded_ce.evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path)
    parser.add_argument("--npusim", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 3:
        parser.error("--repeat must be 1..3 independent complete materializations")
    if args.npusim is not None and args.resolver is None:
        parser.error("NpuSim requires the production ProgramIO resolver")
    trials = []
    for i in range(args.repeat):
        case = build_case()
        evidence = run_finalizer(
            case, executable=args.finalizer,
            output=args.output / f"trial{i}",
            resolver=args.resolver, npusim=args.npusim,
        )
        trials.append(evidence)
    stable_fields = (
        "manifest_sha256", "artifact_sha256", "finalizer_sha256",
        "program_io_sidecar_sha256", "program_io_sha256", "npusim_sha256",
        "hardware_sha256", "ce_forward", "ce_backward",
        "program_io_phases", "loss_probe_timing_only", "memory",
        "makespan_cycles", "drain_zero", "source_sha256",
    )
    stable = all(trial.get(field) == trials[0].get(field)
                 for trial in trials[1:] for field in stable_fields)
    result = {
        "scope": "two-layer Dense CE forward/native seeded backward timing only",
        "deep_full_train": "REJECTED_SOURCE_TASK_MISSING",
        "repeat": args.repeat,
        "stable_markers_and_digests": stable,
        "trial_statuses": [trial.get("scoped_runtime_status", "NOT_RUN")
                           for trial in trials],
        "artifact_sha256": trials[0].get("artifact_sha256"),
        "source_gate": trials[0]["deep_source_reason"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "repeat.evidence.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return (0 if stable and all(trial.get("scoped_runtime_status") == "PASS"
                                for trial in trials) else 2)


if __name__ == "__main__":
    raise SystemExit(main())
