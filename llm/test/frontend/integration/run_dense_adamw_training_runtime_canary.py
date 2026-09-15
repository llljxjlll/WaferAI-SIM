"""Execute two real H16/L2 Dense AdamW steps in one 1x1 timing NpuSim."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from llm.frontend.wafer_frontend.passes.dense_adamw_compile_sequence import (
    compile_dense_adamw_step,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides, build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode, SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.test.frontend.unit.test_dense_adamw_compile_sequence import _adamw_case

from .flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from .run_dense_training_sequence_runtime_canary import _run


_ROOT = Path(__file__).resolve().parents[4]


def _source_oracle(source, physical) -> None:
    physical_weights = {
        item.tensor_ref: item for item in physical.plan.parameter_templates
    }
    if len(physical_weights) != 15 or len(source.logical_graph.operations) == 0:
        raise RuntimeError("full two-layer Dense WGRAD production source is missing")
    logical_state = {
        state.id: state for state in source.logical_graph.state_versions
    }
    expected_ops: dict[int, set[str]] = {0: set(), 1: set()}
    operation_counts: dict[int, int] = {0: 0, 1: 0}
    for operation in source.logical_graph.operations:
        if operation.kind.value != "adamw_update":
            continue
        if operation.step not in expected_ops:
            raise RuntimeError("unexpected AdamW step source")
        operation_counts[operation.step] += 1
        expected_ops[operation.step].add(operation.parameter_ref)
        inputs = tuple(logical_state[ref] for ref in operation.reads)
        outputs = tuple(logical_state[ref] for ref in operation.writes)
        if (
            len(inputs) != 6 or len(outputs) != 5
            or any(item.version != operation.step for item in inputs)
            or any(item.version != operation.step + 1 for item in outputs)
        ):
            raise RuntimeError("AdamW logical six-read/five-write versions drifted")
    if (
        operation_counts != {0: 17, 1: 17}
        or len(expected_ops[0]) != 17
        or len(expected_ops[1]) != 17
        or expected_ops[0] != expected_ops[1]
    ):
        raise RuntimeError("two E2E steps must update the same 17 logical parameters")
    for layer in (0, 1):
        if {
            f"layer.{layer}.mlp_gate.weight",
            f"layer.{layer}.mlp_up.weight",
        }.difference(expected_ops[0]):
            raise RuntimeError("packed gate/up pair is missing from P3 AdamW source")


def _linked_oracle(linked, source, physical, index) -> tuple[int, int]:
    fragment = linked.manifest.fragments[0]
    records = fragment.core_streams[0].records
    if any(record.opcode is RecordOpcode.SGD_UPDATE for record in records):
        raise RuntimeError("AdamW linked step still contains an SGD opcode")
    counts = {
        opcode: sum(record.opcode is opcode for record in records)
        for opcode in (
            RecordOpcode.MATMUL, RecordOpcode.ADAMW_UPDATE,
            RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE,
        )
    }
    if (
        counts[RecordOpcode.MATMUL] != 41
        or counts[RecordOpcode.ADAMW_UPDATE] != 17
        or counts[RecordOpcode.LSU_LOAD] != 83
        or counts[RecordOpcode.LSU_STORE] != 83
    ):
        raise RuntimeError(f"AdamW physical WGRAD/update/83-state DMA changed: {counts}")
    operations = {
        item.id: item for item in source.logical_graph.operations
        if item.kind.value == "adamw_update" and item.step == index
    }
    updates = {
        record.source_global_action_id: (record_index, record)
        for record_index, record in enumerate(records)
        if record.opcode is RecordOpcode.ADAMW_UPDATE
    }
    if set(updates) != set(operations) or any(
        record.operands[17].literal_value != index + 1
        for _record_index, record in updates.values()
    ):
        raise RuntimeError("17 real AdamW source operations/step timing do not close")
    weight_refs = {
        item.state_ref for item in fragment.state_abi
        if item.kind is StateKind.TRAINABLE_PARAMETER
    }
    optimizer_refs = {
        item.state_ref for item in fragment.state_abi
        if item.kind is not StateKind.TRAINABLE_PARAMETER
    }
    if len(weight_refs) != 15 or len(optimizer_refs) != 68:
        raise RuntimeError("15 packed physical weights + 68 independent optimizer states required")
    state_bytes = sum(item.size_bytes for item in fragment.state_abi)
    if state_bytes != 32100:
        raise RuntimeError("whole-state HBM tight ABI changed")
    if (
        max(item.region_offset_bytes + item.size_bytes
            for item in fragment.buffer_abi) >= 65536
    ):
        raise RuntimeError("AdamW compute SRAM state address cannot fit uint16")
    for layer in (0, 1):
        names = (
            f"layer.{layer}.mlp_gate.weight",
            f"layer.{layer}.mlp_up.weight",
        )
        relocs = {
            (item.record_index, item.operand_id): item
            for item in fragment.core_streams[0].address_relocations
        }
        pair = sorted(
            (
                relocs[(updates[op_id][0], SemanticOperandId.COMPUTE_INPUT_ADDRESS)].addend,
                updates[op_id][1].operands[16].literal_value * 2,
                relocs[(updates[op_id][0], SemanticOperandId.COMPUTE_INPUT_ADDRESS)].symbol_ref,
            )
            for op_id, operation in operations.items()
            if operation.parameter_ref in names
        )
        if len(pair) != 2 or pair[0][:2] != (0, 32) or pair[1][:2] != (32, 32) or (
            pair[0][2] != pair[1][2]
        ):
            raise RuntimeError("gate/up 32+32 physical subviews overlap or omit bytes")
    return len(weight_refs), state_bytes


def observe_runtime(output: str, *, state_bytes: int) -> dict[str, object]:
    versions = re.findall(
        r"\[DENSE_TRAINING_SEQUENCE_STATE\] version=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) content_changed=(\d+) "
        r"functional=(\d+) pass=(\d+)",
        output,
    )
    updates = re.findall(
        r"\[DENSE_ADAMW_SEQUENCE_STEP\] index=(\d+) "
        r"input_version=(\d+) output_version=(\d+) "
        r"trainable_states=(\d+) optimizer_states=(\d+) "
        r"adamw_records=(\d+) load_records=(\d+) store_records=(\d+) "
        r"functional=(\d+) pass=(\d+)",
        output,
    )
    segments = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)",
        output,
    )
    probes = re.findall(
        r"\[DENSE_SEQUENCE_PROGRAM_IO\] index=(\d+) probes=(\d+) pass=(\d+)",
        output,
    )
    drain = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",
        output,
    )
    makespan = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", output)
    if (
        len(versions) != 3
        or tuple(int(item[0]) for item in versions) != (0, 1, 2)
        or any(int(item[1]) != state_bytes or item[3:] != ("0", "0", "1")
               for item in versions)
        or len(updates) != 2
        or tuple(item[:3] for item in updates) != (
            ("0", "0", "1"), ("1", "1", "2"),
        )
        or any(item[3:] != ("15", "68", "17", "83", "83", "0", "1")
               for item in updates)
        or segments != [("0", "0"), ("1", "1")]
        or probes != [("0", "83", "1"), ("1", "83", "1")]
        or drain != [("2", "1")]
        or len(makespan) != 1
        or output.count("[TRAIN_ADAMW]") != 34
    ):
        raise RuntimeError(
            "actual two-step AdamW HBM/version/update/ProgramIO/WorkerCore closure failed: "
            f"versions={versions} updates={updates} segments={segments} probes={probes} "
            f"ADAMW={output.count('[TRAIN_ADAMW]')} makespan={makespan} drain={drain}"
        )
    return {
        "versions": [0, 1, 2],
        "state_bytes": state_bytes,
        "state_digests": [item[2] for item in versions],
        "optimizer_states": 68,
        "adamw_updates": 34,
        "functional": False,
        "makespan_cycles": int(makespan[0]),
        "program_io_probes": 166,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source, physical = _adamw_case()
    _source_oracle(source, physical)
    linked = tuple(
        compile_dense_adamw_step(source, physical, step_index)
        for step_index in (0, 1)
    )
    state_count, state_bytes = _linked_oracle(linked[0], source, physical, 0)
    _linked_oracle(linked[1], source, physical, 1)
    if state_count != 15 or (
        tuple((abi.state_ref, abi.address, abi.size_bytes, abi.id)
              for abi in linked[0].manifest.fragments[0].state_abi)
        != tuple((abi.state_ref, abi.address, abi.size_bytes, abi.id)
                 for abi in linked[1].manifest.fragments[0].state_abi)
    ):
        raise RuntimeError("1→2 AdamW HBM address/state ABI continuity failed")
    seeds, expected = build_deterministic_timing_state_overrides(linked[0])
    step_refs = {
        abi.state_ref for abi in linked[0].manifest.fragments[0].state_abi
        if abi.kind is StateKind.OPTIMIZER_STEP
    }
    if len(seeds) != 83 or len(expected) != 68 or len(step_refs) != 17:
        raise RuntimeError("83 real state seed/68 optimizer probe coverage changed")
    for ref in step_refs:
        seeds[ref] = bytes(4)
        expected[ref] = bytes(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    programs = []
    sidecars = []
    for index, item in enumerate(linked):
        manifest = output / f"step_{index}.linked.json"
        program = output / f"step_{index}.npup"
        report_path = output / f"step_{index}.finalizer.json"
        manifest.write_text(canonical_json(item.manifest), encoding="utf-8")
        _run((
            str(args.finalizer.resolve()), "--input", str(manifest),
            "--output", str(program), "--report", str(report_path),
        ), cwd=output, timeout=120)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(program.read_bytes()).hexdigest()
        if (
            report["artifact_sha256"] != digest
            or report["linked_manifest_id"] != item.manifest.id
            or report["linked_manifest_digest"] != canonical_digest(item.manifest)
        ):
            raise RuntimeError(f"step {index} C++ finalizer digest closure failed")
        io = build_timing_program_io(
            item, digest, state_seed_overrides=seeds,
            state_expected_overrides=expected,
        )
        if len(io.output_probes) != 83:
            raise RuntimeError("83 tight HBM ABI output probes are required")
        sidecar = output / f"step_{index}.program_io.json"
        sidecar.write_text(canonical_json(io), encoding="utf-8")
        manifests.append(manifest)
        programs.append(program)
        sidecars.append(sidecar)
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 1 << 20
    hardware_path = output / "hardware.json"
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    mapping = output / "mapping.spec"
    mapping.write_text("0:0\n", encoding="utf-8")
    trace = _run((
        str(args.npusim.resolve()), "--program-sequence",
        ",".join(map(str, programs)),
        "--linked-manifest-sequence", ",".join(map(str, manifests)),
        "--program-io-sequence", ",".join(map(str, sidecars)),
        "--hardware-config", str(hardware_path),
        "--simulation-config", str(args.simulation.resolve()),
        "--mapping-config", str(mapping), "--trace-window", "1000000",
    ), cwd=output, timeout=args.timeout)
    (output / "npusim.stdout.txt").write_text(trace, encoding="utf-8")
    observation = observe_runtime(trace, state_bytes=state_bytes)
    (output / "adamw-runtime-evidence.json").write_text(
        json.dumps(observation, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Dense AdamW two-layer two-step NpuSim PASS "
        f"state_abi={state_bytes} versions=0,1,2 "
        f"adamw_updates=34 functional=0 "
        f"makespan={observation['makespan_cycles']}"
    )
    return observation


def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-debug-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=build / "dense-adamw-training-runtime-canary",
    )
    parser.add_argument(
        "--finalizer", type=Path,
        default=build / "npusim_program_finalizer",
    )
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument(
        "--simulation", type=Path,
        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    for name in ("finalizer", "npusim", "simulation"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name} must name an existing file")
    return args


if __name__ == "__main__":
    run(_parse_args())
