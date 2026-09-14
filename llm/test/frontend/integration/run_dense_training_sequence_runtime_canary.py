"""Run two Dense SGD steps in one 1x1 NpuSim/HBM instance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
)
from llm.test.frontend.unit.test_dense_training_compile_sequence import (
    _sequence,
)
from .flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)


_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class DenseTrainingSequenceRuntimeObservation:
    versions: tuple[int, ...]
    hbm_bytes: int
    hbm_digests: tuple[str, ...]
    step_count: int
    sgd_invocations: int
    makespan_cycles: int
    functional: bool = False


def _run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"returncode={completed.returncode}: {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def _records(linked):
    return tuple(
        record
        for item in linked.manifest.fragments
        for record in (
            item.fragment.core_streams[0].records
            if type(item) is RegionManifest
            else item.core_streams[0].records
        )
    )


def _validate_static_bindings(sequence) -> tuple[int, int]:
    linked = sequence.segments[0].linked_program
    records = _records(linked)
    by_action: dict[str, set[RecordOpcode]] = {}
    for record in records:
        by_action.setdefault(record.source_global_action_id, set()).add(
            record.opcode
        )
    wgrad_refs: set[str] = set()
    sgd_refs: set[str] = set()
    store_refs: set[str] = set()
    for binding in sequence.segments[0].parameter_bindings:
        for shard in binding.legacy_shards:
            wgrad_refs.update(shard.wgrad_action_refs)
            sgd_refs.update(shard.sgd_action_refs)
            store_refs.update(shard.store_action_refs)
    expected = (
        (wgrad_refs, RecordOpcode.MATMUL, "WGRAD"),
        (sgd_refs, RecordOpcode.SGD_UPDATE, "SGD"),
        (store_refs, RecordOpcode.LSU_STORE, "store"),
    )
    for refs, opcode, name in expected:
        if not refs or any(opcode not in by_action.get(ref, set()) for ref in refs):
            raise RuntimeError(f"static {name} action-to-record binding failed")
    if len(sgd_refs) != 15 or len(store_refs) != 15 or len(wgrad_refs) != 15:
        raise RuntimeError("1x1 sequence must bind 15 legacy parameter carriers")
    matmul_count = sum(record.opcode is RecordOpcode.MATMUL for record in records)
    return len(wgrad_refs), matmul_count


def observe_runtime(
    output: str,
    *,
    state_count: int,
    hbm_bytes: int,
    matmul_records: int,
) -> DenseTrainingSequenceRuntimeObservation:
    segments = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)",
        output,
    )
    states = re.findall(
        r"\[DENSE_TRAINING_SEQUENCE_STATE\] version=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) content_changed=(\d+) "
        r"functional=(\d+) pass=(\d+)",
        output,
    )
    steps = re.findall(
        r"\[DENSE_TRAINING_SEQUENCE_STEP\] index=(\d+) "
        r"input_version=(\d+) output_version=(\d+) "
        r"trainable_states=(\d+) matmul_records=(\d+) "
        r"sgd_records=(\d+) store_records=(\d+) functional=(\d+) pass=(\d+)",
        output,
    )
    drain = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",
        output,
    )
    makespan = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", output)
    if segments != [("0", "0"), ("1", "1")]:
        raise RuntimeError(f"two-step pause/drain closure failed: {segments}")
    if len(states) != 3:
        raise RuntimeError(f"requires state versions 0,1,2: {states}")
    if tuple(int(item[0]) for item in states) != (0, 1, 2):
        raise RuntimeError("state versions are not continuous")
    if any(
        int(item[1]) != hbm_bytes
        or item[4:] != ("0", "1")
        for item in states
    ):
        raise RuntimeError("HBM state marker bytes/functional/pass drifted")
    expected_steps = (
        ("0", "0", "1"),
        ("1", "1", "2"),
    )
    if tuple(item[:3] for item in steps) != expected_steps or any(
        int(item[3]) != state_count
        or int(item[4]) != matmul_records
        or int(item[5]) != state_count
        or int(item[6]) != state_count
        or item[7:] != ("0", "1")
        for item in steps
    ):
        raise RuntimeError(f"WGRAD/SGD/store step closure failed: {steps}")
    if drain != [("2", "1")] or len(makespan) != 1:
        raise RuntimeError("final one-shot drain or SIM_RESULT is not exact")
    sgd_invocations = output.count("[TRAIN_SGD]")
    if sgd_invocations != 2 * state_count:
        raise RuntimeError("runtime SGD invocation count is not exact")
    if output.count("[DENSE_SEQUENCE_PROGRAM_IO]") != 2:
        raise RuntimeError("ProgramIO did not verify after both steps")
    return DenseTrainingSequenceRuntimeObservation(
        versions=(0, 1, 2),
        hbm_bytes=hbm_bytes,
        hbm_digests=tuple(item[2] for item in states),
        step_count=2,
        sgd_invocations=sgd_invocations,
        makespan_cycles=int(makespan[0]),
    )


def run(args: argparse.Namespace) -> DenseTrainingSequenceRuntimeObservation:
    sequence = _sequence(1, 1)
    sequence.validate()
    state_count, matmul_records = _validate_static_bindings(sequence)
    linked = sequence.segments[0].linked_program
    state_seeds, state_expected = build_deterministic_timing_state_overrides(
        linked
    )
    if len(state_seeds) != state_count or state_expected:
        raise RuntimeError("deterministic trainable-state seed coverage changed")
    hbm_bytes = sum(map(len, state_seeds.values()))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []
    programs: list[Path] = []
    sidecars: list[Path] = []
    artifacts: list[str] = []
    for index, segment in enumerate(sequence.segments):
        manifest_path = output / f"step_{index}.linked.json"
        artifact_path = output / f"step_{index}.npup"
        report_path = output / f"step_{index}.finalizer.json"
        manifest_path.write_text(
            canonical_json(segment.linked_program.manifest), encoding="utf-8"
        )
        _run(
            (
                str(args.finalizer.resolve()),
                "--input",
                str(manifest_path),
                "--output",
                str(artifact_path),
                "--report",
                str(report_path),
            ),
            cwd=output,
            timeout=120,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if (
            report.get("artifact_sha256") != artifact_digest
            or report.get("linked_manifest_id")
            != segment.linked_program.manifest.id
            or report.get("linked_manifest_digest")
            != canonical_digest(segment.linked_program.manifest)
        ):
            raise RuntimeError(f"step {index} finalizer closure failed")
        contract = build_timing_program_io(
            linked,
            artifact_digest,
            state_seed_overrides=state_seeds,
        )
        contract.validate_against(linked.manifest)
        sidecar_path = output / f"step_{index}.program_io.json"
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        manifests.append(manifest_path)
        programs.append(artifact_path)
        sidecars.append(sidecar_path)
        artifacts.append(artifact_digest)

    hardware_path = output / "hardware.json"
    mapping_path = output / "mapping.spec"
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 1 << 20
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    mapping_path.write_text("0:0\n", encoding="utf-8")
    runtime_output = _run(
        (
            str(args.npusim.resolve()),
            "--program-sequence",
            ",".join(map(str, programs)),
            "--linked-manifest-sequence",
            ",".join(map(str, manifests)),
            "--program-io-sequence",
            ",".join(map(str, sidecars)),
            "--hardware-config",
            str(hardware_path),
            "--simulation-config",
            str(args.simulation.resolve()),
            "--mapping-config",
            str(mapping_path),
            "--trace-window",
            "1000000",
        ),
        cwd=output,
        timeout=args.timeout,
    )
    (output / "npusim.stdout.txt").write_text(
        runtime_output, encoding="utf-8"
    )
    observation = observe_runtime(
        runtime_output,
        state_count=state_count,
        hbm_bytes=hbm_bytes,
        matmul_records=matmul_records,
    )
    print(
        "Dense training sequence runtime canary PASS "
        f"sequence={sequence.digest} artifacts={','.join(artifacts)} "
        f"versions=0,1,2 hbm={','.join(observation.hbm_digests)} "
        "functional=0"
    )
    return observation


def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-debug-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=build / "dense-training-sequence-runtime-canary",
    )
    parser.add_argument(
        "--finalizer", type=Path, default=build / "npusim_program_finalizer"
    )
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument(
        "--simulation",
        type=Path,
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
