"""Strict four-Die SGD static carriers and same-instance two-step markers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode, RegionManifest


@dataclass(frozen=True, slots=True)
class Tp4TrainingObservation:
    versions: tuple[int, ...]
    hbm_digests: tuple[str, ...]
    sgd_invocations: int
    makespan_cycles: int


def _records(linked):
    return tuple(
        record
        for fragment in linked.manifest.fragments
        for stream in (
            fragment.fragment.core_streams
            if isinstance(fragment, RegionManifest) else fragment.core_streams
        )
        for record in stream.records
    )


def _validate_static_bindings(sequence) -> tuple[int, int]:
    sequence.validate()
    linked = sequence.segments[0].linked_program
    records = _records(linked)
    by_action: dict[str, set[RecordOpcode]] = {}
    for record in records:
        by_action.setdefault(record.source_global_action_id, set()).add(record.opcode)
    carriers = (
        ("WGRAD", "wgrad_action_refs", RecordOpcode.MATMUL),
        ("SGD", "sgd_action_refs", RecordOpcode.SGD_UPDATE),
        ("store", "store_action_refs", RecordOpcode.LSU_STORE),
    )
    counts = []
    for name, field, opcode in carriers:
        refs = {
            ref
            for binding in sequence.segments[0].parameter_bindings
            for shard in binding.legacy_shards
            for ref in getattr(shard, field)
        }
        if not refs or any(opcode not in by_action.get(ref, set()) for ref in refs):
            raise RuntimeError(f"physical TP4 {name} carrier did not lower to signed record")
        counts.append(len(refs))
    if len(set(counts)) != 1 or len(records) != linked.record_count:
        raise RuntimeError("TP4 WGRAD/SGD/store or physical core record coverage drifted")
    return counts[0], sum(item.opcode is RecordOpcode.MATMUL for item in records)


def observe_runtime(
    output: str, *, state_count: int, hbm_bytes: int,
    matmul_records: int,
) -> Tp4TrainingObservation:
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
        raise RuntimeError(f"two-step TP4 segment closure failed: {segments}")
    if (len(states) != 3 or tuple(int(item[0]) for item in states) != (0, 1, 2) or
            any(int(item[1]) != hbm_bytes or item[4:] != ("0", "1")
                for item in states)):
        raise RuntimeError(f"TP4 physical state bytes/version/pass failed: {states}")
    if (tuple(item[:3] for item in steps) != (("0", "0", "1"), ("1", "1", "2")) or
            any(int(item[3]) != state_count or
                int(item[4]) != matmul_records or
                int(item[5]) != state_count or
                int(item[6]) != state_count or
                item[7:] != ("0", "1") for item in steps)):
        raise RuntimeError(f"TP4 WGRAD/SGD/store two-step closure failed: {steps}")
    if drain != [("2", "1")] or len(makespan) != 1:
        raise RuntimeError("TP4 one-shot drain/unique SIM_RESULT failed")
    sgd = output.count("[TRAIN_SGD]")
    if sgd != 2 * state_count or output.count("[DENSE_SEQUENCE_PROGRAM_IO]") != 2:
        raise RuntimeError("TP4 SGD or typed ProgramIO step count failed")
    return Tp4TrainingObservation(
        versions=(0, 1, 2),
        hbm_digests=tuple(item[2] for item in states),
        sgd_invocations=sgd, makespan_cycles=int(makespan[0]),
    )


__all__ = ["Tp4TrainingObservation", "_validate_static_bindings", "observe_runtime"]
