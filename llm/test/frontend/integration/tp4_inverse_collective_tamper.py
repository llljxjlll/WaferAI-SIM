"""Tamper real TP4 linked DTE and SUM records; the source gate must reject both."""

from __future__ import annotations

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.full_dense_two_step_physical_dag import (
    require_named_tp_collective_reverse_records,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragments


def require_tp4_inverse_collective_tamper_rejections(linked, plan, physical) -> None:
    """Prove one removed native peer transfer or one removed native SUM fails."""
    dag = physical.physical_dag
    by_id = {action.id: action for action in dag.actions}
    operation_by_source = {
        action.source_action_ref: (
            action.source_action_ref, action.operation_ref,
            action.step, action.layer, action.phase,
        )
        for action in dag.actions
    }
    assert len(operation_by_source) == len(dag.actions)
    native = {}
    for fragment in _leaf_fragments(linked.manifest.fragments):
        for stream in fragment.core_streams:
            for record in stream.records:
                native.setdefault(record.source_global_action_id, set()).add(record.opcode)
    source_edges = tuple(sorted(
        (by_id[send].source_action_ref, by_id[recv].source_action_ref)
        for send, recv in dag.transport_edges
    ))
    original = require_named_tp_collective_reverse_records(
        linked, plan, operation_by_source, native, source_edges,
    )
    if not original:
        raise AssertionError("TP4 source has no inverse collective to test")
    for opcode in (RecordOpcode.DTE_SEND, RecordOpcode.LOCAL_REDUCE):
        candidates = tuple(sorted(
            action_ref for action_ref, (_, operation, step, _, phase)
            in operation_by_source.items()
            if operation in original and step == 0 and phase == "backward"
            and opcode in native.get(action_ref, set())
        ))
        if not candidates:
            raise AssertionError(f"TP4 source has no native {opcode.name} to tamper")
        action_ref = candidates[0]
        forged = {ref: set(opcodes) for ref, opcodes in native.items()}
        forged[action_ref].remove(opcode)
        try:
            require_named_tp_collective_reverse_records(
                linked, plan, operation_by_source, forged, source_edges,
            )
        except SchemaError as exc:
            if exc.path != action_ref or "exact native DTE/SUM opcode" not in str(exc):
                raise AssertionError(f"{opcode.name} tamper failed for a different cause: {exc}") from exc
        else:
            raise AssertionError(f"removed {opcode.name} did not fail physical source gate")


__all__ = ["require_tp4_inverse_collective_tamper_rejections"]
