"""Source-only GEMM dX provenance and an explicit closed public physical gate.

This NEW-only module cannot confer a public RecordOpcode or a numeric
backward graph. It proves the forward X/W and upstream dY identities against
an actually validated IR0 source while production ISA/frontend are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.ir0 import (
    GemmWorkload, IR0, OpKind, OpPhase, StateAccessMode,
)


@dataclass(frozen=True, slots=True)
class GemmInputDxSourceProvenance:
    forward_op_ref: str
    activation_value_ref: str
    weight_value_ref: str
    weight_state_ref: str
    upstream_value_ref: str
    upstream_producer_ref: str
    rank: int
    k: int
    m: int
    n: int


def derive_gemm_input_dx_source_provenance(
    graph: IR0, workload: GemmInputDxWorkload, *, upstream_value_ref: str,
    rank: int = 0,
) -> GemmInputDxSourceProvenance:
    """Check a real forward StateDecl READ, saved X, W and produced dY."""
    if type(graph) is not IR0 or type(workload) is not GemmInputDxWorkload:
        raise SchemaError("GEMM dX source requires validated IR0 and typed workload", path="source")
    graph.validate("gemm_input_dx_source_graph")
    workload.validate()
    if type(rank) is not int or rank < 0:
        raise SchemaError("GEMM dX rank is invalid", path="source.rank")
    nodes = {node.id: node for node in graph.nodes}
    values = {value.id: value for value in graph.values}
    states = {state.id: state for state in graph.persistent_states}
    forward = nodes.get(workload.source_forward_op_ref)
    state = states.get(workload.source_parameter_state_ref)
    upstream = values.get(upstream_value_ref)
    if (forward is None or forward.kind is not OpKind.GEMM
            or forward.phase is not OpPhase.FWD
            or type(forward.workload) is not GemmWorkload
            or len(forward.inputs) != 2 or len(forward.outputs) != 1
            or state is None or upstream is None):
        raise SchemaError("GEMM dX names one real forward GEMM/StateDecl/dY", path="source")
    activation, weight = (values[ref] for ref in forward.inputs)
    output = values[forward.outputs[0]]
    if (activation.shape != (workload.k, workload.m)
            or weight.shape != (workload.m, workload.n)
            or output.shape != (workload.k, workload.n)
            or any(value.dtype is not DType.FP16 for value in (
                activation, weight, output, upstream,
            ))
            or activation.producer is None
            or state.identity.tensor_ref != weight.id
            or state.shape != weight.shape or state.dtype is not DType.FP16
            or state.tensor_bytes != workload.weight_bytes
            or forward.workload.rank_shape != (workload.k, workload.n, workload.m)
            or upstream.shape != output.shape or upstream.producer is None):
        raise SchemaError("GEMM dX source X/W/dY or StateDecl geometry differs", path="source")
    source_accesses = tuple(access for access in graph.state_accesses
                            if access.node_ref == forward.id
                            and access.state_ref == state.id
                            and access.rank == rank
                            and access.mode is StateAccessMode.READ)
    upstream_producer = nodes.get(upstream.producer)
    if (len(source_accesses) != 1 or upstream_producer is None
            or upstream_producer.phase is not OpPhase.DGRAD
            or upstream.id not in upstream_producer.outputs
            or output.id not in upstream_producer.inputs):
        raise SchemaError("GEMM dX must borrow source StateDecl READ and real downstream dY producer",
                          path="source")
    return GemmInputDxSourceProvenance(
        forward.id, activation.id, weight.id, state.id,
        upstream.id, upstream.producer, rank,
        workload.k, workload.m, workload.n,
    )


def require_public_gemm_input_dx_physical_opcode(
    _: GemmInputDxSourceProvenance,
) -> None:
    """Fail closed before public opcode/codec/load/ProgramIO coverage exists."""
    raise SchemaError("public GEMM_DX_TIMING native physical opcode is absent",
                      path="gemm_input_dx_physical")


__all__ = [
    "GemmInputDxSourceProvenance", "derive_gemm_input_dx_source_provenance",
    "require_public_gemm_input_dx_physical_opcode",
]
