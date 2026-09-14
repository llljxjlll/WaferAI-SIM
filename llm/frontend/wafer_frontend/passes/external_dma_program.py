"""Lower a validated blocking offload plan to the C++ DMA sidecar."""

from __future__ import annotations

from pathlib import Path

from ..errors import SchemaError
from ..schema.external_dma_program import (
    ExternalDmaBackendBinding,
    ExternalDmaDescriptor,
    ExternalDmaProbe,
    ExternalDmaProgram,
    ExternalDmaSeed,
)
from ..schema.offload import (
    BlockingOffloadPlan,
    OffloadOperationKind,
)
from ..schema.serde import (
    canonical_digest,
    canonical_json,
    load_json_dataclass,
    loads_dataclass,
)
from .offload import validate_blocking_offload_plan


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(
            "must be a lowercase SHA-256 digest",
            path=path,
        )


def finalize_external_dma_program(
    *,
    plan: BlockingOffloadPlan,
    case_digest: str,
    backend_bindings: tuple[ExternalDmaBackendBinding, ...],
    external_seeds: tuple[ExternalDmaSeed, ...],
    external_probes: tuple[ExternalDmaProbe, ...],
) -> ExternalDmaProgram:
    """Create a source-bound, deterministic blocking DMA sidecar."""

    _validate_digest(case_digest, "case_digest")
    validate_blocking_offload_plan(
        plan,
        request_digest=plan.request_digest,
        logical_graph_digest=plan.logical_graph_digest,
        source_memory_plan_digest=plan.source_memory_plan_digest,
    )
    transfers = {item.id: item for item in plan.transfer_requests}
    transfer_kinds = {
        OffloadOperationKind.BRING_IN,
        OffloadOperationKind.DIRTY_WRITEBACK,
    }
    descriptors: list[ExternalDmaDescriptor] = []
    for operation in plan.operations:
        if operation.kind not in transfer_kinds:
            continue
        transfer = transfers.get(operation.transfer_request_ref or "")
        if transfer is None:
            raise SchemaError(
                "transfer operation has no source request",
                path=f"plan.operations[{operation.sequence}]",
            )
        descriptor = ExternalDmaDescriptor.create(
            sequence=len(descriptors),
            operation_ref=operation.id,
            source_transfer_request_ref=transfer.id,
            source_operation_deps=operation.depends_on,
            depends_on=(
                ()
                if not descriptors
                else (descriptors[-1].id,)
            ),
            connection_ref=transfer.connection_ref,
            direction=transfer.direction,
            external_address=transfer.external_address,
            hbm_address=transfer.hbm_address,
            size_bytes=transfer.size_bytes,
            planned_issue_cycle=operation.start_cycle,
            planned_ready_cycle=operation.ready_cycle,
        )
        descriptors.append(descriptor)
    return ExternalDmaProgram.create(
        case_digest=case_digest,
        request_digest=plan.request_digest,
        logical_graph_digest=plan.logical_graph_digest,
        source_memory_plan_digest=plan.source_memory_plan_digest,
        blocking_offload_plan_id=plan.id,
        blocking_offload_plan_digest=canonical_digest(plan),
        fabric=plan.fabric,
        backend_bindings=backend_bindings,
        descriptors=tuple(descriptors),
        external_seeds=external_seeds,
        external_probes=external_probes,
    )


def serialize_external_dma_program(program: ExternalDmaProgram) -> str:
    program.validate()
    return canonical_json(program) + "\n"


def parse_external_dma_program(text: str) -> ExternalDmaProgram:
    result = loads_dataclass(
        ExternalDmaProgram,
        text,
        path="external_dma_program",
    )
    result.validate()
    return result


def load_external_dma_program(path: Path) -> ExternalDmaProgram:
    result = load_json_dataclass(
        ExternalDmaProgram,
        path,
        path="external_dma_program",
    )
    result.validate()
    return result


def write_external_dma_program(
    program: ExternalDmaProgram,
    path: Path,
) -> None:
    payload = serialize_external_dma_program(program)
    try:
        path.write_text(payload, encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SchemaError(
            str(error),
            path="external_dma_program",
        ) from error


__all__ = [
    "finalize_external_dma_program",
    "load_external_dma_program",
    "parse_external_dma_program",
    "serialize_external_dma_program",
    "write_external_dma_program",
]
