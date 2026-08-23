"""Calibrated resource-cycle estimators for intra-die candidates."""

from __future__ import annotations

import math

from ..errors import SchemaError
from ..schema.intra_die_timing_model import IntraDieTimingModel
from ..schema.ir0 import GemmWorkload
from ..schema.ir1 import IR1
from ..schema.ir2 import SemanticTask, SemanticTaskKind


DEFAULT_INTRA_DIE_HARDWARE_DIGEST = (
    "0696e5805d9b6a94b2652db341a6e3b5ccef387212a55b8e537a0e71a3a7c322"
)
DEFAULT_INTRA_DIE_SIMULATION_DIGEST = (
    "32bce414f159454eff62612d46ffb2d9b85d75fad8e31973af94329bb7b5ce3c"
)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def create_intra_die_timing_model(
    *, hardware_digest: str, simulation_digest: str,
) -> IntraDieTimingModel:
    """Return the frozen calibration table for an exact input pair."""

    return IntraDieTimingModel.create(
        hardware_digest=hardware_digest,
        simulation_digest=simulation_digest,
    )


def resolve_intra_die_timing_model(
    ir1: IR1,
    model: IntraDieTimingModel | None = None,
    *,
    hardware_digest: str | None = None,
    simulation_digest: str | None = None,
) -> IntraDieTimingModel:
    if type(ir1) is not IR1:
        raise SchemaError("must be an IR1", path="ir1")
    ir1.validate("ir1")
    expected_hardware = hardware_digest or DEFAULT_INTRA_DIE_HARDWARE_DIGEST
    expected_simulation = simulation_digest or DEFAULT_INTRA_DIE_SIMULATION_DIGEST
    if model is None:
        model = create_intra_die_timing_model(
            hardware_digest=expected_hardware,
            simulation_digest=expected_simulation,
        )
    elif type(model) is not IntraDieTimingModel:
        raise SchemaError("must be an IntraDieTimingModel", path="timing_model")
    model.validate_binding(
        hardware_digest=expected_hardware,
        simulation_digest=expected_simulation,
    )
    return model


def estimate_gemm_cycles(
    task: SemanticTask,
    model: IntraDieTimingModel,
    *,
    parallel_divisor: int = 1,
) -> int:
    if type(task) is not SemanticTask or task.compute is None:
        raise SchemaError("must be a compute task", path="task")
    workload = task.compute.workload
    if type(workload) is not GemmWorkload:
        raise SchemaError("must carry a GemmWorkload", path="task.compute.workload")
    if type(parallel_divisor) is not int or parallel_divisor < 1:
        raise SchemaError("must be positive", path="parallel_divisor")
    m, n, k = workload.rank_shape
    operations = 2 * m * n * k
    return model.compute_setup_cycles + max(
        1,
        _ceil_div(
            operations,
            model.effective_gemm_ops_per_cycle * parallel_divisor,
        ),
    )


def estimate_transport_cycles(
    payload_bytes: int, model: IntraDieTimingModel, *, operation_count: int = 1,
) -> int:
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise SchemaError("must be non-negative", path="payload_bytes")
    if type(operation_count) is not int or operation_count < 0:
        raise SchemaError("must be non-negative", path="operation_count")
    return model.local_transport_setup_cycles * operation_count + _ceil_div(
        payload_bytes, model.local_transport_bytes_per_cycle
    )


def estimate_reduce_cycles(payload_bytes: int, model: IntraDieTimingModel) -> int:
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise SchemaError("must be non-negative", path="payload_bytes")
    return model.local_reduce_setup_cycles + _ceil_div(
        payload_bytes, model.local_reduce_bytes_per_cycle
    )


def estimate_task_cycles(
    task: SemanticTask,
    ir1: IR1,
    model: IntraDieTimingModel | None = None,
    *,
    hardware_digest: str | None = None,
    simulation_digest: str | None = None,
) -> int:
    """Return a deterministic cycle weight for critical-path ordering."""

    if type(task) is not SemanticTask:
        raise SchemaError("must be a SemanticTask", path="task")
    resolved = resolve_intra_die_timing_model(
        ir1,
        model,
        hardware_digest=hardware_digest,
        simulation_digest=simulation_digest,
    )
    if task.kind is SemanticTaskKind.COMP and task.compute is not None:
        if type(task.compute.workload) is GemmWorkload:
            return estimate_gemm_cycles(task, resolved)
        return resolved.compute_setup_cycles + max(1, math.prod(task.shape or (1,)))
    if task.kind is SemanticTaskKind.REDUCE:
        return estimate_reduce_cycles(task.bytes, resolved)
    if task.kind in {
        SemanticTaskKind.DMA_IN,
        SemanticTaskKind.DMA_OUT,
        SemanticTaskKind.LOCAL_COPY,
        SemanticTaskKind.LOCAL_SEND,
        SemanticTaskKind.LOCAL_RECV,
        SemanticTaskKind.SEND,
        SemanticTaskKind.RECV,
        SemanticTaskKind.TRANSIT,
    }:
        return estimate_transport_cycles(task.bytes, resolved)
    return max(1, resolved.sync_issue_cycles)


__all__ = [
    "DEFAULT_INTRA_DIE_HARDWARE_DIGEST",
    "DEFAULT_INTRA_DIE_SIMULATION_DIGEST",
    "create_intra_die_timing_model",
    "estimate_gemm_cycles",
    "estimate_reduce_cycles",
    "estimate_task_cycles",
    "estimate_transport_cycles",
    "resolve_intra_die_timing_model",
]
