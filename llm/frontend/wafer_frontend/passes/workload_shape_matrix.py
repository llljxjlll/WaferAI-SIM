"""Build the canonical P5 1..10 shape preflight matrix."""

from __future__ import annotations

from ..schema.common import DType
from ..schema.memory_plan import MemoryTier, MemoryTierCapacity
from ..schema.serde import canonical_digest
from ..schema.workload_run import (
    WorkloadCapabilityLevel,
    WorkloadFamily,
    WorkloadFamilyCapability,
    WorkloadInferenceSteps,
    WorkloadMeshSpec,
    WorkloadModelArchitecture,
    WorkloadModelSpec,
    WorkloadOptimizerKind,
    WorkloadOptimizerSpec,
    WorkloadParallelSpec,
    WorkloadRunCapability,
    WorkloadRunRequest,
    WorkloadStepSpec,
    WorkloadTrainingSteps,
)
from ..schema.workload_shape_matrix import (
    P5_MAX_MESH_COLUMNS,
    P5_MAX_MESH_ROWS,
    WorkloadLogicalWork,
    WorkloadShapeCaseReport,
    WorkloadShapeMappingMode,
    WorkloadShapeMatrix,
)
from .workload_materialization import materialize_workload_preflight


_HBM_CAPACITY_BYTES = 1 << 30


def _preflight_capability() -> WorkloadRunCapability:
    schema = WorkloadCapabilityLevel.SCHEMA_ONLY
    not_measured = WorkloadCapabilityLevel.NOT_MEASURED
    unsupported = WorkloadCapabilityLevel.UNSUPPORTED
    families = tuple(
        WorkloadFamilyCapability(
            family=family,
            full_model=schema,
            motif=schema,
            baseline=schema,
            optimized=not_measured,
            lowering=schema,
            runtime=not_measured,
            timing=not_measured,
            functional=not_measured,
            capacity=schema,
            multi_step=schema,
            remote_hbm=unsupported,
            external_offload=schema,
            sgd_optimizer=schema,
            adamw_optimizer=not_measured,
            repeatability=not_measured,
        )
        for family in WorkloadFamily
    )
    return WorkloadRunCapability.create(
        max_mesh_rows=P5_MAX_MESH_ROWS,
        max_mesh_columns=P5_MAX_MESH_COLUMNS,
        max_mesh_ranks=P5_MAX_MESH_ROWS * P5_MAX_MESH_COLUMNS,
        families=families,
    )


def _mapping_mode(rows: int, columns: int) -> WorkloadShapeMappingMode:
    rank_count = rows * columns
    if (
        rank_count <= 4
        or rows == 1
        or columns == 1
        or rows == columns
        or (rows + columns) % 2 == 0
    ):
        return WorkloadShapeMappingMode.MESH_SCALED_ALL_DIES
    return WorkloadShapeMappingMode.FIXED_FOUR_RANK_IDLE_DIES


def _spread_four_dies(rank_count: int) -> tuple[int, ...]:
    return tuple(index * (rank_count - 1) // 3 for index in range(4))


def _largest_divisor(value: int, candidates: tuple[int, ...]) -> int:
    return next(candidate for candidate in candidates if value % candidate == 0)


def _parallel_spec(
    family: WorkloadFamily,
    rows: int,
    columns: int,
) -> tuple[WorkloadShapeMappingMode, WorkloadParallelSpec]:
    rank_count = rows * columns
    mode = _mapping_mode(rows, columns)
    if mode is WorkloadShapeMappingMode.FIXED_FOUR_RANK_IDLE_DIES:
        active = _spread_four_dies(rank_count)
        if family.is_moe:
            return mode, WorkloadParallelSpec(
                tp=2,
                dp=1,
                ep=2,
                active_die_ids=active,
            )
        return mode, WorkloadParallelSpec(
            tp=2,
            dp=2,
            active_die_ids=active,
        )

    active = tuple(range(rank_count))
    tp = _largest_divisor(rank_count, (4, 2, 1))
    remaining = rank_count // tp
    ep = (
        _largest_divisor(remaining, (4, 2, 1))
        if family.is_moe
        else 1
    )
    return mode, WorkloadParallelSpec(
        tp=tp,
        dp=remaining // ep,
        ep=ep,
        active_die_ids=active,
    )


def _model(family: WorkloadFamily) -> WorkloadModelSpec:
    return WorkloadModelSpec(
        architecture=(
            WorkloadModelArchitecture.LLAMA_MOE
            if family.is_moe
            else WorkloadModelArchitecture.LLAMA_DENSE
        ),
        vocabulary_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        head_dim=1,
        max_sequence_length=8,
        dtype=DType.FP16,
        num_experts=4 if family.is_moe else 0,
        experts_per_token=1 if family.is_moe else 0,
    )


def build_workload_shape_request(
    family: WorkloadFamily,
    rows: int,
    columns: int,
) -> tuple[WorkloadShapeMappingMode, WorkloadRunRequest]:
    mode, parallel = _parallel_spec(family, rows, columns)
    if family.is_training:
        steps = WorkloadStepSpec(
            training=WorkloadTrainingSteps(
                step_count=2,
                global_batch_size=parallel.dp,
                micro_batch_size=1,
                micro_batch_count=1,
                sequence_length=2,
            )
        )
        optimizer = WorkloadOptimizerSpec(
            kind=WorkloadOptimizerKind.SGD,
            learning_rate=0.01,
        )
    else:
        steps = WorkloadStepSpec(
            inference=WorkloadInferenceSteps(
                prefill_tokens=2,
                decode_steps=2,
                request_count=parallel.dp,
            )
        )
        optimizer = None
    return mode, WorkloadRunRequest.create(
        family=family,
        model=_model(family),
        steps=steps,
        mesh=WorkloadMeshSpec(rows=rows, columns=columns),
        parallel=parallel,
        optimizer=optimizer,
    )


def _capacities(active_die_ids: tuple[int, ...]) -> tuple[MemoryTierCapacity, ...]:
    return tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{die_id}",
            base_address=0,
            capacity_bytes=_HBM_CAPACITY_BYTES,
            alignment_bytes=16,
        )
        for die_id in active_die_ids
    )


def _logical_work(manifest) -> WorkloadLogicalWork:
    request = manifest.request
    if request.family.is_training:
        training = request.steps.training
        assert training is not None
        step_count = training.step_count
        token_count = (
            training.step_count
            * training.global_batch_size
            * training.sequence_length
        )
    else:
        inference = request.steps.inference
        assert inference is not None
        step_count = 1 + inference.decode_steps
        token_count = (
            inference.prefill_tokens + inference.decode_steps
        ) * inference.request_count
    return WorkloadLogicalWork(
        layer_count=request.model.num_layers,
        workload_step_count=step_count,
        token_count=token_count,
        logical_rank_count=manifest.placement.logical_rank_count,
        operation_count=len(manifest.logical_graph.operations),
        tensor_value_count=len(manifest.logical_graph.tensor_values),
        tensor_value_bytes=sum(
            value.size_bytes for value in manifest.logical_graph.tensor_values
        ),
        transport_request_count=len(manifest.transport_requests),
        transport_payload_bytes=sum(
            item.transfer_bytes for item in manifest.transport_requests
        ),
        memory_state_count=len(manifest.state_inventory),
        memory_reserved_bytes=sum(
            item.reserved_bytes for item in manifest.memory_plan.allocations
        ),
    )


def build_workload_shape_case_report(
    family: WorkloadFamily,
    rows: int,
    columns: int,
    *,
    capability: WorkloadRunCapability | None = None,
) -> WorkloadShapeCaseReport:
    """Materialize one deterministic schema/preflight case; never run a backend."""

    capability = capability or _preflight_capability()
    mode, request = build_workload_shape_request(family, rows, columns)
    manifest = materialize_workload_preflight(
        request,
        capability,
        capacities=_capacities(request.parallel.active_die_ids),
    )
    return WorkloadShapeCaseReport.create(
        request=request,
        mapping_mode=mode,
        active_die_ids=manifest.placement.active_die_ids,
        idle_die_ids=manifest.placement.idle_die_ids,
        logical_work=_logical_work(manifest),
        materialization_id=manifest.id,
        materialization_digest=canonical_digest(manifest),
        placement_digest=manifest.placement.digest,
        logical_graph_digest=manifest.logical_graph_digest,
        transport_requests_digest=canonical_digest(manifest.transport_requests),
        memory_plan_digest=canonical_digest(manifest.memory_plan),
        materialization_status=manifest.status,
        unsupported_requirements=manifest.unsupported_requirements,
        is_large_mesh_canary=(
            rows == P5_MAX_MESH_ROWS
            and columns == P5_MAX_MESH_COLUMNS
        ),
    )


def build_workload_shape_matrix() -> WorkloadShapeMatrix:
    """Materialize all 100 rectangles x four workload families."""

    capability = _preflight_capability()
    reports = tuple(
        build_workload_shape_case_report(
            family,
            row,
            column,
            capability=capability,
        )
        for row in range(1, P5_MAX_MESH_ROWS + 1)
        for column in range(1, P5_MAX_MESH_COLUMNS + 1)
        for family in WorkloadFamily
    )
    return WorkloadShapeMatrix.create(reports=reports)


__all__ = [
    "build_workload_shape_case_report",
    "build_workload_shape_matrix",
    "build_workload_shape_request",
]
