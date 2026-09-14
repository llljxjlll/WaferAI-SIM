"""Build and validate the typed workload/DMA action sidecar."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.e2e_workload_graph import E2EOperationKind
from ..schema.external_dma_action_graph import (
    ExternalDmaActionGraph,
    ExternalDmaBoundAction,
    ExternalDmaBoundActionKind,
)
from ..schema.external_dma_program import ExternalDmaProgram
from ..schema.memory_plan import MemoryObjectKind
from ..schema.offload import BlockingOffloadPlan, OffloadOperationKind
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import WorkloadMaterializationManifest
from .offload import validate_blocking_offload_plan


def build_external_dma_action_graph(
    *,
    manifest: WorkloadMaterializationManifest,
    plan: BlockingOffloadPlan,
    program: ExternalDmaProgram,
    _validate_bindings: bool = True,
) -> ExternalDmaActionGraph:
    """Merge P4 DMA and P3 logical operations with explicit P2 residency refs."""

    manifest.validate("manifest")
    program.validate("program")
    validate_blocking_offload_plan(
        plan,
        request_digest=manifest.request_digest,
        logical_graph_digest=manifest.logical_graph_digest,
        source_memory_plan_digest=canonical_digest(manifest.memory_plan),
    )
    expected = {
        "request_digest": manifest.request_digest,
        "logical_graph_digest": manifest.logical_graph_digest,
        "source_memory_plan_digest": canonical_digest(manifest.memory_plan),
        "blocking_offload_plan_digest": canonical_digest(plan),
    }
    for name, value in expected.items():
        if getattr(program, name) != value:
            raise SchemaError(
                "program source binding does not match workload",
                path=f"program.{name}",
                code="external_dma_action_source_mismatch",
            )
    if program.blocking_offload_plan_id != plan.id:
        raise SchemaError(
            "program plan id does not match workload plan",
            path="program.blocking_offload_plan_id",
            code="external_dma_action_source_mismatch",
        )

    offload_operations = {item.id: item for item in plan.operations}
    residency_by_allocation: dict[str, tuple[str, ...]] = {}
    for allocation in plan.memory_plan.allocations:
        residency_by_allocation[allocation.id] = tuple(
            sorted(
                item.id
                for item in plan.memory_plan.residencies
                if item.allocation_ref == allocation.id
            )
        )
    actions: list[ExternalDmaBoundAction] = []
    dma_action_by_descriptor: dict[str, str] = {}
    bring_in_actions: list[str] = []
    parameter_versions = tuple(
        sorted(
            version.id
            for mapping in plan.state_mappings
            for version in plan.source_memory_plan.state_versions
            for request in plan.source_memory_plan.requests
            if version.id == mapping.source_state_version_ref
            and request.state_version_ref == version.id
            and request.object_kind is MemoryObjectKind.PARAMETER
        )
    )
    parameter_residencies: set[str] = set()
    for descriptor in program.descriptors:
        operation = offload_operations.get(descriptor.operation_ref)
        if operation is None:
            raise SchemaError(
                "descriptor references unknown offload operation",
                path=f"program.descriptors[{descriptor.sequence}].operation_ref",
            )
        dependencies = tuple(
            dma_action_by_descriptor[item]
            for item in descriptor.depends_on
            if item in dma_action_by_descriptor
        )
        if len(dependencies) != len(descriptor.depends_on):
            raise SchemaError(
                "DMA dependency is absent or not topological",
                path=f"program.descriptors[{descriptor.sequence}].depends_on",
            )
        residency_refs = residency_by_allocation.get(
            operation.hbm_allocation_ref or "", ()
        )
        action = ExternalDmaBoundAction.create(
            sequence=len(actions),
            kind=ExternalDmaBoundActionKind.DMA_TRANSFER,
            source_ref=descriptor.id,
            depends_on=dependencies,
            state_version_refs=(operation.state_version_ref,),
            residency_refs=residency_refs,
        )
        actions.append(action)
        dma_action_by_descriptor[descriptor.id] = action.id
        if operation.kind is OffloadOperationKind.BRING_IN:
            bring_in_actions.append(action.id)
            parameter_residencies.update(residency_refs)

    logical_action_by_operation: dict[str, str] = {}
    for operation in manifest.logical_graph.operations:
        dependency_ids = [
            logical_action_by_operation[item]
            for item in operation.deps
            if item in logical_action_by_operation
        ]
        if len(dependency_ids) != len(operation.deps):
            raise SchemaError(
                "logical operation dependency is absent or not topological",
                path=f"logical_graph.operations[{operation.sequence_index}].deps",
            )
        residency_refs: tuple[str, ...] = ()
        state_refs = tuple(sorted(set(operation.reads + operation.writes)))
        if operation.kind is E2EOperationKind.PARAMETER_LOAD or operation.parameter_ref is not None:
            dependency_ids.extend(bring_in_actions)
            state_refs = tuple(sorted(set(state_refs + parameter_versions)))
            residency_refs = tuple(sorted(parameter_residencies))
        action = ExternalDmaBoundAction.create(
            sequence=len(actions),
            kind=ExternalDmaBoundActionKind.LOGICAL_OPERATION,
            source_ref=operation.id,
            depends_on=tuple(dependency_ids),
            state_version_refs=state_refs,
            residency_refs=residency_refs,
        )
        actions.append(action)
        logical_action_by_operation[operation.id] = action.id

    graph = ExternalDmaActionGraph.create(
        manifest_digest=manifest.digest,
        case_digest=program.case_digest,
        request_digest=manifest.request_digest,
        logical_graph_digest=manifest.logical_graph_digest,
        source_memory_plan_digest=canonical_digest(manifest.memory_plan),
        blocking_offload_plan_digest=canonical_digest(plan),
        external_dma_program_digest=canonical_digest(program),
        actions=tuple(actions),
    )
    if _validate_bindings:
        validate_external_dma_action_graph(
            graph, manifest=manifest, plan=plan, program=program
        )
    return graph


def validate_external_dma_action_graph(
    graph: ExternalDmaActionGraph,
    *,
    manifest: WorkloadMaterializationManifest,
    plan: BlockingOffloadPlan,
    program: ExternalDmaProgram,
) -> None:
    graph.validate()
    manifest.validate("manifest")
    plan.validate("plan")
    program.validate("program")
    expected = {
        "manifest_digest": manifest.digest,
        "case_digest": program.case_digest,
        "request_digest": manifest.request_digest,
        "logical_graph_digest": manifest.logical_graph_digest,
        "source_memory_plan_digest": canonical_digest(manifest.memory_plan),
        "blocking_offload_plan_digest": canonical_digest(plan),
        "external_dma_program_digest": canonical_digest(program),
    }
    for name, value in expected.items():
        if getattr(graph, name) != value:
            raise SchemaError(
                "action graph source binding mismatch",
                path=f"graph.{name}",
                code="external_dma_action_source_mismatch",
            )
    dma_refs = {item.id for item in program.descriptors}
    logical_refs = {item.id for item in manifest.logical_graph.operations}
    observed_dma = {
        item.source_ref
        for item in graph.actions
        if item.kind is ExternalDmaBoundActionKind.DMA_TRANSFER
    }
    observed_logical = {
        item.source_ref
        for item in graph.actions
        if item.kind is ExternalDmaBoundActionKind.LOGICAL_OPERATION
    }
    if observed_dma != dma_refs or observed_logical != logical_refs:
        raise SchemaError(
            "action graph does not exactly cover DMA and logical sources",
            path="graph.actions",
            code="external_dma_action_coverage",
        )
    expected_graph = build_external_dma_action_graph(
        manifest=manifest,
        plan=plan,
        program=program,
        _validate_bindings=False,
    )
    if graph.actions != expected_graph.actions:
        raise SchemaError(
            "action dependency, state, or residency binding mismatch",
            path="graph.actions",
            code="external_dma_action_binding_mismatch",
        )
    known_residencies = {item.id for item in plan.memory_plan.residencies}
    if any(
        not set(item.residency_refs).issubset(known_residencies)
        for item in graph.actions
    ):
        raise SchemaError(
            "action references residency outside the bound plan",
            path="graph.actions.residency_refs",
            code="external_dma_action_source_mismatch",
        )


__all__ = [
    "build_external_dma_action_graph",
    "validate_external_dma_action_graph",
]
