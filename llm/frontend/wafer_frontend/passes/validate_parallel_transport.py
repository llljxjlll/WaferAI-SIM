"""Independent P3 operation/value to P1 transport binding oracle."""

from __future__ import annotations

from collections import Counter

from ..errors import SchemaError
from ..schema.e2e_workload_graph import E2EOperationKind, E2EWorkloadGraph
from ..schema.parallel_placement import ParallelGroupKind, ParallelPlacement
from ..schema.parallel_transport import (
    ParallelCommunicationKind,
    ParallelCommunicationRequest,
    ParallelTransportPlan,
)


def _expected_bindings(graph: E2EWorkloadGraph) -> Counter[tuple[object, ...]]:
    placement = graph.placement
    values = {value.id: value for value in graph.tensor_values}
    groups = {group.id: group for group in placement.groups}
    expected: Counter[tuple[object, ...]] = Counter()

    def add(operation, kind, group_id, refs) -> None:
        payload = tuple(refs)
        sizes = {values[value_ref].size_bytes for value_ref in payload}
        if not payload or len(sizes) != 1:
            raise SchemaError(
                "logical communication payload bytes are ambiguous",
                path="graph.operations",
            )
        payload_bytes = sizes.pop()
        expected[
            (
                operation.id,
                operation.phase,
                operation.step,
                operation.layer,
                kind,
                group_id,
                payload,
                payload_bytes,
                payload_bytes == 0,
            )
        ] += 1

    for operation in graph.operations:
        if operation.kind is E2EOperationKind.QKV:
            for group in placement.select_groups(ParallelGroupKind.TP):
                add(
                    operation,
                    ParallelCommunicationKind.ALL_GATHER,
                    group.id,
                    tuple(
                        value_ref
                        for value_ref in operation.input_value_refs
                        if values[value_ref].logical_rank in group.ranks
                    ),
                )
        elif operation.kind is E2EOperationKind.GRADIENT_SYNC:
            for group_ref in operation.group_refs:
                group = groups[group_ref]
                payload = tuple(
                    value_ref
                    for value_ref in operation.input_value_refs
                    if values[value_ref].logical_rank in group.ranks
                )
                for value_ref in payload:
                    value = values[value_ref]
                    owner = next(
                        item
                        for item in placement.ownership_domains
                        if item.id == value.owner_domain_ref
                    )
                    if owner.synchronization_group_id != group_ref:
                        raise SchemaError(
                            "gradient payload owner disagrees with P1 group",
                            path="graph.operations",
                        )
                add(
                    operation,
                    ParallelCommunicationKind.ALL_REDUCE,
                    group_ref,
                    payload,
                )
        elif operation.kind in (E2EOperationKind.DISPATCH, E2EOperationKind.COMBINE):
            source_refs = (
                operation.output_value_refs
                if operation.kind is E2EOperationKind.DISPATCH
                else operation.input_value_refs
            )
            for group in placement.select_groups(ParallelGroupKind.EP):
                for value_ref in source_refs:
                    if values[value_ref].logical_rank in group.ranks:
                        add(
                            operation,
                            ParallelCommunicationKind.ALL_TO_ALL,
                            group.id,
                            (value_ref,),
                        )
    return expected


def validate_parallel_transport_workload_bindings(
    requests: tuple[ParallelCommunicationRequest, ...],
    graph: E2EWorkloadGraph,
    placement: ParallelPlacement,
    plan: ParallelTransportPlan | None = None,
) -> None:
    """Check workload semantics independently from the request producer."""

    graph.validate("graph")
    placement.validate("placement")
    if graph.placement != placement:
        raise SchemaError("graph and transport placement differ", path="placement")
    operations = {operation.id: operation for operation in graph.operations}
    values = {value.id: value for value in graph.tensor_values}
    groups = {group.id: group for group in placement.groups}
    actual: Counter[tuple[object, ...]] = Counter()
    for index, request in enumerate(requests):
        path = f"requests[{index}]"
        request.validate(path)
        if (
            request.workload_case_id != graph.request.case_id
            or request.workload_request_digest != graph.request.digest
        ):
            raise SchemaError("request is bound to another workload case", path=path)
        operation = operations.get(request.logical_operation_ref)
        if operation is None or (
            request.workload_phase,
            request.workload_step,
            request.workload_layer,
        ) != (operation.phase, operation.step, operation.layer):
            raise SchemaError("request operation metadata disagrees", path=path)
        group = groups.get(request.group_id)
        if group is None:
            raise SchemaError("request group is absent", path=path)
        payload_values = tuple(values.get(item) for item in request.payload_value_refs)
        if any(value is None for value in payload_values):
            raise SchemaError("request payload value is absent", path=path)
        if any(value.logical_rank not in group.ranks for value in payload_values):
            raise SchemaError("payload rank is outside request group", path=path)
        if any(value.size_bytes != request.transfer_bytes for value in payload_values):
            raise SchemaError("payload bytes disagree with tensor value", path=path)
        actual[
            (
                operation.id,
                operation.phase,
                operation.step,
                operation.layer,
                request.kind,
                request.group_id,
                request.payload_value_refs,
                request.transfer_bytes,
                request.is_noop,
            )
        ] += 1
    if actual != _expected_bindings(graph):
        raise SchemaError(
            "transport requests do not cover exact logical communication points",
            path="requests",
        )
    if plan is not None:
        if plan.requests != requests:
            raise SchemaError("plan requests differ", path="plan.requests")
        plan.validate()


__all__ = ["validate_parallel_transport_workload_bindings"]
