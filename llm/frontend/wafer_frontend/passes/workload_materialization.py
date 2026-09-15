"""Unified P0 preflight materialization for the four full-workload families."""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import stable_artifact_id
from ..schema.e2e_workload_graph import E2EOperationKind, E2EWorkloadGraph
from ..schema.ir1 import PhysicalFabric
from ..schema.memory_plan import (
    MemoryAllocationRequest,
    MemoryObjectKind,
    MemoryStateVersion,
    MemoryTier,
    MemoryTierCapacity,
)
from ..schema.parallel_placement import (
    ParallelGroupKind,
    ParameterOwnershipKind,
    build_dense_parallel_placement,
    build_moe_parallel_placement,
)
from ..schema.parallel_transport import (
    PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION,
    ParallelCommunicationKind,
    ParallelCommunicationRequest,
)
from ..schema.rect_mesh import RectMeshSpec
from ..schema.workload_materialization import (
    WORKLOAD_MATERIALIZATION_SCHEMA_VERSION,
    WorkloadMaterializationManifest,
    WorkloadStateInventoryItem,
)
from ..schema.workload_run import (
    WorkloadFamily,
    WorkloadMemoryMode,
    WorkloadRunCapability,
    WorkloadRunRequest,
)
from .build_e2e_workload_graph import build_e2e_workload_graph
from .memory_plan import plan_hierarchical_memory
from .parallel_transport import build_parallel_transport_plan
from .validate_e2e_workload_graph import validate_e2e_workload_coverage
from .validate_parallel_transport import (
    validate_parallel_transport_workload_bindings,
)


def _parameter_bytes(request: WorkloadRunRequest) -> tuple[int, int]:
    """Return total shared bytes and per-expert bytes in model dtype."""

    model = request.model
    element_bytes = 2
    qkv = model.hidden_size * (
        model.num_attention_heads + 2 * model.num_kv_heads
    ) * model.head_dim
    attention = qkv + model.hidden_size * model.hidden_size + 2 * model.hidden_size
    embeddings_and_head = (
        model.vocabulary_size * model.hidden_size
        + model.hidden_size
        + model.hidden_size * model.vocabulary_size
    )
    if request.family.is_moe:
        router = model.hidden_size * model.num_experts
        shared_elements = embeddings_and_head + model.num_layers * (attention + router)
        per_expert_elements = (
            model.num_layers * 3 * model.hidden_size * model.intermediate_size
        )
    else:
        dense_mlp = 3 * model.hidden_size * model.intermediate_size
        shared_elements = embeddings_and_head + model.num_layers * (
            attention + dense_mlp
        )
        per_expert_elements = 0
    return shared_elements * element_bytes, per_expert_elements * element_bytes


def _placement(request: WorkloadRunRequest):
    mesh = RectMeshSpec(request.mesh.rows, request.mesh.columns)
    active = request.parallel.active_die_ids or None
    common = {
        "mesh": mesh,
        "tp_degree": request.parallel.tp,
        "dp_degree": request.parallel.dp,
        "pp_degree": request.parallel.pp,
        "active_die_ids": active,
    }
    if request.family.is_moe:
        return build_moe_parallel_placement(
            ep_degree=request.parallel.ep,
            num_experts=request.model.num_experts,
            **common,
        )
    return build_dense_parallel_placement(**common)


def _state(
    *,
    name: str,
    kind: MemoryObjectKind,
    rank: int,
    size: int,
    writable: bool,
    owner: str | None = None,
) -> WorkloadStateInventoryItem:
    return WorkloadStateInventoryItem.create(
        logical_name=name,
        object_kind=kind,
        logical_rank=rank,
        owner_domain_ref=owner,
        size_bytes=max(1, size),
        writable=writable,
    )


def _state_inventory(request: WorkloadRunRequest, placement) -> tuple[WorkloadStateInventoryItem, ...]:
    shared_bytes, expert_bytes = _parameter_bytes(request)
    states: list[WorkloadStateInventoryItem] = []
    for owner in placement.ownership_domains:
        total = shared_bytes if owner.kind is ParameterOwnershipKind.SHARED else expert_bytes
        shard_bytes = (total + request.parallel.tp - 1) // request.parallel.tp
        owner_name = (
            "shared"
            if owner.kind is ParameterOwnershipKind.SHARED
            else f"expert.{owner.expert_id}"
        )
        for rank in owner.replica_ranks:
            states.append(
                _state(
                    name=f"parameter.{owner_name}.tp{owner.tp_shard}.rank{rank}",
                    kind=MemoryObjectKind.PARAMETER,
                    rank=rank,
                    size=shard_bytes,
                    writable=request.family.is_training,
                    owner=owner.id,
                )
            )

    if request.family.is_training:
        assert request.steps.training is not None
        tokens = (
            request.steps.training.micro_batch_size
            * request.steps.training.sequence_length
        )
        parameter_states = tuple(states)
        for parameter in parameter_states:
            states.append(
                _state(
                    name=parameter.logical_name.replace("parameter.", "gradient.", 1),
                    kind=MemoryObjectKind.GRADIENT,
                    rank=parameter.logical_rank,
                    size=parameter.size_bytes * 2,
                    writable=True,
                    owner=parameter.owner_domain_ref,
                )
            )
            if request.optimizer is not None and request.optimizer.kind.value == "adamw":
                for state_name, state_bytes in (
                    ("master", parameter.size_bytes * 2),
                    ("m", parameter.size_bytes * 2),
                    ("v", parameter.size_bytes * 2),
                    ("step", 4),
                ):
                    states.append(
                        _state(
                            name=parameter.logical_name.replace(
                                "parameter.",
                                f"optimizer.adamw.{state_name}.",
                                1,
                            ),
                            kind=MemoryObjectKind.OPTIMIZER,
                            rank=parameter.logical_rank,
                            size=state_bytes,
                            writable=True,
                            owner=parameter.owner_domain_ref,
                        )
                    )
    else:
        assert request.steps.inference is not None
        tokens = (
            request.steps.inference.prefill_tokens
            + request.steps.inference.decode_steps
        ) * request.steps.inference.request_count

    kv_per_rank = (
        request.model.num_layers
        * tokens
        * 2
        * (request.model.num_kv_heads // request.parallel.tp)
        * request.model.head_dim
        * 2
    )
    activation_per_rank = (
        max(1, tokens)
        * request.model.hidden_size
        * 2
        * max(1, request.model.num_layers)
    )
    rank_ids = tuple(item.logical_rank for item in placement.rank_placements)
    for rank in rank_ids:
        if not request.family.is_training:
            states.append(
                _state(
                    name=f"kv_cache.rank{rank}",
                    kind=MemoryObjectKind.KV,
                    rank=rank,
                    size=kv_per_rank,
                    writable=True,
                )
            )
        states.append(
            _state(
                name=f"activation.rank{rank}",
                kind=MemoryObjectKind.ACTIVATION,
                rank=rank,
                size=activation_per_rank,
                writable=True,
            )
        )
        states.append(
            _state(
                name=f"communication.rank{rank}",
                kind=MemoryObjectKind.COMMUNICATION,
                rank=rank,
                size=request.model.hidden_size * 2,
                writable=True,
            )
        )
        if request.family.is_moe:
            states.append(
                _state(
                    name=f"moe_buffer.rank{rank}",
                    kind=MemoryObjectKind.MOE_BUFFER,
                    rank=rank,
                    size=max(1, tokens) * request.model.hidden_size * 2,
                    writable=True,
                )
            )
    return tuple(sorted(states, key=lambda item: item.logical_name))


def _transport_requests(
    request: WorkloadRunRequest,
    placement,
    graph: E2EWorkloadGraph,
) -> tuple[ParallelCommunicationRequest, ...]:
    requests: list[ParallelCommunicationRequest] = []
    values = {value.id: value for value in graph.tensor_values}

    def add(kind: ParallelCommunicationKind, group, operation, refs) -> None:
        payload_refs = tuple(refs)
        if not payload_refs:
            raise SchemaError("communication point has no payload values", path="logical_graph")
        sizes = {values[value_ref].size_bytes for value_ref in payload_refs}
        if len(sizes) != 1:
            raise SchemaError("communication payload values disagree on bytes", path="logical_graph")
        payload_bytes = sizes.pop()
        key = {
            "workload_case_id": request.case_id,
            "workload_request_digest": request.digest,
            "logical_operation_ref": operation.id,
            "workload_phase": operation.phase,
            "workload_step": operation.step,
            "workload_layer": operation.layer,
            "payload_value_refs": payload_refs,
            "kind": kind,
            "group_id": group.id,
            "bytes": payload_bytes,
        }
        requests.append(
            ParallelCommunicationRequest(
                id=stable_artifact_id(
                    "workload_transport_request",
                    key,
                    schema_version=PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION,
                ),
                workload_case_id=request.case_id,
                workload_request_digest=request.digest,
                logical_operation_ref=operation.id,
                workload_phase=operation.phase,
                workload_step=operation.step,
                workload_layer=operation.layer,
                payload_value_refs=payload_refs,
                is_noop=payload_bytes == 0,
                kind=kind,
                group_id=group.id,
                source_rank=None,
                destination_rank=None,
                transfer_bytes=payload_bytes,
            )
        )

    for operation in graph.operations:
        if operation.kind is E2EOperationKind.QKV:
            for group in placement.select_groups(ParallelGroupKind.TP):
                payload = tuple(
                    value_ref
                    for value_ref in operation.input_value_refs
                    if values[value_ref].logical_rank in group.ranks
                )
                add(ParallelCommunicationKind.ALL_GATHER, group, operation, payload)
        elif operation.kind is E2EOperationKind.GRADIENT_SYNC:
            groups = {group.id: group for group in placement.groups}
            for group_ref in operation.group_refs:
                group = groups[group_ref]
                payload = tuple(
                    value_ref
                    for value_ref in operation.input_value_refs
                    if values[value_ref].logical_rank in group.ranks
                )
                add(ParallelCommunicationKind.ALL_REDUCE, group, operation, payload)
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
                            ParallelCommunicationKind.ALL_TO_ALL,
                            group,
                            operation,
                            (value_ref,),
                        )
    return tuple(sorted(requests, key=lambda item: item.id))


def _memory_plan(
    request: WorkloadRunRequest,
    placement,
    inventory: tuple[WorkloadStateInventoryItem, ...],
    capacities: tuple[MemoryTierCapacity, ...],
):
    capacity_keys = {(item.tier, item.location_ref) for item in capacities}
    die_for_rank = {
        item.logical_rank: item.die_id for item in placement.rank_placements
    }
    versions: list[MemoryStateVersion] = []
    allocation_requests: list[MemoryAllocationRequest] = []
    dirty: list[str] = []
    lifetime_end = max(4, len(inventory) + 2)
    for item in inventory:
        version = MemoryStateVersion.create(
            state_ref=item.id,
            generation=0,
            predecessor_ref=None,
            writable=item.writable,
        )
        versions.append(version)
        persistent = item.object_kind in (
            MemoryObjectKind.PARAMETER,
            MemoryObjectKind.KV,
            MemoryObjectKind.OPTIMIZER,
        )
        if (
            request.memory.mode is WorkloadMemoryMode.EXTERNAL_OFFLOAD
            and item.object_kind in (MemoryObjectKind.PARAMETER, MemoryObjectKind.OPTIMIZER)
        ):
            assert request.memory.external_tier_ref is not None
            tier = MemoryTier.EXTERNAL
            location = request.memory.external_tier_ref
        else:
            tier = MemoryTier.HBM
            location = f"die:{die_for_rank[item.logical_rank]}"
        if (tier, location) not in capacity_keys:
            raise SchemaError(
                f"state {item.logical_name!r} has no {tier.value} capacity at {location!r}",
                path="capacities",
                code="memory_capacity_missing",
            )
        start = 0 if persistent else 1
        end = lifetime_end if persistent else 3
        allocation_requests.append(
            MemoryAllocationRequest.create(
                state_version_ref=version.id,
                object_kind=item.object_kind,
                tier=tier,
                location_ref=location,
                size_bytes=item.size_bytes,
                alignment_bytes=16,
                lifetime_start=start,
                lifetime_end_exclusive=end,
            )
        )
        if request.family.is_training and item.writable and persistent:
            dirty.append(version.id)
    return plan_hierarchical_memory(
        capacities=capacities,
        state_versions=tuple(versions),
        requests=tuple(allocation_requests),
        dirty_state_version_refs=tuple(dirty),
    )


def materialize_workload_preflight(
    request: WorkloadRunRequest,
    capability: WorkloadRunCapability,
    *,
    capacities: tuple[MemoryTierCapacity, ...],
    fabric: PhysicalFabric | None = None,
) -> WorkloadMaterializationManifest:
    """Build deterministic P0/P1/P2 artifacts without claiming E2E execution."""

    if type(request) is not WorkloadRunRequest:
        raise SchemaError("must be a WorkloadRunRequest", path="request")
    if type(capability) is not WorkloadRunCapability:
        raise SchemaError("must be a WorkloadRunCapability", path="capability")
    if type(capacities) is not tuple or not capacities:
        raise SchemaError("must be a non-empty tuple", path="capacities")
    request.validate("request")
    capability.validate("capability")
    if request.memory.mode is WorkloadMemoryMode.REMOTE_HBM:
        raise UnsupportedFeatureError(
            "remote HBM placement/staging is not materialized by P0.3",
            path="request.memory.mode",
        )
    placement = _placement(request)
    logical_graph = build_e2e_workload_graph(request, placement)
    validate_e2e_workload_coverage(logical_graph)
    inventory = _state_inventory(request, placement)
    communication = _transport_requests(request, placement, logical_graph)
    validate_parallel_transport_workload_bindings(
        communication,
        logical_graph,
        placement,
    )
    transport = None
    if fabric is not None:
        transport = build_parallel_transport_plan(placement, fabric, communication)
        validate_parallel_transport_workload_bindings(
            communication,
            logical_graph,
            placement,
            transport,
        )
    memory = _memory_plan(request, placement, inventory, capacities)
    return WorkloadMaterializationManifest.create(
        request=request,
        capability=capability,
        logical_graph=logical_graph,
        state_inventory=inventory,
        placement=placement,
        transport_requests=communication,
        transport_plan=transport,
        memory_plan=memory,
    )


__all__ = ["materialize_workload_preflight"]
