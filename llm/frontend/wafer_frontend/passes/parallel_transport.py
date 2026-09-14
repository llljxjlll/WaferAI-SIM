"""Lower explicit parallel placement communication onto physical XY routes."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import CoreRuntimeBinding
from ..schema.common import stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.ir1 import D2DLink, PairRoute, PhysicalFabric, RouteHop
from ..schema.parallel_placement import ParallelPlacement
from ..schema.parallel_transport import (
    PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION,
    ParallelCommunicationKind,
    ParallelCommunicationRequest,
    ParallelTransfer,
    ParallelTransferEndpoint,
    ParallelTransferEndpointRole,
    ParallelTransportPlan,
    expected_transfer_specs,
)
from ..schema.serde import canonical_digest


def _validate_inputs(
    placement: ParallelPlacement,
    fabric: PhysicalFabric,
    requests: tuple[ParallelCommunicationRequest, ...],
) -> None:
    if type(placement) is not ParallelPlacement:
        raise SchemaError("must be a ParallelPlacement", path="placement")
    if type(fabric) is not PhysicalFabric:
        raise SchemaError("must be a PhysicalFabric", path="fabric")
    if type(requests) is not tuple:
        raise SchemaError("must be an immutable tuple", path="requests")
    placement.validate("placement")
    fabric.validate("fabric")
    if fabric.die_grid != placement.mesh.physical_shape:
        raise SchemaError(
            "fabric shape differs from placement Mesh", path="fabric.die_grid"
        )
    request_ids: set[str] = set()
    group_ids = {group.id for group in placement.groups}
    for index, request in enumerate(requests):
        if type(request) is not ParallelCommunicationRequest:
            raise SchemaError(
                "must be a ParallelCommunicationRequest",
                path=f"requests[{index}]",
            )
        request.validate(f"requests[{index}]")
        if request.id in request_ids:
            raise SchemaError("duplicate request id", path=f"requests[{index}].id")
        request_ids.add(request.id)
        if request.kind is ParallelCommunicationKind.P2P:
            assert request.source_rank is not None
            assert request.destination_rank is not None
            for name in ("source_rank", "destination_rank"):
                if getattr(request, name) >= placement.logical_rank_count:
                    raise SchemaError(
                        "P2P rank is not mapped", path=f"requests[{index}].{name}"
                    )
        elif request.group_id not in group_ids:
            raise SchemaError(
                "collective group is not present in the placement",
                path=f"requests[{index}].group_id",
            )


def _core_bindings(
    placement: ParallelPlacement, fabric: PhysicalFabric
) -> tuple[CoreRuntimeBinding, ...]:
    dies = {die.id: die for die in fabric.dies}
    result: list[CoreRuntimeBinding] = []
    for index, rank_placement in enumerate(placement.rank_placements):
        die = dies.get(rank_placement.die_id)
        if die is None:
            raise SchemaError(
                "placed Die is absent from the fabric",
                path=f"placement.rank_placements[{index}].die_id",
            )
        core = next(
            (
                item
                for item in die.cores
                if item.local_core_id == rank_placement.local_core
            ),
            None,
        )
        if core is None:
            raise SchemaError(
                f"local core is outside this Die's {len(die.cores)} cores",
                path=f"placement.rank_placements[{index}].local_core",
            )
        result.append(
            CoreRuntimeBinding(
                logical_core=LogicalCoreRef(die.id, core.local_core_id),
                core_spec_ref=core.id,
                runtime_core_id=core.runtime_core_id,
                sram_profile_ref=core.sram_profile_ref,
            )
        )
    return tuple(result)


def _xy_die_path(
    placement: ParallelPlacement, source_die: int, destination_die: int
) -> tuple[int, ...]:
    mesh = placement.mesh
    x, y = mesh.coordinate(source_die)
    destination_x, destination_y = mesh.coordinate(destination_die)
    result = [source_die]
    while x != destination_x:
        x += 1 if destination_x > x else -1
        result.append(mesh.rank(y, x))
    while y != destination_y:
        y += 1 if destination_y > y else -1
        result.append(mesh.rank(y, x))
    return tuple(result)


def _link_index(fabric: PhysicalFabric) -> dict[tuple[int, int], D2DLink]:
    result: dict[tuple[int, int], D2DLink] = {}
    for index, link in enumerate(fabric.links):
        key = (link.source_die, link.destination_die)
        if key in result:
            raise SchemaError(
                "fabric has duplicate directed Die links",
                path=f"fabric.links[{index}]",
            )
        result[key] = link
    return result


def _pair_route(
    *,
    placement: ParallelPlacement,
    fabric: PhysicalFabric,
    links: dict[tuple[int, int], D2DLink],
    source_rank: int,
    destination_rank: int,
) -> PairRoute:
    rank_to_die = {
        item.logical_rank: item.die_id for item in placement.rank_placements
    }
    source_die = rank_to_die[source_rank]
    destination_die = rank_to_die[destination_rank]
    die_path = _xy_die_path(placement, source_die, destination_die)
    ports = {die.id: {port.id: port for port in die.ports} for die in fabric.dies}
    hops: list[RouteHop] = []
    route_resources: list[str] = []
    for hop_index, (hop_source, hop_destination) in enumerate(
        zip(die_path, die_path[1:])
    ):
        link = links.get((hop_source, hop_destination))
        if link is None:
            raise SchemaError(
                f"missing directed link {hop_source}->{hop_destination}",
                path="fabric.links",
            )
        source_port = ports[hop_source].get(link.source_port_ref)
        if source_port is None:
            raise SchemaError(
                "link references an unknown source port", path="fabric.links"
            )
        resources = (source_port.egress_resource_id, link.resource_id)
        if link.link_group_ref is not None:
            resources += (link.link_group_ref,)
        hops.append(
            RouteHop(
                index=hop_index,
                link_ref=link.id,
                source_die=hop_source,
                source_port_ref=link.source_port_ref,
                destination_die=hop_destination,
                destination_port_ref=link.destination_port_ref,
                resource_ids=resources,
            )
        )
        for resource_id in resources:
            if resource_id not in route_resources:
                route_resources.append(resource_id)
    semantic = {
        "placement_digest": placement.digest,
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "die_path": die_path,
        "hops": tuple(hops),
        "resource_ids": tuple(route_resources),
    }
    route = PairRoute(
        id=stable_artifact_id(
            "parallel_pair_route",
            semantic,
            schema_version=PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION,
        ),
        source_rank=source_rank,
        destination_rank=destination_rank,
        die_path=die_path,
        hops=tuple(hops),
        resource_ids=tuple(route_resources),
    )
    route.validate("parallel_pair_route")
    return route


def build_parallel_transport_plan(
    placement: ParallelPlacement,
    fabric: PhysicalFabric,
    requests: tuple[ParallelCommunicationRequest, ...],
) -> ParallelTransportPlan:
    """Lower P2P/direct/ring communication and bind every active rank core."""

    _validate_inputs(placement, fabric, requests)
    canonical_requests = tuple(sorted(requests, key=lambda item: item.id))
    specs = expected_transfer_specs(canonical_requests, placement)
    route_pairs = sorted({(spec[3], spec[4]) for spec in specs})
    links = _link_index(fabric)
    route_by_pair = {
        pair: _pair_route(
            placement=placement,
            fabric=fabric,
            links=links,
            source_rank=pair[0],
            destination_rank=pair[1],
        )
        for pair in route_pairs
    }
    transfers: list[ParallelTransfer] = []
    endpoints: list[ParallelTransferEndpoint] = []
    core_by_rank = {
        item.logical_rank: LogicalCoreRef(item.die_id, item.local_core)
        for item in placement.rank_placements
    }
    request_by_id = {request.id: request for request in canonical_requests}
    for request_id, algorithm_phase, algorithm_step, source, destination, transfer_bytes in specs:
        request = request_by_id[request_id]
        semantic = {
            "placement_digest": placement.digest,
            "request_id": request_id,
            "workload_case_id": request.workload_case_id,
            "workload_request_digest": request.workload_request_digest,
            "logical_operation_ref": request.logical_operation_ref,
            "workload_phase": request.workload_phase,
            "workload_step": request.workload_step,
            "workload_layer": request.workload_layer,
            "payload_value_refs": request.payload_value_refs,
            "algorithm_phase": algorithm_phase,
            "algorithm_step": algorithm_step,
            "source_rank": source,
            "destination_rank": destination,
            "bytes": transfer_bytes,
        }
        transfer_id = stable_artifact_id(
            "parallel_transfer",
            semantic,
            schema_version=PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION,
        )
        send_id = f"{transfer_id}.send"
        recv_id = f"{transfer_id}.recv"
        transfers.append(
            ParallelTransfer(
                id=transfer_id,
                request_id=request_id,
                workload_phase=request.workload_phase,
                workload_step=request.workload_step,
                workload_layer=request.workload_layer,
                logical_operation_ref=request.logical_operation_ref,
                payload_value_refs=request.payload_value_refs,
                algorithm_phase=algorithm_phase,
                algorithm_step=algorithm_step,
                source_rank=source,
                destination_rank=destination,
                bytes=transfer_bytes,
                route_id=route_by_pair[(source, destination)].id,
                send_endpoint_id=send_id,
                recv_endpoint_id=recv_id,
            )
        )
        endpoints.extend(
            (
                ParallelTransferEndpoint(
                    id=send_id,
                    transfer_id=transfer_id,
                    role=ParallelTransferEndpointRole.SEND,
                    rank=source,
                    peer_rank=destination,
                    logical_core=core_by_rank[source],
                    bytes=transfer_bytes,
                ),
                ParallelTransferEndpoint(
                    id=recv_id,
                    transfer_id=transfer_id,
                    role=ParallelTransferEndpointRole.RECV,
                    rank=destination,
                    peer_rank=source,
                    logical_core=core_by_rank[destination],
                    bytes=transfer_bytes,
                ),
            )
        )
    result = ParallelTransportPlan.create(
        producer_pass="parallel_transport",
        placement_digest=placement.digest,
        fabric_digest=canonical_digest(fabric),
        requests=canonical_requests,
        core_bindings=_core_bindings(placement, fabric),
        routes=tuple(sorted(route_by_pair.values(), key=lambda item: item.id)),
        transfers=tuple(sorted(transfers, key=lambda item: item.id)),
        endpoints=tuple(sorted(endpoints, key=lambda item: item.id)),
    )
    result.validate_against(placement, fabric)
    return result


__all__ = ["build_parallel_transport_plan"]
