"""Exact Stage-4 sliced KV handoff producer.

The PD plan owns the logical head-intersection matrix, placement owns the
physical cross-group routes, and IR-1 owns the persistent state/access
identities.  This pass joins those three authoritative inputs without
inventing payloads or endpoint pairs.
"""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.ir0 import StateAccess, StateAccessMode
from ..schema.ir1 import CrossGroupRoute, IR1, PhysicalGroup
from ..schema.n4 import Stage4InterDiePlannedIR1
from ..schema.persistent_state import PersistentStateDecl, StateKind
from ..schema.stage4_pd import Stage4KvReshardKind, Stage4PdMode, Stage4PdPlan
from ..schema.state_transfer import SlicedKvStateTransferContract


_PRODUCER_PASS = "stage4_state_transfer"
_KV_KIND_ORDER = {
    StateKind.KV_KEY: 0,
    StateKind.KV_VALUE: 1,
}


def _selected_groups(
    ir1: IR1,
    plan: Stage4PdPlan,
) -> tuple[PhysicalGroup, PhysicalGroup]:
    by_instance: dict[str, list[PhysicalGroup]] = defaultdict(list)
    for group in ir1.groups:
        by_instance[group.instance_id].append(group)
    source = by_instance.get(plan.prefill_instance_ref, [])
    destination = by_instance.get(plan.decode_instance_ref, [])
    if len(source) != 1 or len(destination) != 1:
        raise SchemaError(
            "selected Stage 4 instances must each own exactly one physical group",
            path="ir1.groups",
        )
    return source[0], destination[0]


def _route_index(
    ir1: IR1,
    plan: Stage4PdPlan,
    source_group: PhysicalGroup,
    destination_group: PhysicalGroup,
) -> dict[tuple[int, int], CrossGroupRoute]:
    expected_endpoints = {
        (flow.source_rank, flow.destination_rank)
        for handoff in plan.handoffs
        for flow in handoff.flows
    }
    routes: dict[tuple[int, int], CrossGroupRoute] = {}
    for index, route in enumerate(ir1.cross_routes):
        if type(route) is not CrossGroupRoute:
            raise SchemaError(
                "must be a CrossGroupRoute",
                path=f"ir1.cross_routes[{index}]",
            )
        endpoint = (route.source_rank, route.destination_rank)
        if (
            route.source_group_ref != source_group.id
            or route.destination_group_ref != destination_group.id
        ):
            raise SchemaError(
                "Stage 4 route must connect the selected prefill/decode groups",
                path=f"ir1.cross_routes[{index}]",
            )
        if endpoint in routes:
            raise SchemaError(
                "duplicate Stage 4 route endpoint pair",
                path=f"ir1.cross_routes[{index}]",
            )
        routes[endpoint] = route
    if set(routes) != expected_endpoints:
        raise SchemaError(
            "cross routes must exactly equal the plan flow endpoint pairs",
            path="ir1.cross_routes",
        )
    return routes


def _state_access_index(
    ir1: IR1,
) -> tuple[
    dict[
        tuple[str, str, int, StateKind, int],
        tuple[StateAccess, PersistentStateDecl],
    ],
    set[tuple[str, str, int, StateKind, int]],
]:
    manifest = ir1.persistent_state_manifest
    if manifest is None:
        raise SchemaError(
            "Stage 4 handoff requires persistent-state placement",
            path="ir1.persistent_state_manifest",
        )
    declarations = {
        declaration.id: declaration for declaration in manifest.declarations
    }
    index: dict[
        tuple[str, str, int, StateKind, int],
        tuple[StateAccess, PersistentStateDecl],
    ] = {}
    selected: set[tuple[str, str, int, StateKind, int]] = set()
    for access_index, access in enumerate(ir1.state_accesses):
        declaration = declarations.get(access.state_ref)
        if declaration is None:
            raise SchemaError(
                "references an unknown persistent state",
                path=f"ir1.state_accesses[{access_index}].state_ref",
            )
        identity = declaration.identity
        if identity.kind not in _KV_KIND_ORDER:
            continue
        if identity.request_ref is None or identity.layer_index is None:
            raise SchemaError(
                "KV state requires request/layer lineage",
                path=f"ir1.state_accesses[{access_index}]",
            )
        key = (
            identity.instance_ref,
            identity.request_ref,
            identity.layer_index,
            identity.kind,
            access.rank,
        )
        if key in index:
            raise SchemaError(
                "Stage 4 KV access identity must be unique",
                path=f"ir1.state_accesses[{access_index}]",
            )
        index[key] = (access, declaration)
        selected.add(key)
    return index, selected


def _validate_inputs(ir1: IR1, plan: Stage4PdPlan) -> None:
    if type(ir1) is not IR1:
        raise SchemaError("must be an IR1", path="ir1")
    if type(plan) is not Stage4PdPlan:
        raise SchemaError("must be a Stage4PdPlan", path="stage4_pd_plan")
    ir1.validate("ir1")
    plan.validate("stage4_pd_plan")
    if plan.mode is not Stage4PdMode.SEPARATED:
        raise UnsupportedFeatureError(
            "sliced KV transfer requires separated PD",
            path="stage4_pd_plan.mode",
        )
    if ir1.pd_plan_id != plan.id:
        raise SchemaError(
            "must equal the supplied Stage 4 plan id",
            path="ir1.pd_plan_id",
        )
    profiles = {
        binding.instance_ref: binding.profile for binding in ir1.instance_profiles
    }
    if (
        profiles.get(plan.prefill_instance_ref) != plan.prefill_profile.key
        or profiles.get(plan.decode_instance_ref) != plan.decode_profile.key
    ):
        raise SchemaError(
            "instance profile provenance disagrees with the Stage 4 plan",
            path="ir1.instance_profiles",
        )


def _derive_stage4_state_transfers(
    ir1: IR1,
    plan: Stage4PdPlan,
) -> tuple[SlicedKvStateTransferContract, ...]:
    _validate_inputs(ir1, plan)
    source_group, destination_group = _selected_groups(ir1, plan)
    routes = _route_index(
        ir1,
        plan,
        source_group,
        destination_group,
    )
    accesses, observed_kv_keys = _state_access_index(ir1)
    source_width = plan.num_kv_heads // plan.prefill_tp
    destination_width = plan.num_kv_heads // plan.decode_tp
    expected_kv_keys: set[tuple[str, str, int, StateKind, int]] = set()
    result: list[SlicedKvStateTransferContract] = []

    for handoff in plan.handoffs:
        for flow in handoff.flows:
            route = routes[(flow.source_rank, flow.destination_rank)]
            source_head_start = flow.source_rank * source_width
            destination_head_start = flow.destination_rank * destination_width
            source_local_head = flow.head_slice.start - source_head_start
            destination_local_head = (
                flow.head_slice.start - destination_head_start
            )
            shape = (
                handoff.token_count,
                flow.head_slice.count,
                plan.head_dim,
            )
            bytes_per_tensor = (
                handoff.token_count
                * flow.head_slice.count
                * plan.head_dim
                * 2
            )
            if flow.bytes != 2 * bytes_per_tensor:
                raise SchemaError(
                    "plan flow bytes must equal the K/V slice pair",
                    path=(
                        f"stage4_pd_plan.handoffs[{handoff.layer_index}]"
                        ".flows"
                    ),
                )
            for kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
                source_key = (
                    plan.prefill_instance_ref,
                    handoff.request_ref,
                    handoff.layer_index,
                    kind,
                    flow.source_rank,
                )
                destination_key = (
                    plan.decode_instance_ref,
                    handoff.request_ref,
                    handoff.layer_index,
                    kind,
                    flow.destination_rank,
                )
                expected_kv_keys.update((source_key, destination_key))
                source_entry = accesses.get(source_key)
                destination_entry = accesses.get(destination_key)
                if source_entry is None or destination_entry is None:
                    raise SchemaError(
                        "plan flow has no exact source/destination KV state access",
                        path="ir1.state_accesses",
                    )
                source_access, _source_declaration = source_entry
                destination_access, _destination_declaration = destination_entry
                if (
                    source_access.mode is not StateAccessMode.WRITE
                    or destination_access.mode is not StateAccessMode.WRITE
                ):
                    raise SchemaError(
                        "Stage 4 v2 source/destination accesses must be WRITE",
                        path="ir1.state_accesses",
                    )
                contract = SlicedKvStateTransferContract.create(
                    producer_pass=_PRODUCER_PASS,
                    source_ir1_id=ir1.id,
                    source_state_access_ref=source_access.id,
                    destination_state_access_ref=destination_access.id,
                    cross_group_route_ref=route.id,
                    source_local_offset=(0, source_local_head, 0),
                    source_local_shape=shape,
                    destination_local_offset=(0, destination_local_head, 0),
                    destination_local_shape=shape,
                    bytes=bytes_per_tensor,
                )
                contract.validate_against(ir1)
                result.append(contract)

    selected_instances = {
        plan.prefill_instance_ref,
        plan.decode_instance_ref,
    }
    relevant_observed = {
        key for key in observed_kv_keys if key[0] in selected_instances
    }
    if relevant_observed != expected_kv_keys:
        raise SchemaError(
            "selected Stage 4 KV accesses must exactly equal plan K/V lineage",
            path="ir1.state_accesses",
        )
    return tuple(result)


def validate_stage4_state_transfers(
    contracts: tuple[SlicedKvStateTransferContract, ...],
    ir1: IR1,
    plan: Stage4PdPlan,
    *,
    path: str = "stage4_state_transfers",
) -> None:
    """Independently rebuild and compare the complete transfer set."""

    if type(contracts) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    for index, contract in enumerate(contracts):
        if type(contract) is not SlicedKvStateTransferContract:
            raise SchemaError(
                "entries must be SlicedKvStateTransferContract",
                path=f"{path}[{index}]",
            )
        if contract.producer_pass != _PRODUCER_PASS:
            raise SchemaError(
                f"must be {_PRODUCER_PASS!r}",
                path=f"{path}[{index}].producer_pass",
            )
        contract.validate_against(ir1, f"{path}[{index}]")
    expected = _derive_stage4_state_transfers(ir1, plan)
    if contracts != expected:
        raise SchemaError(
            "must exactly equal the plan/IR1-derived sliced KV transfers",
            path=path,
        )
    if len({contract.id for contract in contracts}) != len(contracts):
        raise SchemaError("contract ids must be unique", path=path)
    if sum(contract.bytes for contract in contracts) != sum(
        handoff.logical_unique_bytes for handoff in plan.handoffs
    ):
        raise SchemaError(
            "total contract bytes must equal logical unique handoff bytes",
            path=path,
        )


def build_stage4_state_transfers(
    ir1: IR1,
    plan: Stage4PdPlan,
) -> tuple[SlicedKvStateTransferContract, ...]:
    """Build every per-layer/request/rank K/V slice in canonical plan order."""

    result = _derive_stage4_state_transfers(ir1, plan)
    validate_stage4_state_transfers(result, ir1, plan)
    return result


def _validate_planned_source(
    source: Stage4InterDiePlannedIR1,
) -> None:
    if type(source) is not Stage4InterDiePlannedIR1:
        raise SchemaError(
            "must be a Stage4InterDiePlannedIR1",
            path="source",
        )
    source.validate("source")
    if source.pd_plan.mode is Stage4PdMode.FUSED:
        if source.pd_plan.handoffs or source.graph.cross_routes:
            raise SchemaError(
                "fused PD requires no handoffs or cross-group routes",
                path="source",
            )
        return
    if source.pd_plan.reshard is not Stage4KvReshardKind.ONE_TO_ONE:
        raise UnsupportedFeatureError(
            "segmented KV reshard state transfers are not implemented",
            path="source.pd_plan.reshard",
        )


def validate_stage4_planned_state_transfers(
    contracts: tuple[SlicedKvStateTransferContract, ...],
    source: Stage4InterDiePlannedIR1,
    *,
    path: str = "stage4_planned_state_transfers",
) -> None:
    """Rebuild the exact transfer set from one authoritative N4 carrier."""

    _validate_planned_source(source)
    if source.pd_plan.mode is Stage4PdMode.FUSED:
        if contracts != ():
            raise SchemaError(
                "fused PD must have an empty state-transfer set",
                path=path,
            )
        return
    validate_stage4_state_transfers(
        contracts,
        source.graph,
        source.pd_plan,
        path=path,
    )


def build_stage4_planned_state_transfers(
    source: Stage4InterDiePlannedIR1,
) -> tuple[SlicedKvStateTransferContract, ...]:
    """Build canonical equal-TP sliced KV transfers from a planned carrier."""

    _validate_planned_source(source)
    result = (
        ()
        if source.pd_plan.mode is Stage4PdMode.FUSED
        else _derive_stage4_state_transfers(source.graph, source.pd_plan)
    )
    validate_stage4_planned_state_transfers(result, source)
    return result


__all__ = [
    "build_stage4_planned_state_transfers",
    "build_stage4_state_transfers",
    "validate_stage4_planned_state_transfers",
    "validate_stage4_state_transfers",
]
