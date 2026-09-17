"""Source-bound cross-core STORE0→LOAD1 fences for real DP2 Dense training."""
from __future__ import annotations

from dataclasses import dataclass
import re

from ..errors import SchemaError
from .common import stable_artifact_id
from .global_action import GlobalAction, GlobalActionDAG, LogicalCoreRef
from .ir2 import SemanticTaskKind, StateIoOrigin

_FENCE_SCHEMA = "wafer_frontend.dense_dp2_state_version_fence/v1alpha1"
_STEP = re.compile(r"::step([01])(?:_backward)?__dp([01])$")


@dataclass(frozen=True, slots=True)
class DenseStateVersionFence:
    binding_ref: str
    store_action_id: str
    load_action_id: str
    source_core: LogicalCoreRef
    destination_core: LogicalCoreRef

    @property
    def source_ref(self) -> str:
        return stable_artifact_id("dense_dp2_state_version", {
            "binding_ref": self.binding_ref,
            "store_action_id": self.store_action_id,
            "load_action_id": self.load_action_id,
            "source_core": self.source_core,
            "destination_core": self.destination_core,
        }, schema_version=_FENCE_SCHEMA)

    def symbol_id(self, role: str) -> str:
        if role not in ("source_core", "destination_core", "event_tag"):
            raise SchemaError("unknown dense state-version runtime symbol role", path="role")
        return stable_artifact_id("dense_dp2_state_version_symbol", {
            "fence": self.source_ref, "role": role,
        }, schema_version=_FENCE_SCHEMA)


def dense_dp2_state_version_fences(
    dag: GlobalActionDAG,
) -> tuple[DenseStateVersionFence, ...]:
    """Pair each replica-local trainable STORE0 with first LOAD1 on every peer core.

    The exact DP2 gradient region is the opt-in. Ordinary state DMA and MoE do
    not acquire synthetic events. IR1 validation separately proves that these
    physical actions originate in the source's state-version graph.
    """
    if not any(action.region_id.startswith("region.dp_gradient.")
               for action in dag.actions):
        return ()
    stores: dict[tuple[str, int], GlobalAction] = {}
    reads: dict[tuple[str, int, LogicalCoreRef], list[GlobalAction]] = {}
    for action in dag.actions:
        if action.task_kind not in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT):
            continue
        origin = action.origin_ref
        if not isinstance(origin, StateIoOrigin) or len(action.state_uses) != 1:
            raise SchemaError("DP2 state version requires exact typed StateIoOrigin", path=action.id)
        match = _STEP.search(origin.node_ref)
        if match is None or action.logical_core is None:
            raise SchemaError("DP2 state version requires exact step/replica/core", path=action.id)
        step, dp = map(int, match.groups())
        binding = action.state_uses[0].hbm_binding_ref
        key = (binding, dp)
        if step == 0 and action.task_kind is SemanticTaskKind.DMA_OUT:
            if key in stores:
                raise SchemaError("duplicate step0 physical parameter STORE", path=binding)
            stores[key] = action
        elif step == 1 and action.task_kind is SemanticTaskKind.DMA_IN:
            reads.setdefault((binding, dp, action.logical_core), []).append(action)
    if not stores or set(stores) != {(binding, dp) for binding, dp, _core in reads}:
        raise SchemaError("DP2 state version lacks a store/read pair per HBM binding", path="global_dag.actions")
    result: list[DenseStateVersionFence] = []
    for (binding, dp, destination), actions in sorted(
        reads.items(), key=lambda item: (item[0][0], item[0][1],
                                        item[0][2].die_id, item[0][2].local_core_id),
    ):
        store = stores[binding, dp]
        if store.logical_core.die_id != destination.die_id:
            raise SchemaError("DP2 state binding crosses physical Die homes", path=binding)
        first = min(actions, key=lambda action: action.core_order_index)
        if store.logical_core == destination:
            if first.core_order_index <= store.core_order_index:
                raise SchemaError("same-core LOAD1 precedes STORE0", path=binding)
            continue
        result.append(DenseStateVersionFence(
            binding, store.id, first.id, store.logical_core, destination,
        ))
    return tuple(result)


__all__ = ["DenseStateVersionFence", "dense_dp2_state_version_fences"]
