"""Versioned router-score backward must precede expert dY delivery.

Old P2 COMBINE_BACKWARD follows EXPERT_DGRAD and BACKWARD_DX; using it as a
0x28 dExpert producer creates a physical read-before-produce dependency cycle.
This NEW-only DAG proposal is not a production MoeRectPlan/GlobalActionDAG.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.common import stable_artifact_id
from ..schema.flexible_moe import MoeRectActionKind, MoeRectFlowStage
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_router_dexpert_handoff import (
    MoeRouterDexpertHandoff,
)
from .moe_full_train_router_native_protocol import (
    MoeRouterSourceNativeProtocol,
)
from .moe_full_train_router_return_protocol import (
    MoeRouterSignedReturnProtocol,
)
from .moe_full_train_router_score_source import (
    MoeTrainableSignedRouterRequirements,
)

_VERSION = "wafer_frontend.moe_router_score_backward_pre_dispatch/v1alpha1"
_EARLY_KIND = "score_weight_backward_pre_dispatch"


@dataclass(frozen=True, slots=True)
class MoeRouterDagNode:
    action_ref: str
    predecessors: tuple[str, ...]
    kind: str


@dataclass(frozen=True, slots=True)
class MoeRouterEarlyBackwardPath:
    step: int
    layer: int
    source_rank: int
    early_action_ref: str
    required_dcombined_producer_ref: str
    forward_weighted_action_ref: str
    late_original_combine_backward_ref: str
    backward_send_refs: tuple[str, ...]
    local_expert_dgrad_refs: tuple[str, ...]
    remote_expert_dgrad_refs: tuple[str, ...]
    logical_flops: int


@dataclass(frozen=True, slots=True)
class MoeRouterPreDispatchDagProposal:
    id: str
    source_moe_sequence_ref: str
    source_dynamic_router_ref: str
    source_return_ref: str
    source_native_ref: str
    source_dexpert_handoff_ref: str
    version: str
    paths: tuple[MoeRouterEarlyBackwardPath, ...]
    nodes_by_step_layer: tuple[tuple[int, int, tuple[MoeRouterDagNode, ...]], ...]

    def validate_topology(self) -> None:
        for step, layer, nodes in self.nodes_by_step_layer:
            by_id = {node.action_ref: node for node in nodes}
            if len(by_id) != len(nodes):
                raise SchemaError("new early router DAG has repeated physical action id",
                                  path=f"router_early.step{step}.layer{layer}")
            for node in nodes:
                if len(set(node.predecessors)) != len(node.predecessors):
                    raise SchemaError("new early router DAG repeats a dependency edge",
                                      path=f"router_early.step{step}.layer{layer}")
                if any(pred not in by_id for pred in node.predecessors):
                    raise SchemaError("new early router DAG has a source-unbound predecessor",
                                      path=f"router_early.step{step}.layer{layer}")
            visited = set()
            active = set()
            def walk(action_ref):
                if action_ref in active:
                    raise SchemaError("router pre-dispatch DAG has a physical dependency cycle",
                                      path=f"router_early.step{step}.layer{layer}")
                if action_ref in visited:
                    return
                active.add(action_ref)
                for pred in by_id[action_ref].predecessors:
                    walk(pred)
                active.remove(action_ref)
                visited.add(action_ref)
            for node in nodes:
                walk(node.action_ref)
            def precedes(upstream, downstream):
                if upstream == downstream:
                    return False
                stack = list(by_id[downstream].predecessors)
                seen = set()
                while stack:
                    item = stack.pop()
                    if item == upstream:
                        return True
                    if item not in seen:
                        seen.add(item)
                        stack.extend(by_id[item].predecessors)
                return False
            for path in (item for item in self.paths
                         if (item.step, item.layer) == (step, layer)):
                early, producer = (path.early_action_ref,
                                   path.required_dcombined_producer_ref)
                if (early not in by_id or by_id[early].kind != _EARLY_KIND
                        or producer not in by_id
                        or by_id[producer].kind != "source_claimed_shared_dcombined_producer"
                        or not precedes(producer, early)
                        or not precedes(path.forward_weighted_action_ref,
                                        early)):
                    raise SchemaError("0x28 must consume actual dCombined and retained weighted forward output before expert dispatch",
                                      path=f"router_early.step{step}.layer{layer}")
                for send in path.backward_send_refs:
                    if not precedes(early, send):
                        raise SchemaError("0x28 must dominate BACKWARD_GRADIENT SEND before expert reads its dY",
                                          path=f"router_early.step{step}.layer{layer}")
                for local in path.local_expert_dgrad_refs:
                    if not precedes(early, local):
                        raise SchemaError("0x28 must dominate local expert DGRAD before down derivative reads dY",
                                          path=f"router_early.step{step}.layer{layer}")
                for remote in path.remote_expert_dgrad_refs:
                    if not precedes(early, remote):
                        raise SchemaError("0x28 must dominate remote expert DGRAD through real gradient SEND/WAIT",
                                          path=f"router_early.step{step}.layer{layer}")
                if not all(precedes(item, path.late_original_combine_backward_ref)
                           for item in (*path.local_expert_dgrad_refs,
                                        *path.remote_expert_dgrad_refs)):
                    raise SchemaError("original late COMBINE_BACKWARD must retain real dX return dependencies",
                                      path=f"router_early.step{step}.layer{layer}")
                if path.logical_flops <= 0:
                    raise SchemaError("versioned pre-dispatch score dot+expert scale cannot be zero work",
                                      path=f"router_early.step{step}.layer{layer}")

    def require_production_source(self, sequence: MoeCompileSequence) -> None:
        """Reject the old P2 case until an *official* re-signed plan exists."""
        self.validate_topology()
        units = {(unit.step, unit.layer): unit for unit in sequence.units}
        if self.source_moe_sequence_ref != sequence.id:
            raise SchemaError("early router proposal is not bound to P2 plan/source case",
                              path="router_early.production")
        for path in self.paths:
            unit = units[(path.step, path.layer)]
            actions = {action.id: action for action in unit.plan.actions}
            proposed = actions.get(path.early_action_ref)
            if proposed is None or proposed.kind.value != _EARLY_KIND:
                raise SchemaError("old P2 has no official source-backed pre-dispatch 0x28 action",
                                  path=f"router_early.step{path.step}.layer{path.layer}")
            if path.required_dcombined_producer_ref not in actions:
                raise SchemaError("official new source must bind a real shared dCombined producer before 0x28",
                                  path=f"router_early.step{path.step}.layer{path.layer}")
            # Following production stages must still validate cloned
            # action→IR1/GlobalActionDAG/Schedule, BufferABI and ProgramIO.


def build_moe_router_pre_dispatch_proposal(
    score: MoeTrainableSignedRouterRequirements,
    returned: MoeRouterSignedReturnProtocol,
    native: MoeRouterSourceNativeProtocol,
    handoff: MoeRouterDexpertHandoff,
    sequence: MoeCompileSequence,
    *,
    dcombined_producer_refs: dict[tuple[int, int], str],
) -> MoeRouterPreDispatchDagProposal:
    score.validate_against(sequence)
    returned.validate_against(score, sequence)
    native.validate_against(score, returned, sequence)
    handoff.validate_against(score, returned, native, sequence)
    keys = {(pair.step, pair.layer) for pair in native.pairs}
    if (type(dcombined_producer_refs) is not dict
            or set(dcombined_producer_refs) != keys
            or len(set(dcombined_producer_refs.values())) != len(keys)
            or any(type(ref) is not str or not ref for ref in
                   dcombined_producer_refs.values())):
        raise SchemaError("each step/layer needs its own exact upstream shared dCombined action ref",
                          path="router_early.dcombined")
    units = {(unit.step, unit.layer): unit for unit in sequence.units}
    paths = []
    dags = []
    for pair in native.pairs:
        unit = units[(pair.step, pair.layer)]
        plan = unit.plan
        dcombined_producer_ref = dcombined_producer_refs[(pair.step, pair.layer)]
        score_path = next(item for item in score.paths
                          if (item.step, item.layer, item.source_rank)
                          == (pair.step, pair.layer, pair.source_rank))
        segments = [item for item in handoff.segments
                    if (item.step, item.layer, item.source_rank)
                    == (pair.step, pair.layer, pair.source_rank)]
        remote = tuple(item for item in segments
                       if item.backward_flow_ref is not None)
        local = tuple(item for item in segments
                      if item.backward_flow_ref is None)
        if not remote or not local:
            raise SchemaError("bounded EP2 early router requires local and remote real P2 experts",
                              path=f"router_early.step{pair.step}.layer{pair.layer}")
        early = stable_artifact_id("moe_score_weight_backward_pre_dispatch",
            {"plan": plan.id, "step": pair.step, "layer": pair.layer,
             "source_rank": pair.source_rank,
             "shared_producer": dcombined_producer_ref,
             "weighted_forward": score_path.weighted_combine_action_ref},
             schema_version=_VERSION)
        if early in {action.id for action in plan.actions}:
            raise SchemaError("versioned early backward cannot alias old late action",
                              path="router_early.source")
        if dcombined_producer_ref in {action.id for action in plan.actions}:
            raise SchemaError("P2 shared dCombined producer must be independently proven in a new global source",
                              path="router_early.source")
        paths.append(MoeRouterEarlyBackwardPath(
            pair.step, pair.layer, pair.source_rank, early,
            dcombined_producer_ref, score_path.weighted_combine_action_ref,
            score_path.combine_backward_action_ref,
            tuple(item.send_action_ref for item in remote),
            tuple(item.dgrad_action_ref for item in local),
            tuple(item.dgrad_action_ref for item in remote),
            pair.score_backward.logical_flops))
        nodes = [MoeRouterDagNode(action.id, action.deps,
                                  action.kind.value)
                 for action in plan.actions]
        nodes.append(MoeRouterDagNode(dcombined_producer_ref,
                                      (score_path.weighted_combine_action_ref,),
                                      "source_claimed_shared_dcombined_producer"))
        nodes.append(MoeRouterDagNode(early,
                  (dcombined_producer_ref,
                   score_path.weighted_combine_action_ref), _EARLY_KIND))
        rewrite = {item.send_action_ref for item in remote}
        rewrite.update(item.dgrad_action_ref for item in local)
        nodes = [MoeRouterDagNode(node.action_ref,
                 (*node.predecessors, early) if node.action_ref in rewrite
                 else node.predecessors, node.kind)
                 for node in nodes]
        dags.append((pair.step, pair.layer, tuple(nodes)))
    semantic = {"source_moe_sequence_ref": sequence.id,
                "source_dynamic_router_ref": score.id,
                "source_return_ref": returned.id,
                "source_native_ref": native.id,
                "source_dexpert_handoff_ref": handoff.id,
                "version": _VERSION, "paths": tuple(paths),
                "nodes_by_step_layer": tuple(dags)}
    proposal = MoeRouterPreDispatchDagProposal(
        stable_artifact_id("moe_router_pre_dispatch_dag_proposal",
                           semantic, schema_version=_VERSION), **semantic)
    proposal.validate_topology()
    return proposal


__all__ = ["MoeRouterDagNode", "MoeRouterEarlyBackwardPath",
           "MoeRouterPreDispatchDagProposal",
           "build_moe_router_pre_dispatch_proposal"]
