"""Exact source-derived AdamW HBM state version task edges."""
from __future__ import annotations

from ..errors import SchemaError
from .ir0 import AdamwUpdateWorkload, EdgeKind, StateAccessMode
from .ir1 import IR1


def dense_adamw_two_step_state_access_pairs(ir1: IR1) -> tuple[tuple[str, str], ...]:
    """Map each step-0 update access to its step-1 access for the same StateABI.

    The mapping exists only for the full two-step 15-parameter AdamW source.
    Every pair must carry an explicit update0 -> update1 control edge in IR-1.
    """
    optimizers = {node.id: node for node in ir1.nodes
                  if type(node.workload) is AdamwUpdateWorkload}
    if not optimizers:
        return ()
    if len(optimizers) != 30 or len(ir1.nodes) != 170:
        return ()
    controls = {(edge.source_node, edge.destination_node)
                for edge in ir1.edges if edge.kind is EdgeKind.CONTROL}
    by_state: dict[str, dict[int, object]] = {}
    for access in ir1.state_accesses:
        node = optimizers.get(access.node_ref)
        if node is None:
            continue
        if access.mode is not StateAccessMode.READ_WRITE:
            raise SchemaError("AdamW version access must read and write", path=access.id)
        step = node.workload.step
        if step not in (1, 2):
            raise SchemaError("two-step AdamW has unexpected update step", path=node.id)
        state = by_state.setdefault(access.state_ref, {})
        if step in state:
            raise SchemaError("AdamW state has duplicate update step", path=access.state_ref)
        state[step] = access
    if len(by_state) != 75:
        raise SchemaError("two-step AdamW needs 75 exact HBM states", path="ir1.state_accesses")
    pairs = []
    for state_ref, versions in sorted(by_state.items()):
        if set(versions) != {1, 2}:
            raise SchemaError("AdamW state omits a source version", path=state_ref)
        old, new = versions[1], versions[2]
        if (old.node_ref, new.node_ref) not in controls:
            raise SchemaError("AdamW state lacks source update0-to-update1 edge",
                              path=state_ref)
        pairs.append((old.id, new.id))
    return tuple(pairs)


__all__ = ["dense_adamw_two_step_state_access_pairs"]
