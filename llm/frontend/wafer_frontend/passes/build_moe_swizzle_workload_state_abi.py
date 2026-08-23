"""Project IR1 persistent expert weights into the MoE whole-workload StateABI."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.ir0 import OpKind, StateAccessMode
from ..schema.ir1 import IR1
from ..schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateLifetime,
    StateKind,
)
from ..schema.swizzle_moe_execution import (
    MoeScaleExecution,
    MoeScaleExecutionActionKind,
)
from ..schema.swizzle_moe_scale import MoeSwizzleScaleOracle, MoeSwizzleScaleSpec
from ..schema.swizzle_moe_state import (
    MoeSwizzleStateActionBinding,
    MoeSwizzleStateTensor,
    MoeSwizzleWorkloadStateABI,
    parse_moe_expert_weight_tensor_ref,
)


_ROLES = ("gate", "up", "down")


def build_moe_swizzle_workload_state_abi(
    ir1: IR1,
    execution: MoeScaleExecution,
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
) -> MoeSwizzleWorkloadStateABI:
    """Build and cross-lock the exact 12 expert weights and all 3T DMA loads."""

    ir1.validate("build_moe_swizzle_workload_state_abi.ir1")
    spec.validate("build_moe_swizzle_workload_state_abi.spec")
    oracle.validate("build_moe_swizzle_workload_state_abi.oracle")
    execution.validate_against(spec, oracle, "build_moe_swizzle_workload_state_abi.execution")
    manifest = ir1.persistent_state_manifest
    if manifest is None:
        raise SchemaError("IR1 lacks a persistent state manifest", path="build_moe_swizzle_workload_state_abi.ir1")
    if len(manifest.declarations) != 12 or len(manifest.bindings) != 12:
        raise SchemaError("MoE StateABI requires exactly 12 persistent expert weights", path="build_moe_swizzle_workload_state_abi.ir1.persistent_state_manifest")
    if spec.expert_count != 4 or tuple(spec.expert_home_die_ids) != (0, 1, 2, 3):
        raise SchemaError("MoE StateABI requires canonical EP4 home placement", path="build_moe_swizzle_workload_state_abi.spec")

    spaces = {item.die_id: item for item in manifest.address_spaces}
    bindings = {item.state_ref: item for item in manifest.bindings}
    nodes = {item.id: item for item in ir1.nodes}
    accesses_by_state = defaultdict(list)
    for access in ir1.state_accesses:
        if (
            access.mode is not StateAccessMode.READ
            or access.read_offset is not None
            or access.read_shape is not None
            or access.write_offset is not None
            or access.write_shape is not None
        ):
            raise SchemaError("expert weights require whole-tensor READ accesses", path="build_moe_swizzle_workload_state_abi.ir1.state_accesses")
        node = nodes.get(access.node_ref)
        if node is None or node.kind is not OpKind.GEMM:
            raise SchemaError("expert weight state access must target a GEMM node", path="build_moe_swizzle_workload_state_abi.ir1.state_accesses")
        accesses_by_state[access.state_ref].append(access)
    if set(accesses_by_state) != {item.id for item in manifest.declarations}:
        raise SchemaError("state-access coverage does not exactly cover expert weights", path="build_moe_swizzle_workload_state_abi.ir1.state_accesses")

    tensors = []
    tensor_by_key = {}
    for declaration in manifest.declarations:
        tensor_ref = declaration.identity.tensor_ref
        if tensor_ref is None:
            raise SchemaError("expert parameter lacks tensor_ref", path="build_moe_swizzle_workload_state_abi.ir1.persistent_state_manifest")
        expert, role = parse_moe_expert_weight_tensor_ref(tensor_ref)
        key = (expert, role)
        if key in tensor_by_key:
            raise SchemaError("expert-role state is duplicated", path="build_moe_swizzle_workload_state_abi.ir1.persistent_state_manifest")
        binding = bindings.get(declaration.id)
        space = spaces.get(expert)
        accesses = accesses_by_state[declaration.id]
        if (
            declaration.identity.kind is not StateKind.PARAMETER
            or declaration.identity.shard_index != expert
            or declaration.lifetime is not PersistentStateLifetime.PERSISTENT
            or declaration.access is not PersistentStateAccess.READ_ONLY
            or binding is None
            or binding.die_id != expert
            or binding.size_bytes != declaration.tensor_bytes
            or space is None
            or any(
                access.rank != expert
                or tensor_ref not in nodes[access.node_ref].inputs
                for access in accesses
            )
        ):
            raise SchemaError("expert state declaration/binding/access is not exact", path="build_moe_swizzle_workload_state_abi.ir1.persistent_state_manifest")
        tensor = MoeSwizzleStateTensor(
            expert_index=expert,
            role=role,
            tensor_ref=tensor_ref,
            state_ref=declaration.id,
            hbm_binding_ref=binding.id,
            home_die_id=binding.die_id,
            hbm_address=binding.address,
            size_bytes=binding.size_bytes,
            shape=declaration.shape,
            dtype=declaration.dtype,
            layout=declaration.layout,
            address_space_base=space.base_address,
            address_space_size=space.size_bytes,
            alignment_bytes=space.alignment_bytes,
            state_access_refs=tuple(sorted(item.id for item in accesses)),
        )
        tensor.validate("build_moe_swizzle_workload_state_abi.tensor")
        tensor_by_key[key] = tensor
        tensors.append(tensor)
    if set(tensor_by_key) != {(expert, role) for expert in range(4) for role in _ROLES}:
        raise SchemaError("persistent manifest lacks exact expert-role closure", path="build_moe_swizzle_workload_state_abi.ir1.persistent_state_manifest")

    action_bindings = []
    dma_actions = tuple(
        item for item in execution.actions
        if item.kind is MoeScaleExecutionActionKind.DMA_IN
    )
    if len(dma_actions) != 3 * spec.tokens:
        raise SchemaError("execution DMA_IN cardinality is not 3T", path="build_moe_swizzle_workload_state_abi.execution.actions")
    for action in dma_actions:
        parts = action.role.split(".")
        if (
            len(parts) != 2
            or parts[0] not in _ROLES
            or parts[1] != "weight"
            or action.die_id != spec.expert_home_die_ids[action.expert_index]
            or action.read_values
            or len(action.write_values) != 1
            or action.dtype is not DType.FP16
        ):
            raise SchemaError("execution DMA_IN weight semantics are not exact", path="build_moe_swizzle_workload_state_abi.execution.actions")
        role = parts[0]
        tensor = tensor_by_key[(action.expert_index, role)]
        if action.bytes != tensor.size_bytes:
            raise SchemaError("execution DMA_IN bytes disagree with persistent weight", path="build_moe_swizzle_workload_state_abi.execution.actions")
        action_bindings.append(MoeSwizzleStateActionBinding(
            action.id, action.token_index, action.expert_index, role,
            action.write_values[0], tensor.state_ref, tensor.hbm_binding_ref,
            action.bytes,
        ))
    return MoeSwizzleWorkloadStateABI.create(
        source_ir1_id=ir1.id,
        source_manifest_id=manifest.id,
        source_execution_id=execution.id,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        tensors=tuple(sorted(tensors, key=lambda item: (item.expert_index, _ROLES.index(item.role)))),
        action_bindings=tuple(sorted(action_bindings, key=lambda item: (item.token_index, item.expert_index, _ROLES.index(item.role), item.execution_action_ref))),
    )


def validate_moe_swizzle_workload_state_abi_against(
    abi: MoeSwizzleWorkloadStateABI,
    ir1: IR1,
    execution: MoeScaleExecution,
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
) -> None:
    """Require exact deterministic rebuild from IR1 persistent-state truth."""

    abi.validate("validate_moe_swizzle_workload_state_abi_against.abi")
    expected = build_moe_swizzle_workload_state_abi(
        ir1, execution, spec, oracle,
    )
    if abi != expected:
        raise SchemaError(
            "workload StateABI is not the deterministic persistent-state rebuild",
            path="validate_moe_swizzle_workload_state_abi_against.abi",
        )


__all__ = [
    "build_moe_swizzle_workload_state_abi",
    "validate_moe_swizzle_workload_state_abi_against",
]
