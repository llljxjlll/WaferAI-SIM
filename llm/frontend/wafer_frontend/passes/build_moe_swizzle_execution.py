"""Build generalized 4-Die MoE action truth from typed scale work."""

from __future__ import annotations

from collections import Counter

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.lite_moe_dp4_execution import LiteMoeDp4ExecutionCase
from ..schema.lite_moe_dp4_train_forward import LiteMoeDp4TrainForward
from ..schema.serde import canonical_digest
from ..schema.swizzle_moe_execution import (
    MoeScaleExecutionAction,
    MoeScaleExecutionActionKind,
    MoeScaleExecutionCapacityResult,
    MoeScaleExecution,
    MoeScaleExecutionMode,
    MoeScaleExecutionFlow,
    MoeScaleExecutionFlowRole,
    MoeScaleExecutionTerminal,
    MoeScaleExecutionTerminalKind,
)
from ..schema.swizzle_moe_scale import MoeSwizzleScaleOracle, MoeSwizzleScaleSpec


def _route(source: int, destination: int) -> tuple[int, ...]:
    """Deterministic X-then-Y route on row-major 2x2 die IDs."""

    source_row, source_column = divmod(source, 2)
    destination_row, destination_column = divmod(destination, 2)
    path = [source]
    column = source_column
    if column != destination_column:
        column = destination_column
        path.append(source_row * 2 + column)
    if source_row != destination_row:
        path.append(destination_row * 2 + column)
    return tuple(path)


def _components(
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
    mode: MoeScaleExecutionMode,
) -> tuple[
    tuple[MoeScaleExecutionAction, ...],
    tuple[MoeScaleExecutionFlow, ...],
    tuple[MoeScaleExecutionTerminal, ...],
    MoeScaleExecutionCapacityResult,
]:
    spec.validate("spec")
    oracle.validate_against(spec, "oracle")
    if type(mode) is not MoeScaleExecutionMode:
        raise SchemaError("must use a typed execution mode", path="mode")
    actions: list[MoeScaleExecutionAction] = []
    flows: list[MoeScaleExecutionFlow] = []
    terminals: list[MoeScaleExecutionTerminal] = []
    token_bytes = spec.hidden_size * 2
    tape_bytes = spec.intermediate_size * 2

    def add(
        kind: MoeScaleExecutionActionKind,
        die: int,
        token: int,
        expert: int,
        role: str,
        *,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        deps: tuple[str, ...] = (),
        bytes: int = 0,
        flops: int = 0,
        flow_ref: str | None = None,
        peer: int | None = None,
    ) -> MoeScaleExecutionAction:
        action = MoeScaleExecutionAction.create(
            order_index=len(actions),
            kind=kind,
            die_id=die,
            token_index=token,
            expert_index=expert,
            role=role,
            flow_ref=flow_ref,
            peer_die_id=peer,
            read_values=reads,
            write_values=writes,
            deps=tuple(dict.fromkeys(deps)),
            bytes=bytes,
            flops=flops,
            dtype=DType.FP16,
        )
        actions.append(action)
        return action

    def transfer(
        role: MoeScaleExecutionFlowRole,
        token: int,
        expert: int,
        source_die: int,
        destination_die: int,
        source_value: str,
        destination_value: str,
        deps: tuple[str, ...],
    ) -> MoeScaleExecutionAction:
        flow_ref = f"moe.scale.{spec.name}.token{token}.{role.value}"
        send = add(
            MoeScaleExecutionActionKind.SEND,
            source_die,
            token,
            expert,
            f"{role.value}.send",
            reads=(source_value,),
            deps=deps,
            bytes=token_bytes,
            flow_ref=flow_ref,
            peer=destination_die,
        )
        recv = add(
            MoeScaleExecutionActionKind.RECV,
            destination_die,
            token,
            expert,
            f"{role.value}.recv",
            writes=(destination_value,),
            deps=(send.id,),
            bytes=token_bytes,
            flow_ref=flow_ref,
            peer=source_die,
        )
        wait = add(
            MoeScaleExecutionActionKind.WAIT,
            destination_die,
            token,
            expert,
            f"{role.value}.wait",
            reads=(destination_value,),
            writes=(destination_value,),
            deps=(recv.id,),
            bytes=token_bytes,
            flow_ref=flow_ref,
            peer=source_die,
        )
        flows.append(MoeScaleExecutionFlow.create(
            flow_ref=flow_ref,
            role=role,
            token_index=token,
            expert_index=expert,
            source_die_id=source_die,
            destination_die_id=destination_die,
            die_path=_route(source_die, destination_die),
            source_value_ref=source_value,
            destination_value_ref=destination_value,
            bytes=token_bytes,
            dtype=DType.FP16,
            send_action_ref=send.id,
            recv_action_ref=recv.id,
            wait_action_ref=wait.id,
        ))
        return wait

    for assignment in spec.trace.assignments:
        token = assignment.token_index
        expert = assignment.expert_index
        source_die = spec.token_source_die_ids[token]
        expert_die = spec.expert_home_die_ids[expert]
        prefix = f"moe.scale.{spec.name}.token{token}.expert{expert}"
        token_value = f"{prefix}.input"
        activation = token_value
        activation_dep: tuple[str, ...] = ()
        if source_die != expert_die:
            activation = f"{prefix}.routed"
            dispatch_wait = transfer(
                MoeScaleExecutionFlowRole.DISPATCH,
                token,
                expert,
                source_die,
                expert_die,
                token_value,
                activation,
                (),
            )
            activation_dep = (dispatch_wait.id,)

        gemm_actions = {}
        for role in ("gate", "up"):
            staging = f"{prefix}.{role}.weight_staging"
            output = f"{prefix}.{role}"
            dma = add(
                MoeScaleExecutionActionKind.DMA_IN,
                expert_die,
                token,
                expert,
                f"{role}.weight",
                writes=(staging,),
                bytes=spec.hidden_size * spec.intermediate_size * 2,
            )
            gemm_actions[role] = add(
                MoeScaleExecutionActionKind.GEMM,
                expert_die,
                token,
                expert,
                role,
                reads=(activation, staging),
                writes=(output,),
                deps=(*activation_dep, dma.id),
                flops=2 * spec.hidden_size * spec.intermediate_size,
            )
        swiglu_value = f"{prefix}.swiglu"
        swiglu = add(
            MoeScaleExecutionActionKind.SWIGLU,
            expert_die,
            token,
            expert,
            "swiglu",
            reads=(f"{prefix}.gate", f"{prefix}.up"),
            writes=(swiglu_value,),
            deps=(gemm_actions["gate"].id, gemm_actions["up"].id),
        )
        if mode is MoeScaleExecutionMode.TRAIN_FORWARD:
            tape_value = f"{prefix}.tape"
            tape = add(
                MoeScaleExecutionActionKind.TAPE_COPY,
                expert_die,
                token,
                expert,
                "train_forward.tape",
                reads=(swiglu_value,),
                writes=(tape_value,),
                deps=(swiglu.id,),
                bytes=tape_bytes,
            )
            terminals.append(MoeScaleExecutionTerminal.create(
                kind=MoeScaleExecutionTerminalKind.TAPE,
                token_index=token,
                expert_index=expert,
                die_id=expert_die,
                value_ref=tape_value,
                producer_action_ref=tape.id,
                shape=(1, spec.intermediate_size),
                bytes=tape_bytes,
                dtype=DType.FP16,
            ))
        down_staging = f"{prefix}.down.weight_staging"
        down_dma = add(
            MoeScaleExecutionActionKind.DMA_IN,
            expert_die,
            token,
            expert,
            "down.weight",
            writes=(down_staging,),
            bytes=spec.intermediate_size * spec.hidden_size * 2,
        )
        down_value = f"{prefix}.down"
        down = add(
            MoeScaleExecutionActionKind.GEMM,
            expert_die,
            token,
            expert,
            "down",
            reads=(swiglu_value, down_staging),
            writes=(down_value,),
            deps=(swiglu.id, down_dma.id),
            flops=2 * spec.intermediate_size * spec.hidden_size,
        )
        combined_value = down_value
        producer = down
        if source_die != expert_die:
            combined_value = f"{prefix}.combined"
            producer = transfer(
                MoeScaleExecutionFlowRole.COMBINE,
                token,
                expert,
                expert_die,
                source_die,
                down_value,
                combined_value,
                (down.id,),
            )
        terminals.append(MoeScaleExecutionTerminal.create(
            kind=MoeScaleExecutionTerminalKind.COMBINED,
            token_index=token,
            expert_index=expert,
            die_id=source_die,
            value_ref=combined_value,
            producer_action_ref=producer.id,
            shape=(1, spec.hidden_size),
            bytes=token_bytes,
            dtype=DType.FP16,
        ))

    capacity = MoeScaleExecutionCapacityResult.create(
        required_slots_by_expert=oracle.expert_token_counts,
        configured_slots_by_expert=(spec.capacity_per_expert,) * 4,
        admitted=all(item <= spec.capacity_per_expert for item in oracle.expert_token_counts),
    )
    return tuple(actions), tuple(flows), tuple(terminals), capacity


def build_moe_swizzle_execution(
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
    mode: MoeScaleExecutionMode,
) -> MoeScaleExecution:
    actions, flows, terminals, capacity = _components(spec, oracle, mode)
    result = MoeScaleExecution.create(
        mode=mode,
        # CAPACITY_PROBE is a validation/economic-training role, not an
        # artificial execution rejection. Exact typed capacity decides.
        execution_ready=capacity.admitted,
        source_scale_spec_id=spec.id,
        source_scale_spec_digest=canonical_digest(spec),
        source_scale_oracle_id=oracle.id,
        source_scale_oracle_digest=canonical_digest(oracle),
        actions=actions,
        flows=flows,
        terminals=terminals,
        capacity=capacity,
    )
    validate_moe_swizzle_execution(result, spec, oracle)
    return result


def validate_moe_swizzle_execution(
    result: MoeScaleExecution,
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
) -> None:
    result.validate_against(spec, oracle)
    expected = _components(spec, oracle, result.mode)
    if (result.actions, result.flows, result.terminals, result.capacity) != expected:
        raise SchemaError("execution does not exactly rebuild production scale truth", path="execution")


def validate_moe_swizzle_c0_execution(
    infer: MoeScaleExecution,
    train_forward: MoeScaleExecution,
    legacy_forward: LiteMoeDp4ExecutionCase,
    legacy_train_forward: LiteMoeDp4TrainForward,
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
) -> None:
    """Cross-check new C0 work/quotients against frozen legacy production."""

    legacy_forward.validate("legacy_forward")
    legacy_train_forward.validate("legacy_train_forward")
    if legacy_train_forward.forward != legacy_forward:
        raise SchemaError("legacy train-forward source drifted", path="legacy_train_forward")
    validate_moe_swizzle_execution(infer, spec, oracle)
    validate_moe_swizzle_execution(train_forward, spec, oracle)
    if infer.mode is not MoeScaleExecutionMode.INFER_FORWARD or train_forward.mode is not MoeScaleExecutionMode.TRAIN_FORWARD:
        raise SchemaError("C0 cross-validator requires infer and train-forward", path="execution")
    legacy_flows = tuple(sorted(
        (
            item.token_index,
            item.expert_index,
            item.source_die_id,
            item.destination_die_id,
            item.bytes,
        )
        for item in legacy_forward.projection.flows
    ))
    generalized_flows = tuple(sorted(
        (
            item.token_index,
            item.expert_index,
            item.source_die_id,
            item.destination_die_id,
            item.bytes,
        )
        for item in infer.flows
    ))
    if (
        spec.name != "C0"
        or (len(infer.actions), len(legacy_forward.global_dag.actions)) != (92, 92)
        or legacy_flows != generalized_flows
        or len(train_forward.actions) != len(infer.actions) + len(legacy_train_forward.tape_copies)
        or len(legacy_train_forward.tape_copies) != spec.tokens
        or sum(item.bytes for item in legacy_train_forward.tape_copies)
        != oracle.train_tape_terminal_bytes
        or Counter(item.kind for item in infer.actions)
        != Counter({
            MoeScaleExecutionActionKind.DMA_IN: 24,
            MoeScaleExecutionActionKind.GEMM: 24,
            MoeScaleExecutionActionKind.SWIGLU: 8,
            MoeScaleExecutionActionKind.SEND: 12,
            MoeScaleExecutionActionKind.RECV: 12,
            MoeScaleExecutionActionKind.WAIT: 12,
        })
    ):
        raise SchemaError("generalized C0 execution drifted from legacy production", path="execution")


__all__ = [
    "build_moe_swizzle_execution",
    "validate_moe_swizzle_c0_execution",
    "validate_moe_swizzle_execution",
]
