"""Generalized 4-Die MoE execution truth for Swizzle V2 scale points."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import re

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .serde import canonical_digest
from .swizzle_moe_scale import (
    MoeSwizzleScaleOracle,
    MoeSwizzleScaleSpec,
)


MOE_SCALE_EXECUTION_ACTION_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_action/v1alpha1"
MOE_SCALE_EXECUTION_FLOW_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_flow/v1alpha1"
MOE_SCALE_EXECUTION_TERMINAL_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_terminal/v1alpha1"
MOE_SCALE_EXECUTION_CAPACITY_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_capacity/v1alpha1"
MOE_SCALE_EXECUTION_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_execution/v1alpha1"


class MoeScaleExecutionMode(str, Enum):
    INFER_FORWARD = "infer_forward"
    TRAIN_FORWARD = "train_forward"


class MoeScaleExecutionActionKind(str, Enum):
    DMA_IN = "dma_in"
    GEMM = "gemm"
    SWIGLU = "swiglu"
    SEND = "send"
    RECV = "recv"
    WAIT = "wait"
    TAPE_COPY = "tape_copy"


class MoeScaleExecutionFlowRole(str, Enum):
    DISPATCH = "dispatch"
    COMBINE = "combine"


class MoeScaleExecutionTerminalKind(str, Enum):
    COMBINED = "combined"
    TAPE = "tape"


def _semantic(instance: object) -> dict[str, object]:
    return {
        name: getattr(instance, name)
        for name in instance.__dataclass_fields__
        if name not in ("schema_version", "producer_pass", "id")
    }


def _stable(instance: object, prefix: str, version: str, path: str) -> None:
    expected = stable_artifact_id(prefix, _semantic(instance), schema_version=version)
    if getattr(instance, "id") != expected:
        raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


def _digest(value: str, path: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class MoeScaleExecutionAction:
    schema_version: str
    id: str
    order_index: int
    kind: MoeScaleExecutionActionKind
    die_id: int
    token_index: int
    expert_index: int
    role: str
    flow_ref: str | None
    peer_die_id: int | None
    read_values: tuple[str, ...]
    write_values: tuple[str, ...]
    deps: tuple[str, ...]
    bytes: int
    flops: int
    dtype: DType

    @classmethod
    def create(cls, **semantic: object) -> "MoeScaleExecutionAction":
        result = cls(
            MOE_SCALE_EXECUTION_ACTION_SCHEMA_VERSION,
            stable_artifact_id(
                "moe_swizzle_action",
                semantic,
                schema_version=MOE_SCALE_EXECUTION_ACTION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_swizzle_action") -> None:
        if self.schema_version != MOE_SCALE_EXECUTION_ACTION_SCHEMA_VERSION:
            raise SchemaError("unsupported action schema", path=path)
        if type(self.kind) is not MoeScaleExecutionActionKind or self.dtype is not DType.FP16:
            raise SchemaError("action kind/dtype is not exact", path=path)
        for name in ("order_index", "die_id", "token_index", "expert_index", "bytes", "flops"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.die_id > 3 or self.expert_index > 3:
            raise SchemaError("action must remain on EP4/DP4", path=path)
        validate_nonempty(self.role, f"{path}.role")
        for name, values in (("read_values", self.read_values), ("write_values", self.write_values), ("deps", self.deps)):
            if len(values) != len(set(values)) or any(type(item) is not str or not item for item in values):
                raise SchemaError("references must be unique and nonempty", path=f"{path}.{name}")
        transfer = self.kind in (
            MoeScaleExecutionActionKind.SEND,
            MoeScaleExecutionActionKind.RECV,
            MoeScaleExecutionActionKind.WAIT,
        )
        if transfer != (self.flow_ref is not None and self.peer_die_id is not None):
            raise SchemaError("transfer action flow/peer closure drifted", path=path)
        if self.peer_die_id is not None and (self.peer_die_id > 3 or self.peer_die_id == self.die_id):
            raise SchemaError("transfer peer is invalid", path=f"{path}.peer_die_id")
        if self.kind is MoeScaleExecutionActionKind.GEMM:
            if self.flops == 0 or self.bytes != 0:
                raise SchemaError("GEMM must carry only positive FLOPs", path=path)
        elif self.flops != 0:
            raise SchemaError("non-GEMM action cannot claim GEMM FLOPs", path=f"{path}.flops")
        elif self.kind is MoeScaleExecutionActionKind.SWIGLU:
            if self.bytes != 0:
                raise SchemaError("SwiGLU control action has no transfer bytes", path=path)
        elif self.bytes == 0:
            raise SchemaError("data/control action requires positive bytes", path=f"{path}.bytes")
        _stable(self, "moe_swizzle_action", MOE_SCALE_EXECUTION_ACTION_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class MoeScaleExecutionFlow:
    schema_version: str
    id: str
    flow_ref: str
    role: MoeScaleExecutionFlowRole
    token_index: int
    expert_index: int
    source_die_id: int
    destination_die_id: int
    die_path: tuple[int, ...]
    source_value_ref: str
    destination_value_ref: str
    bytes: int
    dtype: DType
    send_action_ref: str
    recv_action_ref: str
    wait_action_ref: str

    @classmethod
    def create(cls, **semantic: object) -> "MoeScaleExecutionFlow":
        result = cls(
            MOE_SCALE_EXECUTION_FLOW_SCHEMA_VERSION,
            stable_artifact_id("moe_swizzle_flow", semantic, schema_version=MOE_SCALE_EXECUTION_FLOW_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_swizzle_flow") -> None:
        if self.schema_version != MOE_SCALE_EXECUTION_FLOW_SCHEMA_VERSION or type(self.role) is not MoeScaleExecutionFlowRole:
            raise SchemaError("unsupported flow schema/role", path=path)
        for name in ("token_index", "expert_index", "source_die_id", "destination_die_id", "bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.expert_index > 3
            or self.source_die_id > 3
            or self.destination_die_id > 3
            or self.source_die_id == self.destination_die_id
            or self.bytes == 0
            or self.dtype is not DType.FP16
            or not self.die_path
            or self.die_path[0] != self.source_die_id
            or self.die_path[-1] != self.destination_die_id
            or len(self.die_path) != len(set(self.die_path))
            or any(item > 3 for item in self.die_path)
        ):
            raise SchemaError("flow geometry is not exact 4-Die traffic", path=path)
        for name in (
            "flow_ref", "source_value_ref", "destination_value_ref", "send_action_ref",
            "recv_action_ref", "wait_action_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        _stable(self, "moe_swizzle_flow", MOE_SCALE_EXECUTION_FLOW_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class MoeScaleExecutionTerminal:
    schema_version: str
    id: str
    kind: MoeScaleExecutionTerminalKind
    token_index: int
    expert_index: int
    die_id: int
    value_ref: str
    producer_action_ref: str
    shape: tuple[int, int]
    bytes: int
    dtype: DType

    @classmethod
    def create(cls, **semantic: object) -> "MoeScaleExecutionTerminal":
        result = cls(
            MOE_SCALE_EXECUTION_TERMINAL_SCHEMA_VERSION,
            stable_artifact_id("moe_swizzle_terminal", semantic, schema_version=MOE_SCALE_EXECUTION_TERMINAL_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_swizzle_terminal") -> None:
        if self.schema_version != MOE_SCALE_EXECUTION_TERMINAL_SCHEMA_VERSION or type(self.kind) is not MoeScaleExecutionTerminalKind:
            raise SchemaError("unsupported terminal schema/kind", path=path)
        for name in ("token_index", "expert_index", "die_id", "bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.expert_index > 3
            or self.die_id > 3
            or len(self.shape) != 2
            or any(type(item) is not int or item <= 0 for item in self.shape)
            or self.bytes != self.shape[0] * self.shape[1] * 2
            or self.dtype is not DType.FP16
        ):
            raise SchemaError("terminal shape/bytes are not exact FP16", path=path)
        validate_nonempty(self.value_ref, f"{path}.value_ref")
        validate_nonempty(self.producer_action_ref, f"{path}.producer_action_ref")
        _stable(self, "moe_swizzle_terminal", MOE_SCALE_EXECUTION_TERMINAL_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class MoeScaleExecutionCapacityResult:
    schema_version: str
    id: str
    required_slots_by_expert: tuple[int, ...]
    configured_slots_by_expert: tuple[int, ...]
    admitted: bool

    @classmethod
    def create(cls, **semantic: object) -> "MoeScaleExecutionCapacityResult":
        result = cls(
            MOE_SCALE_EXECUTION_CAPACITY_SCHEMA_VERSION,
            stable_artifact_id("moe_swizzle_capacity", semantic, schema_version=MOE_SCALE_EXECUTION_CAPACITY_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_swizzle_capacity") -> None:
        if self.schema_version != MOE_SCALE_EXECUTION_CAPACITY_SCHEMA_VERSION:
            raise SchemaError("unsupported capacity schema", path=path)
        if len(self.required_slots_by_expert) != 4 or len(self.configured_slots_by_expert) != 4:
            raise SchemaError("capacity result must cover four experts", path=path)
        for name in ("required_slots_by_expert", "configured_slots_by_expert"):
            values = getattr(self, name)
            for index, value in enumerate(values):
                validate_uint64(value, f"{path}.{name}[{index}]")
                if value == 0:
                    raise SchemaError("capacity slots must be positive", path=f"{path}.{name}[{index}]")
        if type(self.admitted) is not bool or self.admitted != all(
            required <= configured
            for required, configured in zip(
                self.required_slots_by_expert,
                self.configured_slots_by_expert,
                strict=True,
            )
        ):
            raise SchemaError("capacity admission does not match typed slots", path=f"{path}.admitted")
        _stable(self, "moe_swizzle_capacity", MOE_SCALE_EXECUTION_CAPACITY_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class MoeScaleExecution:
    schema_version: str
    producer_pass: str
    id: str
    mode: MoeScaleExecutionMode
    execution_ready: bool
    source_scale_spec_id: str
    source_scale_spec_digest: str
    source_scale_oracle_id: str
    source_scale_oracle_digest: str
    actions: tuple[MoeScaleExecutionAction, ...]
    flows: tuple[MoeScaleExecutionFlow, ...]
    terminals: tuple[MoeScaleExecutionTerminal, ...]
    capacity: MoeScaleExecutionCapacityResult

    @classmethod
    def create(cls, **semantic: object) -> "MoeScaleExecution":
        result = cls(
            MOE_SCALE_EXECUTION_SCHEMA_VERSION,
            "build_moe_swizzle_execution",
            stable_artifact_id("moe_swizzle_execution", semantic, schema_version=MOE_SCALE_EXECUTION_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_swizzle_execution") -> None:
        if (
            self.schema_version != MOE_SCALE_EXECUTION_SCHEMA_VERSION
            or self.producer_pass != "build_moe_swizzle_execution"
            or type(self.mode) is not MoeScaleExecutionMode
            or type(self.execution_ready) is not bool
        ):
            raise SchemaError("unsupported execution schema/producer/mode", path=path)
        for name in ("source_scale_spec_id", "source_scale_oracle_id"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in ("source_scale_spec_digest", "source_scale_oracle_digest"):
            _digest(getattr(self, name), f"{path}.{name}")
        ids = set()
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
            if action.order_index != index or action.id in ids or any(dep not in ids for dep in action.deps):
                raise SchemaError("actions must be unique canonical topological order", path=f"{path}.actions[{index}]")
            ids.add(action.id)
        for index, flow in enumerate(self.flows):
            flow.validate(f"{path}.flows[{index}]")
            action_by_id = {item.id: item for item in self.actions}
            triple = tuple(action_by_id.get(ref) for ref in (flow.send_action_ref, flow.recv_action_ref, flow.wait_action_ref))
            if (
                any(item is None for item in triple)
                or tuple(item.kind for item in triple)  # type: ignore[union-attr]
                != (MoeScaleExecutionActionKind.SEND, MoeScaleExecutionActionKind.RECV, MoeScaleExecutionActionKind.WAIT)
                or any(item.flow_ref != flow.flow_ref for item in triple)  # type: ignore[union-attr]
            ):
                raise SchemaError("flow/action closure drifted", path=f"{path}.flows[{index}]")
        if len({item.id for item in self.flows}) != len(self.flows):
            raise SchemaError("duplicate flow id", path=f"{path}.flows")
        action_by_id = {item.id: item for item in self.actions}
        for index, terminal in enumerate(self.terminals):
            terminal.validate(f"{path}.terminals[{index}]")
            producer = action_by_id.get(terminal.producer_action_ref)
            if producer is None or producer.die_id != terminal.die_id or terminal.value_ref not in producer.write_values:
                raise SchemaError("terminal producer closure drifted", path=f"{path}.terminals[{index}]")
        if len({item.id for item in self.terminals}) != len(self.terminals):
            raise SchemaError("duplicate terminal id", path=f"{path}.terminals")
        self.capacity.validate(f"{path}.capacity")
        _stable(self, "moe_swizzle_execution", MOE_SCALE_EXECUTION_SCHEMA_VERSION, path)

    def validate_against(
        self,
        spec: MoeSwizzleScaleSpec,
        oracle: MoeSwizzleScaleOracle,
        path: str = "moe_swizzle_execution",
    ) -> None:
        self.validate(path)
        spec.validate(f"{path}.spec")
        oracle.validate_against(spec, f"{path}.oracle")
        if (
            self.source_scale_spec_id != spec.id
            or self.source_scale_spec_digest != canonical_digest(spec)
            or self.source_scale_oracle_id != oracle.id
            or self.source_scale_oracle_digest != canonical_digest(oracle)
            or self.execution_ready != self.capacity.admitted
        ):
            raise SchemaError("execution source/readiness provenance drifted", path=path)
        remote = len(oracle.remote_token_indices)
        expected = Counter({
            MoeScaleExecutionActionKind.DMA_IN: 3 * spec.tokens,
            MoeScaleExecutionActionKind.GEMM: 3 * spec.tokens,
            MoeScaleExecutionActionKind.SWIGLU: spec.tokens,
            MoeScaleExecutionActionKind.SEND: 2 * remote,
            MoeScaleExecutionActionKind.RECV: 2 * remote,
            MoeScaleExecutionActionKind.WAIT: 2 * remote,
        })
        if self.mode is MoeScaleExecutionMode.TRAIN_FORWARD:
            expected[MoeScaleExecutionActionKind.TAPE_COPY] = spec.tokens
        if Counter(item.kind for item in self.actions) != expected:
            raise SchemaError("execution action formula drifted", path=f"{path}.actions")
        if (
            Counter(item.role for item in self.flows)
            != Counter({MoeScaleExecutionFlowRole.DISPATCH: remote, MoeScaleExecutionFlowRole.COMBINE: remote})
            or sum(item.bytes for item in self.flows) != oracle.logical_p2p_bytes
            or sum(item.flops for item in self.actions) != oracle.total_expert_gemm_flops
        ):
            raise SchemaError("execution work/flow formula drifted", path=path)
        terminals = Counter(item.kind for item in self.terminals)
        expected_terminals = Counter({MoeScaleExecutionTerminalKind.COMBINED: spec.tokens})
        if self.mode is MoeScaleExecutionMode.TRAIN_FORWARD:
            expected_terminals[MoeScaleExecutionTerminalKind.TAPE] = spec.tokens
        if terminals != expected_terminals:
            raise SchemaError("terminal coverage formula drifted", path=f"{path}.terminals")
        combined_bytes = sum(item.bytes for item in self.terminals if item.kind is MoeScaleExecutionTerminalKind.COMBINED)
        tape_bytes = sum(item.bytes for item in self.terminals if item.kind is MoeScaleExecutionTerminalKind.TAPE)
        if (
            combined_bytes != oracle.combined_terminal_bytes
            or tape_bytes != (
                oracle.train_tape_terminal_bytes
                if self.mode is MoeScaleExecutionMode.TRAIN_FORWARD else 0
            )
            or self.capacity.required_slots_by_expert != oracle.expert_token_counts
            or self.capacity.configured_slots_by_expert != (spec.capacity_per_expert,) * 4
            or not self.capacity.admitted
        ):
            raise SchemaError("terminal/capacity formula drifted", path=path)


__all__ = [
    name
    for name in globals()
    if name.startswith("MoeScaleExecution")
    or name.startswith("MOE_SCALE_EXECUTION")
]
