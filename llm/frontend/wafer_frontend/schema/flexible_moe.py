"""Opt-in arbitrary-rectangle MoE timing carrier.

This v2 schema intentionally lives beside the frozen EP4/LiteMoE schemas.  It
describes a deterministic executable baseline without changing their wire
format or stable IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import heapq
import hashlib
import struct

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .rect_mesh import RectMeshSpec
from .serde import canonical_digest


FLEXIBLE_MOE_TRACE_SCHEMA_VERSION = "wafer_frontend.flexible_moe_trace/v2alpha1"
FLEXIBLE_MOE_SPEC_SCHEMA_VERSION = "wafer_frontend.flexible_moe_spec/v2alpha1"
FLEXIBLE_MOE_PLAN_SCHEMA_VERSION = "wafer_frontend.flexible_moe_plan/v2alpha1"
_FP32_ONE_BITS = 0x3F800000


class FlexibleMoeMode(str, Enum):
    INFERENCE = "inference"
    TRAIN = "train"


class MoeRectFlowStage(str, Enum):
    DISPATCH = "dispatch"
    COMBINE = "combine"
    BACKWARD_GRADIENT = "backward_gradient"
    BACKWARD_DX = "backward_dx"
    GATE_ALL_REDUCE = "gate_all_reduce"


class MoeRectActionKind(str, Enum):
    STATE_LOAD = "state_load"
    GATE = "gate"
    PACK = "pack"
    SEND = "send"
    RECV = "recv"
    WAIT = "wait"
    EXPERT_FORWARD = "expert_forward"
    WEIGHTED_COMBINE = "weighted_combine"
    EXPERT_DGRAD = "expert_dgrad"
    EXPERT_WGRAD = "expert_wgrad"
    COMBINE_BACKWARD = "combine_backward"
    SHARED_DCOMBINED_IMPORT = "shared_dcombined_import"
    SCORE_WEIGHT_BACKWARD_PRE_DISPATCH = "score_weight_backward_pre_dispatch"
    GATE_WGRAD = "gate_wgrad"
    GATE_GRADIENT_LOCAL_REDUCE = "gate_gradient_local_reduce"
    GATE_GRADIENT_ALL_REDUCE = "gate_gradient_all_reduce"
    EXPERT_SGD = "expert_sgd"
    GATE_SGD = "gate_sgd"
    STATE_STORE = "state_store"


class MoeRectStateRole(str, Enum):
    EXPERT_PARAMETER = "expert_parameter"
    EXPERT_GRADIENT = "expert_gradient"
    GATE_PARAMETER = "gate_parameter"
    GATE_GRADIENT = "gate_gradient"


@dataclass(frozen=True, slots=True)
class MoeRectTraceAssignment:
    token_index: int
    source_rank: int
    expert_index: int
    expert_home_rank: int
    slot_index: int
    gate_weight_f32_bits: int = _FP32_ONE_BITS

    def validate(self, path: str = "moe_rect_assignment") -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.gate_weight_f32_bits > 0xFFFFFFFF:
            raise SchemaError("must contain FP32 bits", path=f"{path}.gate_weight_f32_bits")


@dataclass(frozen=True, slots=True)
class MoeRectStaticTrace:
    schema_version: str
    producer_pass: str
    id: str
    token_count: int
    expert_count: int
    capacity_per_expert: int
    assignments: tuple[MoeRectTraceAssignment, ...]
    expert_histogram: tuple[int, ...]

    @classmethod
    def create(
        cls,
        *,
        token_count: int,
        expert_count: int,
        capacity_per_expert: int,
        assignments: tuple[MoeRectTraceAssignment, ...],
    ) -> "MoeRectStaticTrace":
        histogram = tuple(
            sum(item.expert_index == expert for item in assignments)
            for expert in range(expert_count)
        )
        semantic = {
            "token_count": token_count,
            "expert_count": expert_count,
            "capacity_per_expert": capacity_per_expert,
            "assignments": assignments,
            "expert_histogram": histogram,
        }
        result = cls(
            FLEXIBLE_MOE_TRACE_SCHEMA_VERSION,
            "flexible_moe_static_trace",
            stable_artifact_id(
                "flexible_moe_trace",
                semantic,
                schema_version=FLEXIBLE_MOE_TRACE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "token_count", "expert_count", "capacity_per_expert",
                "assignments", "expert_histogram",
            )
        }

    def validate(self, path: str = "flexible_moe_trace") -> None:
        if (
            self.schema_version != FLEXIBLE_MOE_TRACE_SCHEMA_VERSION
            or self.producer_pass != "flexible_moe_static_trace"
        ):
            raise SchemaError("unsupported trace schema/producer", path=path)
        for name in ("token_count", "expert_count", "capacity_per_expert"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")
        if type(self.assignments) is not tuple or len(self.assignments) != self.token_count:
            raise SchemaError(
                "top-k=1 requires one assignment per token",
                path=f"{path}.assignments",
            )
        slots: list[list[int]] = [[] for _ in range(self.expert_count)]
        for index, assignment in enumerate(self.assignments):
            if type(assignment) is not MoeRectTraceAssignment:
                raise SchemaError("must be a typed assignment", path=f"{path}.assignments[{index}]")
            assignment.validate(f"{path}.assignments[{index}]")
            if assignment.token_index != index:
                raise SchemaError("tokens must be canonical and gap-free", path=f"{path}.assignments")
            if assignment.expert_index >= self.expert_count:
                raise SchemaError("expert lies outside the trace", path=f"{path}.assignments[{index}].expert_index")
            slots[assignment.expert_index].append(assignment.slot_index)
        for expert, observed in enumerate(slots):
            if tuple(observed) != tuple(range(len(observed))):
                raise SchemaError("expert slots must be canonical and gap-free", path=f"{path}.assignments")
            if len(observed) > self.capacity_per_expert:
                raise SchemaError("expert capacity exceeded; token drop is forbidden", path=f"{path}.assignments")
        expected_histogram = tuple(len(item) for item in slots)
        if self.expert_histogram != expected_histogram:
            raise SchemaError("expert histogram drifted", path=f"{path}.expert_histogram")
        expected = stable_artifact_id(
            "flexible_moe_trace", self._semantic(),
            schema_version=FLEXIBLE_MOE_TRACE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMoeLimits:
    max_actions: int = 1_000_000
    max_flows: int = 400_000
    max_state_bindings: int = 400
    max_sessions_per_rank_wave: int = 3
    max_records: int = 1_000_000
    max_artifact_file_bytes: int = 64 * 1024 * 1024

    def validate(self, path: str = "flexible_moe_limits") -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class FlexibleMoeSpec:
    schema_version: str
    producer_pass: str
    id: str
    mesh: RectMeshSpec
    mode: FlexibleMoeMode
    hidden_size: int
    intermediate_size: int
    expert_count: int
    expert_parallel_degree: int
    top_k: int
    trace_mode: str
    trace: MoeRectStaticTrace
    limits: FlexibleMoeLimits
    expert_dtype: DType
    combine_dtype: DType
    token_drop: bool

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleMoeSpec":
        result = cls(
            FLEXIBLE_MOE_SPEC_SCHEMA_VERSION,
            "flexible_moe_spec",
            stable_artifact_id(
                "flexible_moe_spec", semantic,
                schema_version=FLEXIBLE_MOE_SPEC_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "flexible_moe_spec") -> None:
        if self.schema_version != FLEXIBLE_MOE_SPEC_SCHEMA_VERSION or self.producer_pass != "flexible_moe_spec":
            raise SchemaError("unsupported spec schema/producer", path=path)
        if type(self.mesh) is not RectMeshSpec:
            raise SchemaError("must carry RectMeshSpec", path=f"{path}.mesh")
        self.mesh.validate(f"{path}.mesh")
        if type(self.mode) is not FlexibleMoeMode:
            raise SchemaError("must use a typed mode", path=f"{path}.mode")
        for name in ("hidden_size", "intermediate_size", "expert_count", "expert_parallel_degree", "top_k"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")
        if self.expert_count != self.mesh.rank_count or self.expert_parallel_degree != self.mesh.rank_count:
            raise SchemaError("v2 requires EP=expert_count=Mesh ranks", path=path)
        if self.top_k != 1:
            raise SchemaError("v2 requires top-k=1", path=f"{path}.top_k")
        if self.trace_mode != "static":
            raise SchemaError("v2 requires a static trace", path=f"{path}.trace_mode")
        if type(self.trace) is not MoeRectStaticTrace:
            raise SchemaError("must carry a typed static trace", path=f"{path}.trace")
        self.trace.validate(f"{path}.trace")
        if self.trace.expert_count != self.expert_count:
            raise SchemaError("trace expert count drifted", path=f"{path}.trace")
        for index, assignment in enumerate(self.trace.assignments):
            if assignment.source_rank >= self.mesh.rank_count:
                raise SchemaError("source rank lies outside Mesh", path=f"{path}.trace.assignments[{index}].source_rank")
            if assignment.expert_home_rank != assignment.expert_index:
                raise SchemaError("one expert per Die requires home rank=expert index", path=f"{path}.trace.assignments[{index}].expert_home_rank")
        if type(self.limits) is not FlexibleMoeLimits:
            raise SchemaError("must carry typed limits", path=f"{path}.limits")
        self.limits.validate(f"{path}.limits")
        if self.expert_dtype is not DType.FP16 or self.combine_dtype is not DType.FP32:
            raise SchemaError("v2 requires FP16 experts and FP32 combine", path=path)
        if self.token_drop is not False:
            raise SchemaError("token drop is forbidden", path=f"{path}.token_drop")
        expected = stable_artifact_id(
            "flexible_moe_spec", self._semantic(),
            schema_version=FLEXIBLE_MOE_SPEC_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate_against_mesh(
        self,
        mesh: RectMeshSpec,
        path: str = "flexible_moe_spec",
    ) -> None:
        self.validate(path)
        mesh.validate(f"{path}.requested_mesh")
        if self.mesh != mesh:
            raise SchemaError("spec belongs to another Mesh", path=f"{path}.mesh")


@dataclass(frozen=True, slots=True)
class MoeRectFlow:
    id: str
    stage: MoeRectFlowStage
    source_rank: int
    destination_rank: int
    wave_index: int
    assignment_refs: tuple[str, ...]
    logical_bytes: int
    die_path: tuple[int, ...]

    @classmethod
    def create(cls, **semantic: object) -> "MoeRectFlow":
        result = cls(
            stable_artifact_id("flexible_moe_flow", semantic, schema_version=FLEXIBLE_MOE_PLAN_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_rect_flow") -> None:
        if type(self.stage) is not MoeRectFlowStage:
            raise SchemaError("must use a typed stage", path=f"{path}.stage")
        for name in ("source_rank", "destination_rank", "wave_index", "logical_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.source_rank == self.destination_rank or self.wave_index == 0 or self.logical_bytes == 0:
            raise SchemaError("remote flow/wave/bytes must be nonzero", path=path)
        if not self.assignment_refs or len(set(self.assignment_refs)) != len(self.assignment_refs):
            raise SchemaError("must contain unique assignment refs", path=f"{path}.assignment_refs")
        if len(self.die_path) < 2 or self.die_path[0] != self.source_rank or self.die_path[-1] != self.destination_rank:
            raise SchemaError("die path endpoints do not match flow", path=f"{path}.die_path")


@dataclass(frozen=True, slots=True)
class MoeRectAction:
    id: str
    rank: int
    kind: MoeRectActionKind
    deps: tuple[str, ...]
    assignment_refs: tuple[str, ...] = ()
    flow_ref: str | None = None
    logical_bytes: int = 0
    flops: int = 0
    state_refs: tuple[str, ...] = ()

    @classmethod
    def create(cls, **semantic: object) -> "MoeRectAction":
        result = cls(
            stable_artifact_id("flexible_moe_action", semantic, schema_version=FLEXIBLE_MOE_PLAN_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_rect_action") -> None:
        validate_uint64(self.rank, f"{path}.rank")
        if type(self.kind) is not MoeRectActionKind:
            raise SchemaError("must use a typed action kind", path=f"{path}.kind")
        for name in ("deps", "assignment_refs", "state_refs"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise SchemaError("contains duplicate refs", path=f"{path}.{name}")
            for index, value in enumerate(values):
                validate_nonempty(value, f"{path}.{name}[{index}]")
        if self.flow_ref is not None:
            validate_nonempty(self.flow_ref, f"{path}.flow_ref")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.flops, f"{path}.flops")


@dataclass(frozen=True, slots=True)
class MoeRectStateBinding:
    id: str
    role: MoeRectStateRole
    owner_rank: int
    expert_index: int | None
    dtype: DType
    size_bytes: int
    persistent: bool

    @classmethod
    def create(cls, **semantic: object) -> "MoeRectStateBinding":
        result = cls(
            stable_artifact_id("flexible_moe_state", semantic, schema_version=FLEXIBLE_MOE_PLAN_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_rect_state") -> None:
        if type(self.role) is not MoeRectStateRole:
            raise SchemaError("must use typed state role", path=f"{path}.role")
        validate_uint64(self.owner_rank, f"{path}.owner_rank")
        if self.expert_index is not None:
            validate_uint64(self.expert_index, f"{path}.expert_index")
        if type(self.dtype) is not DType:
            raise SchemaError("must use typed dtype", path=f"{path}.dtype")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes == 0:
            raise SchemaError("must be positive", path=f"{path}.size_bytes")
        if type(self.persistent) is not bool:
            raise SchemaError("must be bool", path=f"{path}.persistent")


@dataclass(frozen=True, slots=True)
class MoeRectGateAllReduce:
    participant_ranks: tuple[int, ...]
    logical_bytes_per_rank: int
    wave_count: int
    max_sessions_per_rank_wave: int

    def validate(self, path: str = "moe_rect_gate_ar") -> None:
        if self.participant_ranks != tuple(range(len(self.participant_ranks))):
            raise SchemaError("participants must be canonical", path=f"{path}.participant_ranks")
        for name in ("logical_bytes_per_rank", "wave_count", "max_sessions_per_rank_wave"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        expected_waves = max(0, len(self.participant_ranks) - 1)
        if self.wave_count != expected_waves:
            raise SchemaError("must use cyclic-delta waves", path=f"{path}.wave_count")
        expected_sessions = 0 if len(self.participant_ranks) == 1 else 2
        if self.max_sessions_per_rank_wave != expected_sessions:
            raise SchemaError("session witness drifted", path=f"{path}.max_sessions_per_rank_wave")


_SIGNED_ROUTER_SOURCE_SCHEMA_VERSION = (
    "wafer_frontend.flexible_moe_signed_top1_train_source/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class MoeRectSignedTop1TrainSource:
    """Explicit new score model and named shared gradient import for old P2.

    This source signs an 80B INT32 route blob. It does *not* establish the
    claimed Dense shared backward BufferABI; a later full timeline must bind
    that action/manifest to a true compute output before simulation.
    """

    id: str
    version: str
    original_spec_ref: str
    original_trace_ref: str
    dynamic_case_ref: str
    shared_dcombined_producer_ref: str
    shared_full_model_manifest_ref: str
    route_rows: int
    route_bytes: int
    route_sha256: str
    route_words: tuple[int, ...]

    @classmethod
    def create(cls, spec: FlexibleMoeSpec, *,
               dynamic_case_ref: str,
               shared_dcombined_producer_ref: str,
               shared_full_model_manifest_ref: str):
        spec.validate("signed_top1_router.source_spec")
        if (spec.mode is not FlexibleMoeMode.TRAIN
                or spec.mesh.rank_count != 2
                or any(assignment.source_rank != 0
                       or assignment.gate_weight_f32_bits != _FP32_ONE_BITS
                       for assignment in spec.trace.assignments)):
            raise SchemaError(
                "bounded signed top1 source requires original TRAIN EP2, rank0 tokens and unfalsified gate=ONE trace",
                path="signed_top1_router.source_spec",
            )
        for name, ref in (("dynamic_case_ref", dynamic_case_ref),
                          ("shared_dcombined_producer_ref",
                           shared_dcombined_producer_ref),
                          ("shared_full_model_manifest_ref",
                           shared_full_model_manifest_ref)):
            validate_nonempty(ref, f"signed_top1_router.{name}")
        if (dynamic_case_ref in (spec.id, spec.trace.id)
                or shared_dcombined_producer_ref ==
                   shared_full_model_manifest_ref):
            raise SchemaError("new scored case and shared producer need distinct source identities",
                              path="signed_top1_router.source")
        words = tuple(value for assignment in spec.trace.assignments
                      for value in (assignment.token_index,
                                    assignment.source_rank,
                                    assignment.expert_index,
                                    assignment.expert_home_rank,
                                    assignment.slot_index))
        if any(type(value) is not int or not (0 <= value <= 0x7FFFFFFF)
               for value in words):
            raise SchemaError("five route fields must be physical nonnegative INT32",
                              path="signed_top1_router.route_words")
        blob = struct.pack("<" + "I" * len(words), *words)
        semantic = {
            "version": _SIGNED_ROUTER_SOURCE_SCHEMA_VERSION,
            "original_spec_ref": spec.id,
            "original_trace_ref": spec.trace.id,
            "dynamic_case_ref": dynamic_case_ref,
            "shared_dcombined_producer_ref": shared_dcombined_producer_ref,
            "shared_full_model_manifest_ref": shared_full_model_manifest_ref,
            "route_rows": spec.trace.token_count,
            "route_bytes": len(blob),
            "route_sha256": hashlib.sha256(blob).hexdigest(),
            "route_words": words,
        }
        source = cls(
            stable_artifact_id("moe_signed_top1_train_source", semantic,
                               schema_version=_SIGNED_ROUTER_SOURCE_SCHEMA_VERSION),
            **semantic,
        )
        source.validate_against(spec)
        return source

    @property
    def route_blob(self) -> bytes:
        return struct.pack("<" + "I" * len(self.route_words),
                           *self.route_words)

    def validate_against(self, spec: FlexibleMoeSpec) -> None:
        spec.validate("signed_top1_router.source_spec")
        if (self.version != _SIGNED_ROUTER_SOURCE_SCHEMA_VERSION
                or spec.mode is not FlexibleMoeMode.TRAIN
                or spec.mesh.rank_count != 2
                or self.original_spec_ref != spec.id
                or self.original_trace_ref != spec.trace.id
                or not self.dynamic_case_ref
                or self.dynamic_case_ref in (spec.id, spec.trace.id)
                or not self.shared_dcombined_producer_ref
                or not self.shared_full_model_manifest_ref
                or self.shared_dcombined_producer_ref ==
                   self.shared_full_model_manifest_ref
                or any(assignment.source_rank != 0
                       or assignment.gate_weight_f32_bits != _FP32_ONE_BITS
                       for assignment in spec.trace.assignments)):
            raise SchemaError("signed router TRAIN source/model/shared producer must preserve original EP2 spec",
                              path="signed_top1_router")
        expected = tuple(value for assignment in spec.trace.assignments
                         for value in (assignment.token_index,
                                       assignment.source_rank,
                                       assignment.expert_index,
                                       assignment.expert_home_rank,
                                       assignment.slot_index))
        if (any(type(value) is not int or not (0 <= value <= 0x7FFFFFFF)
                for value in self.route_words)
                or self.route_words != expected
                or self.route_rows != spec.trace.token_count
                or self.route_bytes != 5 * 4 * self.route_rows
                or self.route_sha256 !=
                   hashlib.sha256(self.route_blob).hexdigest()
                or not any(byte != 0 for byte in self.route_blob)):
            raise SchemaError("signed router route table must be one complete nonzero INT32 P2 assignment blob",
                              path="signed_top1_router.route")
        semantic = {name: getattr(self, name) for name in
                    self.__dataclass_fields__ if name != "id"}
        if self.id != stable_artifact_id(
                "moe_signed_top1_train_source", semantic,
                schema_version=_SIGNED_ROUTER_SOURCE_SCHEMA_VERSION):
            raise SchemaError("signed router source artifact identity drifted",
                              path="signed_top1_router.id")


@dataclass(frozen=True, slots=True)
class FlexibleMoeExecutablePlan:
    schema_version: str
    producer_pass: str
    id: str
    source_spec_id: str
    source_spec_digest: str
    mesh_digest: str
    actions: tuple[MoeRectAction, ...]
    flows: tuple[MoeRectFlow, ...]
    state_bindings: tuple[MoeRectStateBinding, ...]
    gate_all_reduce: MoeRectGateAllReduce | None
    terminal_action_refs: tuple[str, ...]
    symbolic_record_count: int
    symbolic_file_bytes: int
    timing_execution: bool
    functional_execution: bool

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def flow_count(self) -> int:
        return len(self.flows)

    @property
    def state_binding_count(self) -> int:
        return len(self.state_bindings)

    @property
    def wave_count(self) -> int:
        return len({(item.stage, item.wave_index) for item in self.flows})

    @property
    def max_sessions_per_rank_wave(self) -> int:
        sessions: dict[tuple[MoeRectFlowStage, int, int], int] = {}
        for flow in self.flows:
            for rank in (flow.source_rank, flow.destination_rank):
                key = (flow.stage, flow.wave_index, rank)
                sessions[key] = sessions.get(key, 0) + 1
        return max(sessions.values(), default=0)

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleMoeExecutablePlan":
        result = cls(
            FLEXIBLE_MOE_PLAN_SCHEMA_VERSION,
            "compile_flexible_moe_baseline",
            stable_artifact_id("flexible_moe_plan", semantic, schema_version=FLEXIBLE_MOE_PLAN_SCHEMA_VERSION),
            **semantic,
        )
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate_against(
        self, spec: FlexibleMoeSpec, path: str = "flexible_moe_plan", *,
        signed_source: MoeRectSignedTop1TrainSource | None = None,
    ) -> None:
        spec.validate(f"{path}.spec")
        if self.schema_version != FLEXIBLE_MOE_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported plan schema/producer", path=path)
        if self.producer_pass == "compile_flexible_moe_baseline":
            if signed_source is not None:
                raise SchemaError("baseline plan cannot carry a signed score source", path=path)
            if any(action.kind in (
                    MoeRectActionKind.SHARED_DCOMBINED_IMPORT,
                    MoeRectActionKind.SCORE_WEIGHT_BACKWARD_PRE_DISPATCH,
                ) for action in self.actions):
                raise SchemaError("baseline cannot carry versioned score gradient actions", path=f"{path}.actions")
        elif self.producer_pass == "compile_flexible_moe_signed_top1_train_source":
            if type(signed_source) is not MoeRectSignedTop1TrainSource:
                raise SchemaError("signed plan requires exact versioned source binding", path=path)
            signed_source.validate_against(spec)
        else:
            raise SchemaError("unsupported plan schema/producer", path=path)
        if (self.source_spec_id, self.source_spec_digest, self.mesh_digest) != (spec.id, spec.digest, spec.mesh.digest):
            raise SchemaError("source provenance drifted", path=path)
        action_index: dict[str, MoeRectAction] = {}
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
            if action.id in action_index or action.rank >= spec.mesh.rank_count:
                raise SchemaError("duplicate action or rank outside Mesh", path=f"{path}.actions")
            action_index[action.id] = action
        for action in self.actions:
            if any(dep not in action_index for dep in action.deps):
                raise SchemaError("action dependency is missing", path=f"{path}.actions")
        indegree = {item.id: len(item.deps) for item in self.actions}
        dependents: dict[str, list[str]] = {item.id: [] for item in self.actions}
        for item in self.actions:
            for dep in item.deps:
                dependents[dep].append(item.id)
        ready = [ref for ref, count in indegree.items() if count == 0]
        heapq.heapify(ready)
        visited = 0
        while ready:
            ref = heapq.heappop(ready)
            visited += 1
            for target in dependents[ref]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(ready, target)
        if visited != len(self.actions):
            raise SchemaError("action graph contains a cycle", path=f"{path}.actions")
        assignment_ids = {f"assignment.{item.token_index}" for item in spec.trace.assignments}
        flow_index: dict[str, MoeRectFlow] = {}
        sessions: dict[tuple[MoeRectFlowStage, int, int], int] = {}
        by_assignment_stage: dict[tuple[str, MoeRectFlowStage], MoeRectFlow] = {}
        for index, flow in enumerate(self.flows):
            flow.validate(f"{path}.flows[{index}]")
            if flow.id in flow_index:
                raise SchemaError("duplicate flow", path=f"{path}.flows")
            flow_index[flow.id] = flow
            expected_wave = (flow.destination_rank - flow.source_rank) % spec.mesh.rank_count
            if flow.wave_index != expected_wave or flow.wave_index == 0:
                raise SchemaError("flow is not in its cyclic-delta wave", path=f"{path}.flows[{index}].wave_index")
            source_x, source_y = spec.mesh.coordinate(flow.source_rank)
            destination_x, destination_y = spec.mesh.coordinate(flow.destination_rank)
            expected_path = [flow.source_rank]
            while source_x != destination_x:
                source_x += 1 if destination_x > source_x else -1
                expected_path.append(spec.mesh.rank(source_y, source_x))
            while source_y != destination_y:
                source_y += 1 if destination_y > source_y else -1
                expected_path.append(spec.mesh.rank(source_y, source_x))
            if flow.die_path != tuple(expected_path):
                raise SchemaError("flow does not use the exact X-first route", path=f"{path}.flows[{index}].die_path")
            if flow.stage is MoeRectFlowStage.GATE_ALL_REDUCE:
                if (
                    flow.source_rank > 0
                    and flow.destination_rank == (flow.source_rank - 1) // 2
                ):
                    expected_ref = f"gate_gradient.reduce.rank.{flow.source_rank}"
                elif (
                    flow.destination_rank > 0
                    and flow.source_rank == (flow.destination_rank - 1) // 2
                ):
                    expected_ref = f"gate_gradient.broadcast.rank.{flow.destination_rank}"
                else:
                    raise SchemaError(
                        "gate AllReduce flow is not a row-major binary-tree edge",
                        path=f"{path}.flows[{index}]",
                    )
                if flow.assignment_refs != (expected_ref,):
                    raise SchemaError(
                        "gate AllReduce flow source witness drifted",
                        path=f"{path}.flows[{index}].assignment_refs",
                    )
            else:
                for ref in flow.assignment_refs:
                    if ref not in assignment_ids or (ref, flow.stage) in by_assignment_stage:
                        raise SchemaError("flow assignment coverage is invalid", path=f"{path}.flows[{index}].assignment_refs")
                    by_assignment_stage[(ref, flow.stage)] = flow
            for rank in (flow.source_rank, flow.destination_rank):
                key = (flow.stage, flow.wave_index, rank)
                sessions[key] = sessions.get(key, 0) + 1
        if max(sessions.values(), default=0) > 2 or max(sessions.values(), default=0) > spec.limits.max_sessions_per_rank_wave:
            raise SchemaError("runtime session budget exceeded", path=f"{path}.flows")
        gate_pairs = {
            (item.source_rank, item.destination_rank)
            for item in self.flows
            if item.stage is MoeRectFlowStage.GATE_ALL_REDUCE
        }
        expected_gate_pairs = (
            {
                pair
                for rank in range(1, spec.mesh.rank_count)
                for pair in (
                    (rank, (rank - 1) // 2),
                    ((rank - 1) // 2, rank),
                )
            }
            if spec.mode is FlexibleMoeMode.TRAIN else set()
        )
        if gate_pairs != expected_gate_pairs:
            raise SchemaError(
                "gate AllReduce must cover exact binary-tree reduce/broadcast pairs",
                path=f"{path}.flows",
            )
        expected_stages = (
            (MoeRectFlowStage.DISPATCH, MoeRectFlowStage.COMBINE)
            if spec.mode is FlexibleMoeMode.INFERENCE
            else (
                MoeRectFlowStage.DISPATCH,
                MoeRectFlowStage.COMBINE,
                MoeRectFlowStage.BACKWARD_GRADIENT,
                MoeRectFlowStage.BACKWARD_DX,
            )
        )
        for assignment in spec.trace.assignments:
            if assignment.source_rank == assignment.expert_home_rank:
                continue
            ref = f"assignment.{assignment.token_index}"
            if any((ref, stage) not in by_assignment_stage for stage in expected_stages):
                raise SchemaError("remote assignment lacks route closure", path=f"{path}.flows")
            dispatch = by_assignment_stage[(ref, MoeRectFlowStage.DISPATCH)]
            combine = by_assignment_stage[(ref, MoeRectFlowStage.COMBINE)]
            if (dispatch.source_rank, dispatch.destination_rank) != (combine.destination_rank, combine.source_rank):
                raise SchemaError("dispatch/combine routes are not reverse-closed", path=f"{path}.flows")
            if spec.mode is FlexibleMoeMode.TRAIN:
                backward = by_assignment_stage[(ref, MoeRectFlowStage.BACKWARD_GRADIENT)]
                dx = by_assignment_stage[(ref, MoeRectFlowStage.BACKWARD_DX)]
                if (backward.source_rank, backward.destination_rank) != (dispatch.source_rank, dispatch.destination_rank) or (dx.source_rank, dx.destination_rank) != (combine.source_rank, combine.destination_rank):
                    raise SchemaError("four-way backward route closure drifted", path=f"{path}.flows")
        for index, state in enumerate(self.state_bindings):
            state.validate(f"{path}.state_bindings[{index}]")
            if state.owner_rank >= spec.mesh.rank_count or (state.expert_index is not None and state.owner_rank != state.expert_index):
                raise SchemaError("state ownership drifted", path=f"{path}.state_bindings[{index}]")
        if len({item.id for item in self.state_bindings}) != len(self.state_bindings):
            raise SchemaError("duplicate state bindings", path=f"{path}.state_bindings")
        state_ids = {item.id for item in self.state_bindings}
        state_by_id = {item.id: item for item in self.state_bindings}
        flow_actions: dict[str, list[MoeRectAction]] = {}
        for action in self.actions:
            if any(ref not in state_ids for ref in action.state_refs):
                raise SchemaError("action references unknown state", path=f"{path}.actions")
            if action.flow_ref is not None:
                if action.flow_ref not in flow_index:
                    raise SchemaError("action references unknown flow", path=f"{path}.actions")
                flow_actions.setdefault(action.flow_ref, []).append(action)
        transport_kinds = (
            MoeRectActionKind.SEND,
            MoeRectActionKind.RECV,
            MoeRectActionKind.WAIT,
        )
        for flow_ref in flow_index:
            if tuple(item.kind for item in flow_actions.get(flow_ref, ())) != transport_kinds:
                raise SchemaError("flow must lower to exact SEND/RECV/WAIT actions", path=f"{path}.actions")
        incoming_flows: dict[tuple[MoeRectFlowStage, int], list[MoeRectFlow]] = {}
        for flow in self.flows:
            incoming_flows.setdefault(
                (flow.stage, flow.destination_rank), []
            ).append(flow)
        for key, members in incoming_flows.items():
            if len(members) <= 3:
                continue
            expected = tuple(
                item.id for item in sorted(
                    members,
                    key=lambda item: (
                        item.wave_index, item.source_rank, item.id,
                    ),
                )
            )
            actual = tuple(
                action.flow_ref for action in self.actions
                if action.kind is MoeRectActionKind.RECV
                and action.flow_ref is not None
                and (
                    flow_index[action.flow_ref].stage,
                    flow_index[action.flow_ref].destination_rank,
                ) == key
            )
            if actual != expected:
                raise SchemaError(
                    "fan-in RECV actions must follow cyclic-wave order",
                    path=f"{path}.actions",
                )
            for index, flow_ref in enumerate(expected):
                send = flow_actions[flow_ref][0]
                wait_dependencies = tuple(
                    ref for ref in send.deps
                    if action_index[ref].kind is MoeRectActionKind.WAIT
                )
                expected_wait_dependencies = (
                    () if index == 0
                    else (flow_actions[expected[index - 1]][2].id,)
                )
                if wait_dependencies != expected_wait_dependencies:
                    raise SchemaError(
                        "segmented fan-in SEND requires the previous wave WAIT",
                        path=f"{path}.actions",
                    )
        outgoing_flows: dict[tuple[MoeRectFlowStage, int], list[MoeRectFlow]] = {}
        for flow in self.flows:
            outgoing_flows.setdefault(
                (flow.stage, flow.source_rank), []
            ).append(flow)
        for key, members in outgoing_flows.items():
            if len(members) <= 3:
                continue
            expected = tuple(
                item.id for item in sorted(
                    members,
                    key=lambda item: (
                        item.wave_index, item.destination_rank, item.id,
                    ),
                )
            )
            actual = tuple(
                action.flow_ref for action in self.actions
                if action.kind is MoeRectActionKind.SEND
                and action.flow_ref is not None
                and (
                    flow_index[action.flow_ref].stage,
                    flow_index[action.flow_ref].source_rank,
                ) == key
            )
            if actual != expected:
                raise SchemaError(
                    "fan-out SEND actions must follow cyclic-wave order",
                    path=f"{path}.actions",
                )
            for index, flow_ref in enumerate(expected):
                send = flow_actions[flow_ref][0]
                wait_dependencies = tuple(
                    ref for ref in send.deps
                    if action_index[ref].kind is MoeRectActionKind.WAIT
                )
                expected_wait_dependencies = (
                    () if index == 0
                    else (flow_actions[expected[index - 1]][2].id,)
                )
                if wait_dependencies != expected_wait_dependencies:
                    raise SchemaError(
                        "segmented fan-out SEND requires the previous wave WAIT",
                        path=f"{path}.actions",
                    )
        for action in self.actions:
            dep_kinds = {action_index[ref].kind for ref in action.deps}
            if action.kind is MoeRectActionKind.SEND and action.flow_ref is not None:
                flow = flow_index[action.flow_ref]
                if flow.stage is MoeRectFlowStage.GATE_ALL_REDUCE:
                    expected_dep = (
                        MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE
                        if flow.source_rank > flow.destination_rank
                        else MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE
                    )
                    valid = dep_kinds == {expected_dep} and len(action.deps) == 1
                    if not valid:
                        raise SchemaError(
                            "gate tree SEND predecessor drifted",
                            path=f"{path}.actions",
                        )
            if action.kind is MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE:
                expected_child_pairs = {
                    (child, action.rank)
                    for child in (2 * action.rank + 1, 2 * action.rank + 2)
                    if child < spec.mesh.rank_count
                }
                actual_child_pairs = {
                    (
                        flow_index[action_index[ref].flow_ref].source_rank,
                        flow_index[action_index[ref].flow_ref].destination_rank,
                    )
                    for ref in action.deps
                    if action_index[ref].kind is MoeRectActionKind.WAIT
                    and action_index[ref].flow_ref is not None
                }
                expected_kinds = {MoeRectActionKind.GATE_WGRAD}
                if expected_child_pairs:
                    expected_kinds.add(MoeRectActionKind.WAIT)
                if (
                    dep_kinds != expected_kinds
                    or len(action.deps) != 1 + len(expected_child_pairs)
                    or actual_child_pairs != expected_child_pairs
                ):
                    raise SchemaError(
                        "gate local reduce must wait for exact child tree edges",
                        path=f"{path}.actions",
                    )
            if action.kind is MoeRectActionKind.EXPERT_SGD and dep_kinds != {MoeRectActionKind.EXPERT_WGRAD}:
                raise SchemaError("expert SGD must wait for exact expert WGRAD", path=f"{path}.actions")
            if action.kind is MoeRectActionKind.GATE_SGD and dep_kinds != {MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE}:
                raise SchemaError("gate SGD must wait for gate AllReduce", path=f"{path}.actions")
            if action.kind is MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE:
                expected = {MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE}
                if action.rank > 0:
                    expected.add(MoeRectActionKind.WAIT)
                wait_flows = {
                    (
                        flow_index[action_index[ref].flow_ref].source_rank,
                        flow_index[action_index[ref].flow_ref].destination_rank,
                    )
                    for ref in action.deps
                    if action_index[ref].kind is MoeRectActionKind.WAIT
                    and action_index[ref].flow_ref is not None
                }
                expected_wait_flows = (
                    set() if action.rank == 0
                    else {((action.rank - 1) // 2, action.rank)}
                )
                expected_dep_count = (
                    1 if action.rank == 0 else 2
                )
                if (
                    dep_kinds != expected
                    or len(action.deps) != expected_dep_count
                    or wait_flows != expected_wait_flows
                ):
                    raise SchemaError(
                        "gate tree sync must wait for exact local WGRAD and tree phase",
                        path=f"{path}.actions",
                    )
            if action.kind is MoeRectActionKind.STATE_STORE:
                if len(action.state_refs) != 1:
                    raise SchemaError("state store must carry one parameter", path=f"{path}.actions")
                role = state_by_id[action.state_refs[0]].role
                if spec.mode is FlexibleMoeMode.INFERENCE:
                    if (
                        role is not MoeRectStateRole.EXPERT_PARAMETER
                        or dep_kinds != {MoeRectActionKind.WEIGHTED_COMBINE}
                    ):
                        raise SchemaError(
                            "inference retention store must wait for combine",
                            path=f"{path}.actions",
                        )
                    continue
                expected_dep = {
                    MoeRectStateRole.EXPERT_PARAMETER: MoeRectActionKind.EXPERT_SGD,
                    MoeRectStateRole.GATE_PARAMETER: MoeRectActionKind.GATE_SGD,
                }.get(role)
                if expected_dep is None or dep_kinds != {expected_dep}:
                    raise SchemaError("state store must wait for its exact optimizer", path=f"{path}.actions")
        if spec.mode is FlexibleMoeMode.TRAIN:
            if self.gate_all_reduce is None:
                raise SchemaError("training requires gate-gradient AllReduce", path=f"{path}.gate_all_reduce")
            self.gate_all_reduce.validate(f"{path}.gate_all_reduce")
        elif self.gate_all_reduce is not None:
            raise SchemaError("inference cannot carry gradient AllReduce", path=f"{path}.gate_all_reduce")
        if not self.terminal_action_refs or any(ref not in action_index for ref in self.terminal_action_refs):
            raise SchemaError("terminal action closure is invalid", path=f"{path}.terminal_action_refs")
        if len(self.actions) > spec.limits.max_actions or len(self.flows) > spec.limits.max_flows or len(self.state_bindings) > spec.limits.max_state_bindings:
            raise SchemaError("typed carrier capacity exceeded", path=path)
        if self.symbolic_record_count > spec.limits.max_records or self.symbolic_file_bytes > spec.limits.max_artifact_file_bytes:
            raise SchemaError("symbolic artifact capacity exceeded", path=path)
        if self.timing_execution is not True or self.functional_execution is not False:
            raise SchemaError("v2 is timing-only", path=path)
        expected = stable_artifact_id(
            "flexible_moe_plan", self._semantic(),
            schema_version=FLEXIBLE_MOE_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")
        if signed_source is not None:
            routes = tuple(
                f"assignment.{assignment.token_index}"
                for assignment in spec.trace.assignments
                if assignment.source_rank == 0
            )
            imported = tuple(action for action in self.actions if action.kind is MoeRectActionKind.SHARED_DCOMBINED_IMPORT)
            earlier = tuple(action for action in self.actions if action.kind is MoeRectActionKind.SCORE_WEIGHT_BACKWARD_PRE_DISPATCH)
            if len(imported) != 1 or len(earlier) != 1:
                raise SchemaError("signed router needs one pre-dispatch gradient and one shared import", path=f"{path}.actions")
            shared, score = imported[0], earlier[0]
            if (shared.rank != 0 or score.rank != 0
                    or shared.assignment_refs != routes or score.assignment_refs != routes
                    or shared.flow_ref is not None or score.flow_ref is not None
                    or shared.state_refs or score.state_refs
                    or shared.logical_bytes != 2 * len(routes) * spec.hidden_size
                    or shared.flops != 0
                    or score.logical_bytes != 2 * len(routes) * (spec.hidden_size + spec.expert_count)
                    or score.flops != 3 * len(routes) * spec.hidden_size
                    or len(shared.deps) != 1
                    or action_index[shared.deps[0]].kind is not MoeRectActionKind.WEIGHTED_COMBINE
                    or action_index[shared.deps[0]].rank != 0
                    or set(score.deps) != {shared.id, shared.deps[0]}
                    or len(score.deps) != 2):
                raise SchemaError("signed router import/score work and forward dependency drifted", path=f"{path}.actions")
            for action in self.actions:
                if (action.rank == 0 and action.kind is MoeRectActionKind.SEND
                        and action.flow_ref is not None
                        and flow_index[action.flow_ref].stage is MoeRectFlowStage.BACKWARD_GRADIENT
                        and (score.id not in action.deps or shared.deps[0] in action.deps)):
                    raise SchemaError("remote dExpert SEND must consume early scored producer", path=f"{path}.actions")
                if (action.rank == 0 and action.assignment_refs
                        and action.kind in (MoeRectActionKind.EXPERT_DGRAD, MoeRectActionKind.EXPERT_WGRAD)
                        and score.id not in action.deps):
                    raise SchemaError("local expert reverse must consume early scored producer", path=f"{path}.actions")

    def validate_against_signed(
        self, spec: FlexibleMoeSpec, signed_source: MoeRectSignedTop1TrainSource,
        path: str = "flexible_moe_signed_plan",
    ) -> None:
        self.validate_against(spec, path, signed_source=signed_source)
        from ..passes.moe_signed_router_train_source_plan import compile_moe_signed_top1_train_source_plan
        canonical = compile_moe_signed_top1_train_source_plan(
            spec, signed_source=signed_source, _validate_plan=False,
        )
        if self != canonical:
            raise SchemaError("signed MoE action/flow/state DAG differs from canonical original P2 source", path=path)


__all__ = [
    "FLEXIBLE_MOE_PLAN_SCHEMA_VERSION",
    "FLEXIBLE_MOE_SPEC_SCHEMA_VERSION",
    "FLEXIBLE_MOE_TRACE_SCHEMA_VERSION",
    "FlexibleMoeExecutablePlan",
    "FlexibleMoeLimits",
    "FlexibleMoeMode",
    "FlexibleMoeSpec",
    "MoeRectAction",
    "MoeRectActionKind",
    "MoeRectSignedTop1TrainSource",
    "MoeRectFlow",
    "MoeRectFlowStage",
    "MoeRectGateAllReduce",
    "MoeRectStateBinding",
    "MoeRectStateRole",
    "MoeRectStaticTrace",
    "MoeRectTraceAssignment",
]
