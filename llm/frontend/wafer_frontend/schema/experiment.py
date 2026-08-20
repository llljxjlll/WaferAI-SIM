"""Strict experiment input for the Naive Dense static-forward MVP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError, UnsupportedFeatureError
from .common import (
    DType,
    ProfileKey,
    RoundingMode,
    ValidationMode,
    validate_nonempty,
    validate_uint64,
)
from .ir0 import PipelineSchedule, RecomputeMode, TrainStructure


EXPERIMENT_SCHEMA_VERSION = "wafer_frontend.experiment/v1alpha5"


class ModelSource(str, Enum):
    ANALYTIC = "analytic"
    HLO = "hlo"


class ModelArch(str, Enum):
    LLAMA = "llama"


class WorkloadMode(str, Enum):
    INFER = "infer"
    TRAIN = "train"


class InferSource(str, Enum):
    STATIC_PROFILE = "static_profile"
    SHAPE_DIST = "shape_dist"
    PD_STATIC = "pd_static"


class InferOutput(str, Enum):
    LOGITS = "logits"
    GREEDY_SAMPLE = "greedy_sample"


class TrainOptimizer(str, Enum):
    NONE = "none"
    SGD = "sgd"
    ADAMW = "adamw"


class InstanceRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"
    TRAIN = "train"


class PlacementStrategy(str, Enum):
    COMPACT = "compact"
    EXPLICIT = "explicit"


class PartitionPolicy(str, Enum):
    GEMM_COLL = "gemm_coll"
    AUTO = "auto"


class InterDiePolicyName(str, Enum):
    NAIVE = "naive"
    SWIZZLE_TOPO = "swizzle_topo"


class IntraDiePolicyName(str, Enum):
    NAIVE = "naive"
    OPTIMIZED = "optimized"


class ExecutionBackend(str, Enum):
    UNIFIED_STREAM = "unified_stream"


class OrdinaryLoweringName(str, Enum):
    JSON_COARSE = "json_coarse"


class FusedLoweringName(str, Enum):
    ISA_REGION = "isa_region"


class StandaloneLoweringName(str, Enum):
    STRICT_ACTIONS = "strict_actions"


class TransportMode(str, Enum):
    STRICT = "strict"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


def _unsupported(path: str, value: object, supported: object) -> None:
    raise UnsupportedFeatureError(
        f"{value!r} is outside the Naive Dense static-forward MVP; supported: {supported!r}",
        path=path,
    )


def _derived_uint64(value: int, path: str) -> int:
    validate_uint64(value, path)
    return value


def _positive_finite(value: float, path: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise SchemaError("must be a finite positive float", path=path)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    source: ModelSource
    arch: ModelArch
    V: int
    H: int
    I: int
    NH: int
    KVH: int
    DH: int
    rotary_dim: int
    L: int
    dtype: DType
    tie_word_embeddings: bool
    rms_norm_epsilon: float
    rope_theta: float
    max_position_embeddings: int
    moe: object | None = None

    def validate(self, path: str) -> None:
        for name in (
            "V",
            "H",
            "I",
            "NH",
            "KVH",
            "DH",
            "rotary_dim",
            "L",
            "max_position_embeddings",
        ):
            _positive(getattr(self, name), f"{path}.{name}")
        if type(self.tie_word_embeddings) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.tie_word_embeddings")
        if self.tie_word_embeddings:
            _unsupported(f"{path}.tie_word_embeddings", True, False)
        _positive_finite(self.rms_norm_epsilon, f"{path}.rms_norm_epsilon")
        _positive_finite(self.rope_theta, f"{path}.rope_theta")
        if self.source is not ModelSource.ANALYTIC:
            _unsupported(f"{path}.source", self.source.value, ModelSource.ANALYTIC.value)
        if self.dtype is not DType.FP16:
            _unsupported(f"{path}.dtype", self.dtype.value, DType.FP16.value)
        if self.moe is not None:
            _unsupported(f"{path}.moe", self.moe, None)
        hidden_from_heads = _derived_uint64(self.NH * self.DH, f"{path}.derived.H")
        if self.H != hidden_from_heads:
            raise SchemaError("must equal NH * DH", path=f"{path}.H")
        if self.KVH > self.NH:
            raise SchemaError("must not exceed NH", path=f"{path}.KVH")
        if self.NH % self.KVH != 0:
            raise SchemaError("NH must be divisible by KVH", path=f"{path}.KVH")
        if self.rotary_dim != self.DH:
            _unsupported(f"{path}.rotary_dim", self.rotary_dim, self.DH)
        q_width = _derived_uint64(
            (self.NH + 2 * self.KVH) * self.DH, f"{path}.derived.Q"
        )
        qkv_params = _derived_uint64(self.H * q_width, f"{path}.derived.qkv_params")
        output_params = _derived_uint64(
            self.NH * self.DH * self.H, f"{path}.derived.output_params"
        )
        gate_up_params = _derived_uint64(
            2 * self.H * self.I, f"{path}.derived.gate_up_params"
        )
        down_params = _derived_uint64(
            self.I * self.H, f"{path}.derived.down_params"
        )
        per_layer_params = _derived_uint64(
            qkv_params + output_params + gate_up_params + down_params,
            f"{path}.derived.per_layer_params",
        )
        per_layer_with_norm = _derived_uint64(
            per_layer_params + 2 * self.H,
            f"{path}.derived.per_layer_params_with_norm",
        )
        layers = _derived_uint64(
            per_layer_with_norm * self.L,
            f"{path}.derived.total_layer_params",
        )
        embedding = _derived_uint64(
            self.V * self.H, f"{path}.derived.embedding_params"
        )
        lm_head = _derived_uint64(
            self.H * self.V, f"{path}.derived.lm_head_params"
        )
        _derived_uint64(
            embedding + layers + self.H + lm_head,
            f"{path}.derived.total_params",
        )

    def parameter_elements(self) -> int:
        self.validate("model")
        q_width = (self.NH + 2 * self.KVH) * self.DH
        per_layer_linear = self.H * q_width + self.H * self.H + 3 * self.H * self.I
        return (
            self.V * self.H
            + self.L * (per_layer_linear + 2 * self.H)
            + self.H
            + self.H * self.V
        )

    def parameter_bytes(self) -> int:
        return self.parameter_elements() * 2


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    ref: str

    def validate(self, path: str) -> None:
        if not self.ref:
            raise SchemaError("must be a non-empty path", path=f"{path}.ref")


@dataclass(frozen=True, slots=True)
class WeightedProfile:
    key: ProfileKey
    weight: float

    def validate(self, path: str) -> None:
        if type(self.weight) is not float or not math.isfinite(self.weight) or self.weight <= 0.0:
            raise SchemaError("must be a finite positive float", path=f"{path}.weight")


@dataclass(frozen=True, slots=True)
class ShapeDistribution:
    profiles: tuple[WeightedProfile, ...]

    def validate(self, path: str) -> None:
        if not self.profiles:
            raise SchemaError("must contain at least one profile", path=f"{path}.profiles")
        seen: set[str] = set()
        for index, entry in enumerate(self.profiles):
            entry.validate(f"{path}.profiles[{index}]")
            entry.key.validate(f"{path}.profiles[{index}].key")
            profile_id = entry.key.stable_id()
            if profile_id in seen:
                raise SchemaError("contains a duplicate ProfileKey", path=f"{path}.profiles[{index}].key")
            seen.add(profile_id)
        if abs(math.fsum(entry.weight for entry in self.profiles) - 1.0) > 1e-12:
            raise SchemaError(
                "weights must sum to 1 within 1e-12", path=f"{path}.profiles"
            )


@dataclass(frozen=True, slots=True)
class PdStaticWorkloadSpec:
    """One static prefill/decode pair and the selected decode replica."""

    prefill_profile: ProfileKey
    decode_profile: ProfileKey
    prefill_instance_ref: str
    decode_instance_ref: str

    def validate(self, path: str) -> None:
        self.prefill_profile.validate(f"{path}.prefill_profile")
        self.decode_profile.validate(f"{path}.decode_profile")
        validate_nonempty(
            self.prefill_instance_ref, f"{path}.prefill_instance_ref"
        )
        validate_nonempty(
            self.decode_instance_ref, f"{path}.decode_instance_ref"
        )
        if (
            self.prefill_profile.prefill_tokens == 0
            or self.prefill_profile.decode_tokens != 0
        ):
            raise SchemaError(
                "must be a pure prefill profile",
                path=f"{path}.prefill_profile",
            )
        if (
            self.decode_profile.prefill_tokens != 0
            or self.decode_profile.decode_tokens == 0
        ):
            raise SchemaError(
                "must be a pure decode profile",
                path=f"{path}.decode_profile",
            )
        if self.prefill_profile.num_seqs != self.decode_profile.num_seqs:
            raise SchemaError(
                "prefill/decode profiles must contain the same request count",
                path=path,
            )
        if (
            self.decode_profile.context_sum
            != self.prefill_profile.context_sum
            + self.decode_profile.decode_tokens
        ):
            raise SchemaError(
                "decode context_sum must equal prefill context_sum + decode_tokens",
                path=f"{path}.decode_profile.context_sum",
            )
        if self.decode_profile.context_max < self.prefill_profile.context_max:
            raise SchemaError(
                "must not be smaller than prefill context_max",
                path=f"{path}.decode_profile.context_max",
            )


@dataclass(frozen=True, slots=True)
class InferWorkloadSpec:
    source: InferSource
    output: InferOutput
    profile: ProfileKey | None = None
    shape_dist: ShapeDistribution | None = None
    pd_static: PdStaticWorkloadSpec | None = None

    def validate(self, path: str) -> None:
        if type(self.output) is not InferOutput:
            raise SchemaError("must be an InferOutput", path=f"{path}.output")
        if self.source is InferSource.STATIC_PROFILE:
            if self.profile is None:
                raise SchemaError("is required for static_profile", path=f"{path}.profile")
            if self.shape_dist is not None:
                raise SchemaError("must be null for static_profile", path=f"{path}.shape_dist")
            if self.pd_static is not None:
                raise SchemaError("must be null for static_profile", path=f"{path}.pd_static")
            self.profile.validate(f"{path}.profile")
            return
        if self.source is InferSource.SHAPE_DIST:
            if self.profile is not None:
                raise SchemaError("must be null for shape_dist", path=f"{path}.profile")
            if self.shape_dist is None:
                raise SchemaError("is required for shape_dist", path=f"{path}.shape_dist")
            if self.pd_static is not None:
                raise SchemaError("must be null for shape_dist", path=f"{path}.pd_static")
            self.shape_dist.validate(f"{path}.shape_dist")
            return
        if self.source is not InferSource.PD_STATIC:
            raise SchemaError("unsupported infer source", path=f"{path}.source")
        if self.profile is not None or self.shape_dist is not None:
            raise SchemaError(
                "profile and shape_dist must be null for pd_static", path=path
            )
        if self.pd_static is None:
            raise SchemaError("is required for pd_static", path=f"{path}.pd_static")
        self.pd_static.validate(f"{path}.pd_static")

    def profiles(self) -> tuple[ProfileKey, ...]:
        if self.pd_static is not None:
            return (
                self.pd_static.prefill_profile,
                self.pd_static.decode_profile,
            )
        if self.profile is not None:
            return (self.profile,)
        assert self.shape_dist is not None
        return tuple(entry.key for entry in self.shape_dist.profiles)


@dataclass(frozen=True, slots=True)
class TrainWorkloadSpec:
    """Forward-only train step contract; DAG expansion starts in N6.1."""

    global_batch: int
    micro_batch: int
    seq_len: int
    backward: bool
    optimizer: TrainOptimizer
    structure: TrainStructure

    def validate(self, path: str) -> None:
        for name in ("global_batch", "micro_batch", "seq_len"):
            _positive(getattr(self, name), f"{path}.{name}")
        if type(self.backward) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.backward")
        if self.backward:
            _unsupported(f"{path}.backward", True, False)
        if type(self.optimizer) is not TrainOptimizer:
            raise SchemaError(
                "must be a TrainOptimizer", path=f"{path}.optimizer"
            )
        if self.optimizer is not TrainOptimizer.NONE:
            _unsupported(
                f"{path}.optimizer",
                self.optimizer.value,
                TrainOptimizer.NONE.value,
            )
        if type(self.structure) is not TrainStructure:
            raise SchemaError(
                "must be a TrainStructure", path=f"{path}.structure"
            )
        self.structure.validate(f"{path}.structure")
        if type(self.structure.recompute) is not RecomputeMode:
            raise SchemaError(
                "must be a RecomputeMode",
                path=f"{path}.structure.recompute",
            )
        if type(self.structure.pp_schedule) is not PipelineSchedule:
            raise SchemaError(
                "must be a PipelineSchedule",
                path=f"{path}.structure.pp_schedule",
            )
        if self.structure.recompute is not RecomputeMode.NONE:
            _unsupported(
                f"{path}.structure.recompute",
                self.structure.recompute.value,
                RecomputeMode.NONE.value,
            )
        if self.structure.pp_schedule is not PipelineSchedule.GPIPE:
            _unsupported(
                f"{path}.structure.pp_schedule",
                self.structure.pp_schedule.value,
                PipelineSchedule.GPIPE.value,
            )
        if self.structure.interleave_chunks != 1:
            _unsupported(
                f"{path}.structure.interleave_chunks",
                self.structure.interleave_chunks,
                1,
            )


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    mode: WorkloadMode
    infer: InferWorkloadSpec | None = None
    train: TrainWorkloadSpec | None = None

    def validate(self, path: str) -> None:
        if self.mode is WorkloadMode.INFER:
            if self.infer is None:
                raise SchemaError("is required for infer", path=f"{path}.infer")
            if self.train is not None:
                raise SchemaError("must be null for infer", path=f"{path}.train")
            if type(self.infer) is not InferWorkloadSpec:
                raise SchemaError(
                    "must be an InferWorkloadSpec", path=f"{path}.infer"
                )
            self.infer.validate(f"{path}.infer")
            return
        if self.mode is WorkloadMode.TRAIN:
            if self.train is None:
                raise SchemaError("is required for train", path=f"{path}.train")
            if self.infer is not None:
                raise SchemaError("must be null for train", path=f"{path}.infer")
            if type(self.train) is not TrainWorkloadSpec:
                raise SchemaError(
                    "must be a TrainWorkloadSpec", path=f"{path}.train"
                )
            self.train.validate(f"{path}.train")
            return
        raise SchemaError("unsupported workload mode", path=f"{path}.mode")


@dataclass(frozen=True, slots=True)
class ParallelInstanceSpec:
    id: str
    role: InstanceRole
    tp: int
    sp: bool
    replicas: int = 1
    dp: int = 1
    pp: int = 1
    ep: int = 1

    def validate(self, path: str) -> None:
        if not self.id:
            raise SchemaError("must be non-empty", path=f"{path}.id")
        if type(self.role) is not InstanceRole:
            raise SchemaError("must be an InstanceRole", path=f"{path}.role")
        if type(self.sp) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.sp")
        for name in ("tp", "replicas", "dp", "pp", "ep"):
            _positive(getattr(self, name), f"{path}.{name}")
        if self.role is InstanceRole.TRAIN:
            for name in ("replicas", "pp", "ep"):
                if getattr(self, name) != 1:
                    _unsupported(f"{path}.{name}", getattr(self, name), 1)
            return
        for name in ("replicas", "dp", "pp", "ep"):
            if getattr(self, name) != 1:
                _unsupported(f"{path}.{name}", getattr(self, name), 1)


@dataclass(frozen=True, slots=True)
class ParallelSpec:
    instances: tuple[ParallelInstanceSpec, ...]

    def validate(self, path: str) -> None:
        if not self.instances:
            raise SchemaError(
                "must contain at least one instance", path=f"{path}.instances"
            )
        role_order = {
            InstanceRole.PREFILL: 0,
            InstanceRole.BOTH: 1,
            InstanceRole.DECODE: 2,
            InstanceRole.TRAIN: 3,
        }
        expected = tuple(
            sorted(self.instances, key=lambda item: (role_order[item.role], item.id))
        )
        if self.instances != expected:
            raise SchemaError(
                "must use canonical role/id order", path=f"{path}.instances"
            )
        ids = [instance.id for instance in self.instances]
        if len(set(ids)) != len(ids):
            raise SchemaError(
                "contains a duplicate instance id", path=f"{path}.instances"
            )
        for index, instance in enumerate(self.instances):
            instance.validate(f"{path}.instances[{index}]")


@dataclass(frozen=True, slots=True)
class ExplicitGroupPlacement:
    instance_id: str
    mesh_ref: str
    die_ids: tuple[int, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.instance_id, f"{path}.instance_id")
        validate_nonempty(self.mesh_ref, f"{path}.mesh_ref")
        if not self.die_ids:
            raise SchemaError("must contain at least one die id", path=f"{path}.die_ids")
        if len(set(self.die_ids)) != len(self.die_ids):
            raise SchemaError("contains a duplicate die id", path=f"{path}.die_ids")
        for index, die_id in enumerate(self.die_ids):
            validate_uint64(die_id, f"{path}.die_ids[{index}]")


@dataclass(frozen=True, slots=True)
class PlacementSpec:
    strategy: PlacementStrategy
    groups: tuple[ExplicitGroupPlacement, ...]

    def validate(self, path: str) -> None:
        if self.strategy is PlacementStrategy.COMPACT:
            if self.groups:
                raise SchemaError(
                    "must be empty for compact placement", path=f"{path}.groups"
                )
            return
        if self.strategy is not PlacementStrategy.EXPLICIT:
            raise SchemaError("unsupported placement strategy", path=f"{path}.strategy")
        if not self.groups:
            raise SchemaError(
                "must contain every instance/mesh placement for explicit strategy",
                path=f"{path}.groups",
            )
        keys: list[tuple[str, str]] = []
        for index, group in enumerate(self.groups):
            group.validate(f"{path}.groups[{index}]")
            keys.append((group.instance_id, group.mesh_ref))
        if len(set(keys)) != len(keys):
            raise SchemaError(
                "contains a duplicate instance/mesh placement", path=f"{path}.groups"
            )
        if keys != sorted(keys):
            raise SchemaError(
                "must be in canonical instance_id/mesh_ref order",
                path=f"{path}.groups",
            )


@dataclass(frozen=True, slots=True)
class PolicySpec:
    partition: PartitionPolicy
    inter_die: InterDiePolicyName
    intra_die: IntraDiePolicyName

    def validate(self, path: str) -> None:
        if type(self.partition) is not PartitionPolicy:
            raise SchemaError(
                "must be a PartitionPolicy", path=f"{path}.partition"
            )
        if type(self.inter_die) is not InterDiePolicyName:
            raise SchemaError(
                "must be an InterDiePolicyName", path=f"{path}.inter_die"
            )
        if type(self.intra_die) is not IntraDiePolicyName:
            raise SchemaError(
                "must be an IntraDiePolicyName", path=f"{path}.intra_die"
            )
        if self.partition is not PartitionPolicy.GEMM_COLL:
            _unsupported(
                f"{path}.partition", self.partition.value, PartitionPolicy.GEMM_COLL.value
            )


@dataclass(frozen=True, slots=True)
class ReductionContract:
    accumulate: DType
    rounding: RoundingMode
    validation: ValidationMode

    def validate(self, path: str) -> None:
        if self.accumulate is not DType.FP32:
            _unsupported(f"{path}.accumulate", self.accumulate.value, DType.FP32.value)


@dataclass(frozen=True, slots=True)
class BackendSpec:
    execution: ExecutionBackend
    ordinary_lowering: OrdinaryLoweringName
    fused_lowering: FusedLoweringName
    standalone_collective_lowering: StandaloneLoweringName
    reduction_contract: ReductionContract
    transport: TransportMode
    static_link: bool
    dynamic_region_dispatch: bool

    def validate(self, path: str) -> None:
        if not self.static_link:
            _unsupported(f"{path}.static_link", self.static_link, True)
        if self.dynamic_region_dispatch:
            _unsupported(f"{path}.dynamic_region_dispatch", True, False)
        self.reduction_contract.validate(f"{path}.reduction_contract")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    schema_version: str
    model: ModelSpec
    hardware: HardwareSpec
    workload: WorkloadSpec
    parallel: ParallelSpec
    placement: PlacementSpec
    policy: PolicySpec
    backend: BackendSpec

    def validate(self, path: str = "spec") -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}; expected {EXPERIMENT_SCHEMA_VERSION!r}",
                path=f"{path}.schema_version",
            )
        self.model.validate(f"{path}.model")
        self.hardware.validate(f"{path}.hardware")
        self.workload.validate(f"{path}.workload")
        self.parallel.validate(f"{path}.parallel")
        self.placement.validate(f"{path}.placement")
        self.policy.validate(f"{path}.policy")
        self.backend.validate(f"{path}.backend")

        instances = self.parallel.instances
        if self.workload.mode is WorkloadMode.TRAIN:
            train = self.workload.train
            assert train is not None
            if len(instances) != 1:
                _unsupported(
                    f"{path}.parallel.instances",
                    len(instances),
                    "exactly one TRAIN instance",
                )
            instance = instances[0]
            if instance.role is not InstanceRole.TRAIN:
                raise SchemaError(
                    "train workload requires role=TRAIN",
                    path=f"{path}.parallel.instances[0].role",
                )
            q_width = (self.model.NH + 2 * self.model.KVH) * self.model.DH
            for dimension_name, dimension in (
                ("H", self.model.H),
                ("NH", self.model.NH),
                ("KVH", self.model.KVH),
                ("I", self.model.I),
                ("Q", q_width),
            ):
                if dimension % instance.tp != 0:
                    raise SchemaError(
                        f"model dimension {dimension_name} must be divisible by tp={instance.tp}",
                        path=f"{path}.parallel.instances[0].tp",
                    )
            expected_global_batch = _derived_uint64(
                train.micro_batch
                * instance.dp
                * train.structure.micro_batch_count,
                f"{path}.workload.train.derived.global_batch",
            )
            if train.global_batch != expected_global_batch:
                raise SchemaError(
                    "must equal micro_batch * dp * micro_batch_count",
                    path=f"{path}.workload.train.global_batch",
                )
            if train.seq_len > self.model.max_position_embeddings:
                raise SchemaError(
                    "must not exceed model.max_position_embeddings",
                    path=f"{path}.workload.train.seq_len",
                )
            local_tokens = _derived_uint64(
                train.micro_batch * train.seq_len,
                f"{path}.workload.train.derived.local_tokens",
            )
            if instance.sp and local_tokens % instance.tp != 0:
                raise SchemaError(
                    f"sequence-parallel local token count must be divisible by tp={instance.tp}",
                    path=f"{path}.workload.train.seq_len",
                )
            return

        infer = self.workload.infer
        assert infer is not None
        if any(instance.role is InstanceRole.TRAIN for instance in instances):
            raise SchemaError(
                "infer workload cannot use role=TRAIN",
                path=f"{path}.parallel.instances",
            )
        instance_by_id = {instance.id: instance for instance in instances}
        pd_static = infer.pd_static
        if infer.source is not InferSource.PD_STATIC:
            if len(instances) != 1:
                _unsupported(
                    f"{path}.parallel.instances",
                    len(instances),
                    "exactly one instance outside pd_static",
                )
        else:
            assert pd_static is not None
            prefill = instance_by_id.get(pd_static.prefill_instance_ref)
            decode = instance_by_id.get(pd_static.decode_instance_ref)
            if prefill is None:
                raise SchemaError(
                    "references an unknown instance",
                    path=(
                        f"{path}.workload.infer.pd_static"
                        ".prefill_instance_ref"
                    ),
                )
            if decode is None:
                raise SchemaError(
                    "references an unknown instance",
                    path=(
                        f"{path}.workload.infer.pd_static"
                        ".decode_instance_ref"
                    ),
                )
            if prefill.id == decode.id:
                if len(instances) != 1 or prefill.role is not InstanceRole.BOTH:
                    raise SchemaError(
                        "fused PD requires exactly one BOTH instance",
                        path=f"{path}.parallel.instances",
                    )
            else:
                prefill_instances = tuple(
                    item
                    for item in instances
                    if item.role is InstanceRole.PREFILL
                )
                decode_instances = tuple(
                    item
                    for item in instances
                    if item.role is InstanceRole.DECODE
                )
                if (
                    len(prefill_instances) != 1
                    or prefill != prefill_instances[0]
                    or decode not in decode_instances
                    or not decode_instances
                    or len(prefill_instances) + len(decode_instances)
                    != len(instances)
                ):
                    raise SchemaError(
                        "separated PD requires one PREFILL and one or more DECODE instances",
                        path=f"{path}.parallel.instances",
                    )
                if self.placement.strategy is not PlacementStrategy.EXPLICIT:
                    raise SchemaError(
                        "separated PD requires explicit instance placement",
                        path=f"{path}.placement.strategy",
                    )
                expected_placements = {
                    (item.id, f"{item.id}.mesh.tp"): item.tp
                    for item in instances
                }
                actual_placements = {
                    (group.instance_id, group.mesh_ref): len(group.die_ids)
                    for group in self.placement.groups
                }
                if actual_placements != expected_placements:
                    raise SchemaError(
                        "must exactly place every PD instance TP mesh",
                        path=f"{path}.placement.groups",
                    )
                all_dies = tuple(
                    die_id
                    for group in self.placement.groups
                    for die_id in group.die_ids
                )
                if len(set(all_dies)) != len(all_dies):
                    raise SchemaError(
                        "PD instance placements must be disjoint",
                        path=f"{path}.placement.groups",
                    )

        profiles = infer.profiles()
        q_width = (self.model.NH + 2 * self.model.KVH) * self.model.DH
        partitioned_dimensions = (
            ("H", self.model.H),
            ("NH", self.model.NH),
            ("KVH", self.model.KVH),
            ("I", self.model.I),
            ("Q", q_width),
        )
        for instance_index, instance in enumerate(self.parallel.instances):
            instance_path = f"{path}.parallel.instances[{instance_index}]"
            for dimension_name, dimension in partitioned_dimensions:
                if dimension % instance.tp != 0:
                    raise SchemaError(
                        f"model dimension {dimension_name} must be divisible by tp={instance.tp}",
                        path=f"{instance_path}.tp",
                    )
            if pd_static is None:
                instance_profiles = tuple(
                    (
                        profile,
                        (
                            f"{path}.workload.infer.profile"
                            if infer.profile is not None
                            else (
                                f"{path}.workload.infer.shape_dist"
                                f".profiles[{profile_index}].key"
                            )
                        ),
                    )
                    for profile_index, profile in enumerate(profiles)
                )
            elif instance.id == pd_static.prefill_instance_ref:
                instance_profiles = (
                    (
                        pd_static.prefill_profile,
                        f"{path}.workload.infer.pd_static.prefill_profile",
                    ),
                    (
                        pd_static.decode_profile,
                        f"{path}.workload.infer.pd_static.decode_profile",
                    ),
                ) if instance.id == pd_static.decode_instance_ref else (
                    (
                        pd_static.prefill_profile,
                        f"{path}.workload.infer.pd_static.prefill_profile",
                    ),
                )
            else:
                instance_profiles = (
                    (
                        pd_static.decode_profile,
                        f"{path}.workload.infer.pd_static.decode_profile",
                    ),
                )
            for profile, profile_path in instance_profiles:
                if profile.expert_load is not None:
                    raise SchemaError(
                        "must be null for a Dense model",
                        path=f"{profile_path}.expert_load",
                    )
                if profile.context_max > self.model.max_position_embeddings:
                    raise SchemaError(
                        "must not exceed model.max_position_embeddings",
                        path=f"{profile_path}.context_max",
                    )
                if instance.sp:
                    tokens = profile.prefill_tokens + profile.decode_tokens
                    if tokens % instance.tp != 0:
                        raise SchemaError(
                            f"sequence-parallel token count must be divisible by tp={instance.tp}",
                            path=f"{profile_path}.prefill_tokens",
                        )
                if instance.role is InstanceRole.PREFILL and profile.decode_tokens != 0:
                    raise SchemaError(
                        "prefill instance requires decode_tokens == 0",
                        path=f"{profile_path}.decode_tokens",
                    )
                if instance.role is InstanceRole.DECODE and profile.prefill_tokens != 0:
                    raise SchemaError(
                        "decode instance requires prefill_tokens == 0",
                        path=f"{profile_path}.prefill_tokens",
                    )
