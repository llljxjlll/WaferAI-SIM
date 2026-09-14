"""Unified, versioned run request and capability contract for full workloads.

This module describes intent only.  Family dispatch, placement materialization,
memory planning, and runtime evidence are separate production stages.  Keeping
those boundaries explicit prevents a schema-only or motif result from being
reported as full-model runtime support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError, UnsupportedFeatureError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .serde import canonical_digest


WORKLOAD_RUN_REQUEST_SCHEMA_VERSION = "wafer_frontend.workload_run_request/v1alpha1"
WORKLOAD_RUN_CAPABILITY_SCHEMA_VERSION = (
    "wafer_frontend.workload_run_capability/v1alpha1"
)


class WorkloadFamily(str, Enum):
    DENSE_INFERENCE = "dense_inference_e2e"
    DENSE_TRAINING = "dense_training_e2e"
    MOE_INFERENCE = "moe_inference_e2e"
    MOE_TRAINING = "moe_training_e2e"

    @property
    def is_training(self) -> bool:
        return self in (self.DENSE_TRAINING, self.MOE_TRAINING)

    @property
    def is_moe(self) -> bool:
        return self in (self.MOE_INFERENCE, self.MOE_TRAINING)


class WorkloadModelArchitecture(str, Enum):
    LLAMA_DENSE = "llama_dense"
    LLAMA_MOE = "llama_moe"


class WorkloadOptimizerKind(str, Enum):
    SGD = "sgd"
    ADAMW = "adamw"


class WorkloadMemoryMode(str, Enum):
    RESIDENT_HBM = "resident_hbm"
    REMOTE_HBM = "remote_hbm"
    EXTERNAL_OFFLOAD = "external_offload"


class WorkloadRankOrder(str, Enum):
    ROW_MAJOR = "row_major"


class WorkloadRoutePolicy(str, Enum):
    X_FIRST = "x_first"


class WorkloadExecutionStrategy(str, Enum):
    BASELINE = "baseline"
    OPTIMIZED = "optimized"


class WorkloadCapabilityLevel(str, Enum):
    UNSUPPORTED = "unsupported"
    SCHEMA_ONLY = "schema_only"
    NOT_MEASURED = "not_measured"
    SUPPORTED = "supported"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


def _finite(value: float, path: str, *, positive: bool = False) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise SchemaError("must be a finite float", path=path)
    if positive and value <= 0.0:
        raise SchemaError("must be greater than zero", path=path)
    if not positive and value < 0.0:
        raise SchemaError("must be non-negative", path=path)


@dataclass(frozen=True, slots=True)
class WorkloadMeshSpec:
    rows: int
    columns: int
    rank_order: WorkloadRankOrder = WorkloadRankOrder.ROW_MAJOR
    route_policy: WorkloadRoutePolicy = WorkloadRoutePolicy.X_FIRST
    ranks_per_die: int = 1

    def validate(self, path: str = "mesh") -> None:
        _positive(self.rows, f"{path}.rows")
        _positive(self.columns, f"{path}.columns")
        validate_uint64(self.rank_count, f"{path}.rank_count")
        if type(self.rank_order) is not WorkloadRankOrder:
            raise SchemaError("must be a WorkloadRankOrder", path=f"{path}.rank_order")
        if self.rank_order is not WorkloadRankOrder.ROW_MAJOR:
            raise SchemaError("unsupported rank order", path=f"{path}.rank_order")
        if type(self.route_policy) is not WorkloadRoutePolicy:
            raise SchemaError(
                "must be a WorkloadRoutePolicy", path=f"{path}.route_policy"
            )
        if self.route_policy is not WorkloadRoutePolicy.X_FIRST:
            raise SchemaError("unsupported route policy", path=f"{path}.route_policy")
        if type(self.ranks_per_die) is not int or self.ranks_per_die != 1:
            raise SchemaError(
                "v1alpha1 requires one rank per die", path=f"{path}.ranks_per_die"
            )

    @property
    def rank_count(self) -> int:
        return self.rows * self.columns * self.ranks_per_die


@dataclass(frozen=True, slots=True)
class WorkloadModelSpec:
    architecture: WorkloadModelArchitecture
    vocabulary_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    max_sequence_length: int
    dtype: DType = DType.FP16
    num_experts: int = 0
    experts_per_token: int = 0

    def validate(self, path: str = "model") -> None:
        if type(self.architecture) is not WorkloadModelArchitecture:
            raise SchemaError(
                "must be a WorkloadModelArchitecture",
                path=f"{path}.architecture",
            )
        for name in (
            "vocabulary_size",
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_attention_heads",
            "num_kv_heads",
            "head_dim",
            "max_sequence_length",
        ):
            _positive(getattr(self, name), f"{path}.{name}")
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        if self.dtype is not DType.FP16:
            raise UnsupportedFeatureError(
                "v1alpha1 runtime lowering requires fp16 model state",
                path=f"{path}.dtype",
            )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise SchemaError(
                "must equal num_attention_heads * head_dim",
                path=f"{path}.hidden_size",
            )
        if self.num_kv_heads > self.num_attention_heads:
            raise SchemaError(
                "must not exceed num_attention_heads", path=f"{path}.num_kv_heads"
            )
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise SchemaError(
                "num_attention_heads must be divisible by num_kv_heads",
                path=f"{path}.num_kv_heads",
            )
        for name in ("num_experts", "experts_per_token"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.architecture is WorkloadModelArchitecture.LLAMA_DENSE:
            if self.num_experts != 0 or self.experts_per_token != 0:
                raise SchemaError(
                    "Dense models must not declare experts", path=f"{path}.num_experts"
                )
            return
        if self.architecture is not WorkloadModelArchitecture.LLAMA_MOE:
            raise SchemaError("unsupported architecture", path=f"{path}.architecture")
        if self.num_experts == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.num_experts")
        if self.experts_per_token != 1:
            raise UnsupportedFeatureError(
                "v1alpha1 supports static top-1 routing only",
                path=f"{path}.experts_per_token",
            )


@dataclass(frozen=True, slots=True)
class WorkloadInferenceSteps:
    prefill_tokens: int
    decode_steps: int
    request_count: int

    def validate(self, path: str = "steps.inference") -> None:
        validate_uint64(self.prefill_tokens, f"{path}.prefill_tokens")
        validate_uint64(self.decode_steps, f"{path}.decode_steps")
        _positive(self.request_count, f"{path}.request_count")
        if self.prefill_tokens == 0 and self.decode_steps == 0:
            raise SchemaError(
                "prefill_tokens and decode_steps cannot both be zero", path=path
            )


@dataclass(frozen=True, slots=True)
class WorkloadTrainingSteps:
    step_count: int
    global_batch_size: int
    micro_batch_size: int
    micro_batch_count: int
    sequence_length: int

    def validate(self, path: str = "steps.training") -> None:
        for name in (
            "step_count",
            "global_batch_size",
            "micro_batch_size",
            "micro_batch_count",
            "sequence_length",
        ):
            _positive(getattr(self, name), f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class WorkloadStepSpec:
    inference: WorkloadInferenceSteps | None = None
    training: WorkloadTrainingSteps | None = None

    def validate(self, path: str = "steps") -> None:
        if (self.inference is None) == (self.training is None):
            raise SchemaError(
                "exactly one of inference or training is required", path=path
            )
        if self.inference is not None:
            if type(self.inference) is not WorkloadInferenceSteps:
                raise SchemaError(
                    "must be WorkloadInferenceSteps", path=f"{path}.inference"
                )
            self.inference.validate(f"{path}.inference")
        if self.training is not None:
            if type(self.training) is not WorkloadTrainingSteps:
                raise SchemaError(
                    "must be WorkloadTrainingSteps", path=f"{path}.training"
                )
            self.training.validate(f"{path}.training")


@dataclass(frozen=True, slots=True)
class WorkloadParallelSpec:
    tp: int = 1
    dp: int = 1
    ep: int = 1
    pp: int = 1
    active_die_ids: tuple[int, ...] = ()

    def validate(self, path: str = "parallel") -> None:
        for name in ("tp", "dp", "ep", "pp"):
            _positive(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.logical_rank_count, f"{path}.logical_rank_count")

    def validate_against_mesh(
        self, mesh: WorkloadMeshSpec, path: str = "parallel"
    ) -> None:
        self.validate(path)
        if type(mesh) is not WorkloadMeshSpec:
            raise SchemaError("must be a WorkloadMeshSpec", path="mesh")
        mesh.validate("mesh")
        if self.logical_rank_count > mesh.rank_count:
            raise SchemaError(
                "logical rank count exceeds mesh rank count", path=path
            )
        if self.active_die_ids:
            if len(self.active_die_ids) != self.logical_rank_count:
                raise SchemaError(
                    "must contain one die id per logical rank",
                    path=f"{path}.active_die_ids",
                )
            if len(set(self.active_die_ids)) != len(self.active_die_ids):
                raise SchemaError(
                    "contains a duplicate die id", path=f"{path}.active_die_ids"
                )
            for index, die_id in enumerate(self.active_die_ids):
                validate_uint64(die_id, f"{path}.active_die_ids[{index}]")
                if die_id >= mesh.rank_count:
                    raise SchemaError(
                        "die id lies outside the mesh",
                        path=f"{path}.active_die_ids[{index}]",
                    )

    @property
    def logical_rank_count(self) -> int:
        return self.tp * self.dp * self.ep * self.pp


@dataclass(frozen=True, slots=True)
class WorkloadMemoryPolicy:
    mode: WorkloadMemoryMode = WorkloadMemoryMode.RESIDENT_HBM
    allow_sram_spill: bool = True
    external_tier_ref: str | None = None

    def validate(self, path: str = "memory") -> None:
        if type(self.mode) is not WorkloadMemoryMode:
            raise SchemaError("must be a WorkloadMemoryMode", path=f"{path}.mode")
        if type(self.allow_sram_spill) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.allow_sram_spill")
        if self.mode is WorkloadMemoryMode.EXTERNAL_OFFLOAD:
            if self.external_tier_ref is None:
                raise SchemaError(
                    "is required for external_offload",
                    path=f"{path}.external_tier_ref",
                )
            validate_nonempty(self.external_tier_ref, f"{path}.external_tier_ref")
        elif self.external_tier_ref is not None:
            raise SchemaError(
                "must be absent unless mode is external_offload",
                path=f"{path}.external_tier_ref",
            )


@dataclass(frozen=True, slots=True)
class WorkloadOptimizerSpec:
    kind: WorkloadOptimizerKind
    learning_rate: float
    weight_decay: float = 0.0
    beta1: float | None = None
    beta2: float | None = None
    epsilon: float | None = None
    state_dtype: DType = DType.FP32

    def validate(self, path: str = "optimizer") -> None:
        if type(self.kind) is not WorkloadOptimizerKind:
            raise SchemaError("must be a WorkloadOptimizerKind", path=f"{path}.kind")
        _finite(self.learning_rate, f"{path}.learning_rate", positive=True)
        _finite(self.weight_decay, f"{path}.weight_decay")
        if self.state_dtype is not DType.FP32:
            raise UnsupportedFeatureError(
                "optimizer state must use fp32", path=f"{path}.state_dtype"
            )
        adam_fields = (self.beta1, self.beta2, self.epsilon)
        if self.kind is WorkloadOptimizerKind.SGD:
            if any(value is not None for value in adam_fields):
                raise SchemaError(
                    "SGD must not declare AdamW coefficients", path=path
                )
            return
        if any(value is None for value in adam_fields):
            raise SchemaError(
                "AdamW requires beta1, beta2, and epsilon", path=path
            )
        assert self.beta1 is not None and self.beta2 is not None
        assert self.epsilon is not None
        for name, value in (
            ("beta1", self.beta1),
            ("beta2", self.beta2),
            ("epsilon", self.epsilon),
        ):
            _finite(value, f"{path}.{name}", positive=True)
        if self.beta1 >= 1.0 or self.beta2 >= 1.0:
            raise SchemaError("beta coefficients must be less than one", path=path)


@dataclass(frozen=True, slots=True)
class WorkloadExecutionSpec:
    timing: bool = True
    functional: bool = False
    independent_repeats: int = 1
    strategy: WorkloadExecutionStrategy = WorkloadExecutionStrategy.BASELINE

    def validate(self, path: str = "execution") -> None:
        if type(self.timing) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.timing")
        if type(self.functional) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.functional")
        if not self.timing and not self.functional:
            raise SchemaError("at least one execution mode is required", path=path)
        _positive(self.independent_repeats, f"{path}.independent_repeats")
        if type(self.strategy) is not WorkloadExecutionStrategy:
            raise SchemaError(
                "must be a WorkloadExecutionStrategy", path=f"{path}.strategy"
            )


@dataclass(frozen=True, slots=True)
class WorkloadRunRequest:
    schema_version: str
    case_id: str
    family: WorkloadFamily
    model: WorkloadModelSpec
    steps: WorkloadStepSpec
    mesh: WorkloadMeshSpec
    parallel: WorkloadParallelSpec
    memory: WorkloadMemoryPolicy
    optimizer: WorkloadOptimizerSpec | None
    execution: WorkloadExecutionSpec

    @classmethod
    def create(
        cls,
        *,
        family: WorkloadFamily,
        model: WorkloadModelSpec,
        steps: WorkloadStepSpec,
        mesh: WorkloadMeshSpec,
        parallel: WorkloadParallelSpec,
        memory: WorkloadMemoryPolicy = WorkloadMemoryPolicy(),
        optimizer: WorkloadOptimizerSpec | None = None,
        execution: WorkloadExecutionSpec = WorkloadExecutionSpec(),
    ) -> "WorkloadRunRequest":
        semantic_key = {
            "family": family,
            "model": model,
            "steps": steps,
            "mesh": mesh,
            "parallel": parallel,
            "memory": memory,
            "optimizer": optimizer,
            "execution": execution,
        }
        result = cls(
            schema_version=WORKLOAD_RUN_REQUEST_SCHEMA_VERSION,
            case_id=stable_artifact_id(
                "workload_case",
                semantic_key,
                schema_version=WORKLOAD_RUN_REQUEST_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "family": self.family,
            "model": self.model,
            "steps": self.steps,
            "mesh": self.mesh,
            "parallel": self.parallel,
            "memory": self.memory,
            "optimizer": self.optimizer,
            "execution": self.execution,
        }

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "workload_run_request") -> None:
        if self.schema_version != WORKLOAD_RUN_REQUEST_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.family) is not WorkloadFamily:
            raise SchemaError("must be a WorkloadFamily", path=f"{path}.family")
        expected_id = stable_artifact_id(
            "workload_case",
            self._semantic_key(),
            schema_version=WORKLOAD_RUN_REQUEST_SCHEMA_VERSION,
        )
        if self.case_id != expected_id:
            raise SchemaError(
                f"unstable case id; expected {expected_id!r}", path=f"{path}.case_id"
            )
        self.model.validate(f"{path}.model")
        self.steps.validate(f"{path}.steps")
        self.mesh.validate(f"{path}.mesh")
        self.parallel.validate_against_mesh(self.mesh, f"{path}.parallel")
        self.memory.validate(f"{path}.memory")
        self.execution.validate(f"{path}.execution")

        if self.family.is_moe != (
            self.model.architecture is WorkloadModelArchitecture.LLAMA_MOE
        ):
            raise SchemaError(
                "family and model architecture disagree", path=f"{path}.model.architecture"
            )
        if self.family.is_training:
            if self.steps.training is None:
                raise SchemaError("training steps are required", path=f"{path}.steps")
            if self.optimizer is None:
                raise SchemaError("is required for training", path=f"{path}.optimizer")
            self.optimizer.validate(f"{path}.optimizer")
            expected_batch = (
                self.steps.training.micro_batch_size
                * self.steps.training.micro_batch_count
                * self.parallel.dp
            )
            if self.steps.training.global_batch_size != expected_batch:
                raise SchemaError(
                    "must equal micro_batch_size * micro_batch_count * dp",
                    path=f"{path}.steps.training.global_batch_size",
                )
            sequence_length = self.steps.training.sequence_length
        else:
            if self.steps.inference is None:
                raise SchemaError("inference steps are required", path=f"{path}.steps")
            if self.optimizer is not None:
                raise SchemaError("must be absent for inference", path=f"{path}.optimizer")
            sequence_length = (
                self.steps.inference.prefill_tokens + self.steps.inference.decode_steps
            )
        if sequence_length > self.model.max_sequence_length:
            raise SchemaError(
                "requested sequence exceeds model max_sequence_length",
                path=f"{path}.steps",
            )
        if not self.family.is_moe and self.parallel.ep != 1:
            raise SchemaError("Dense workloads require ep=1", path=f"{path}.parallel.ep")
        if self.family.is_moe:
            if self.model.num_experts % self.parallel.ep != 0:
                raise SchemaError(
                    "num_experts must be divisible by ep", path=f"{path}.parallel.ep"
                )
        for name, dimension in (
            ("hidden_size", self.model.hidden_size),
            ("intermediate_size", self.model.intermediate_size),
            ("num_attention_heads", self.model.num_attention_heads),
            ("num_kv_heads", self.model.num_kv_heads),
        ):
            if dimension % self.parallel.tp != 0:
                raise SchemaError(
                    f"model {name} must be divisible by tp",
                    path=f"{path}.parallel.tp",
                )
        if self.parallel.pp != 1:
            raise UnsupportedFeatureError(
                "v1alpha1 execution requires pp=1", path=f"{path}.parallel.pp"
            )


@dataclass(frozen=True, slots=True)
class WorkloadFamilyCapability:
    family: WorkloadFamily
    full_model: WorkloadCapabilityLevel
    motif: WorkloadCapabilityLevel
    baseline: WorkloadCapabilityLevel
    optimized: WorkloadCapabilityLevel
    lowering: WorkloadCapabilityLevel
    runtime: WorkloadCapabilityLevel
    timing: WorkloadCapabilityLevel
    functional: WorkloadCapabilityLevel
    capacity: WorkloadCapabilityLevel
    multi_step: WorkloadCapabilityLevel
    remote_hbm: WorkloadCapabilityLevel
    external_offload: WorkloadCapabilityLevel
    sgd_optimizer: WorkloadCapabilityLevel
    adamw_optimizer: WorkloadCapabilityLevel
    repeatability: WorkloadCapabilityLevel

    def validate(self, path: str = "family_capability") -> None:
        if type(self.family) is not WorkloadFamily:
            raise SchemaError("must be a WorkloadFamily", path=f"{path}.family")
        for name in (
            "full_model",
            "motif",
            "baseline",
            "optimized",
            "lowering",
            "runtime",
            "timing",
            "functional",
            "capacity",
            "multi_step",
            "remote_hbm",
            "external_offload",
            "sgd_optimizer",
            "adamw_optimizer",
            "repeatability",
        ):
            if type(getattr(self, name)) is not WorkloadCapabilityLevel:
                raise SchemaError(
                    "must be a WorkloadCapabilityLevel", path=f"{path}.{name}"
                )
        supported = WorkloadCapabilityLevel.SUPPORTED
        if self.full_model is supported and self.lowering is not supported:
            raise SchemaError(
                "full_model support requires lowering support", path=f"{path}.full_model"
            )
        if self.runtime is supported and self.lowering is not supported:
            raise SchemaError(
                "runtime support requires lowering support", path=f"{path}.runtime"
            )
        for name in ("timing", "functional", "multi_step"):
            if getattr(self, name) is supported and self.runtime is not supported:
                raise SchemaError(
                    f"{name} support requires runtime support", path=f"{path}.{name}"
                )
        if self.optimized is supported and self.baseline is not supported:
            raise SchemaError(
                "optimized support requires baseline support", path=f"{path}.optimized"
            )
        if self.repeatability is supported and self.runtime is not supported:
            raise SchemaError(
                "repeatability support requires runtime support",
                path=f"{path}.repeatability",
            )
        for name in ("remote_hbm", "external_offload"):
            if getattr(self, name) is supported and (
                self.capacity is not supported or self.runtime is not supported
            ):
                raise SchemaError(
                    f"{name} support requires capacity and runtime support",
                    path=f"{path}.{name}",
                )


@dataclass(frozen=True, slots=True)
class WorkloadRunCapability:
    schema_version: str
    id: str
    max_mesh_rows: int
    max_mesh_columns: int
    max_mesh_ranks: int
    families: tuple[WorkloadFamilyCapability, ...]

    @classmethod
    def create(
        cls,
        *,
        max_mesh_rows: int,
        max_mesh_columns: int,
        max_mesh_ranks: int,
        families: tuple[WorkloadFamilyCapability, ...],
    ) -> "WorkloadRunCapability":
        semantic_key = {
            "max_mesh_rows": max_mesh_rows,
            "max_mesh_columns": max_mesh_columns,
            "max_mesh_ranks": max_mesh_ranks,
            "families": families,
        }
        result = cls(
            schema_version=WORKLOAD_RUN_CAPABILITY_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_run_capability",
                semantic_key,
                schema_version=WORKLOAD_RUN_CAPABILITY_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "max_mesh_rows": self.max_mesh_rows,
            "max_mesh_columns": self.max_mesh_columns,
            "max_mesh_ranks": self.max_mesh_ranks,
            "families": self.families,
        }

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "workload_run_capability") -> None:
        if self.schema_version != WORKLOAD_RUN_CAPABILITY_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in ("max_mesh_rows", "max_mesh_columns", "max_mesh_ranks"):
            _positive(getattr(self, name), f"{path}.{name}")
        if self.max_mesh_ranks > self.max_mesh_rows * self.max_mesh_columns:
            raise SchemaError(
                "must not exceed max_mesh_rows * max_mesh_columns",
                path=f"{path}.max_mesh_ranks",
            )
        expected_families = tuple(WorkloadFamily)
        if tuple(item.family for item in self.families) != expected_families:
            raise SchemaError(
                "must contain all four families in canonical order",
                path=f"{path}.families",
            )
        for index, family in enumerate(self.families):
            family.validate(f"{path}.families[{index}]")
        expected_id = stable_artifact_id(
            "workload_run_capability",
            self._semantic_key(),
            schema_version=WORKLOAD_RUN_CAPABILITY_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable capability id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def unsupported_requirements(
        self, request: WorkloadRunRequest
    ) -> tuple[str, ...]:
        self.validate()
        request.validate()
        missing: list[str] = []
        if request.mesh.rows > self.max_mesh_rows:
            missing.append("mesh.rows")
        if request.mesh.columns > self.max_mesh_columns:
            missing.append("mesh.columns")
        if request.mesh.rank_count > self.max_mesh_ranks:
            missing.append("mesh.rank_count")
        family = self.families[tuple(WorkloadFamily).index(request.family)]
        required = ["full_model", "lowering", "runtime", "capacity"]
        required.append(request.execution.strategy.value)
        if request.execution.timing:
            required.append("timing")
        if request.execution.functional:
            required.append("functional")
        if request.family.is_training:
            assert request.steps.training is not None
            multi_step = request.steps.training.step_count > 1
        else:
            assert request.steps.inference is not None
            multi_step = request.steps.inference.decode_steps > 0
        if request.execution.independent_repeats > 1:
            required.append("repeatability")
        if multi_step:
            required.append("multi_step")
        if request.memory.mode is WorkloadMemoryMode.REMOTE_HBM:
            required.append("remote_hbm")
        if request.memory.mode is WorkloadMemoryMode.EXTERNAL_OFFLOAD:
            required.append("external_offload")
        if request.optimizer is not None:
            required.append(f"{request.optimizer.kind.value}_optimizer")
        for name in required:
            if getattr(family, name) is not WorkloadCapabilityLevel.SUPPORTED:
                missing.append(f"family.{name}")
        return tuple(missing)

    def require_supported(self, request: WorkloadRunRequest) -> None:
        missing = self.unsupported_requirements(request)
        if missing:
            raise UnsupportedFeatureError(
                f"request is outside capability: {', '.join(missing)}",
                path="workload_run_request",
            )


__all__ = [
    "WORKLOAD_RUN_CAPABILITY_SCHEMA_VERSION",
    "WORKLOAD_RUN_REQUEST_SCHEMA_VERSION",
    "WorkloadCapabilityLevel",
    "WorkloadExecutionSpec",
    "WorkloadFamily",
    "WorkloadFamilyCapability",
    "WorkloadInferenceSteps",
    "WorkloadMemoryMode",
    "WorkloadExecutionStrategy",
    "WorkloadMemoryPolicy",
    "WorkloadMeshSpec",
    "WorkloadModelArchitecture",
    "WorkloadModelSpec",
    "WorkloadOptimizerKind",
    "WorkloadOptimizerSpec",
    "WorkloadParallelSpec",
    "WorkloadRankOrder",
    "WorkloadRoutePolicy",
    "WorkloadRunCapability",
    "WorkloadRunRequest",
    "WorkloadStepSpec",
    "WorkloadTrainingSteps",
]
