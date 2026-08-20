"""Profile-independent Dense templates and profile-expanded IR-0 bundles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .experiment import InferOutput
from .common import (
    DType,
    MeshAxisName,
    ProfileKey,
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
)
from .ir0 import IR0, JobKind, LogicalInstance, LogicalRole
from .stage3_profile import Stage3ProfileMode, Stage3StaticProfile


IR0_TEMPLATE_SCHEMA_VERSION = "wafer_frontend.ir0_template/v1alpha3"
EXPANDED_IR0_BUNDLE_SCHEMA_VERSION = (
    "wafer_frontend.expanded_ir0_bundle/v1alpha5"
)


class DenseLayerKind(str, Enum):
    LLAMA_DENSE_BLOCK_V1 = "llama_dense_block_v1"


class NormKind(str, Enum):
    RMS_NORM = "rms_norm"


class ElementwiseKind(str, Enum):
    SWIGLU = "swiglu"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


def _derived_uint64(value: int, path: str) -> int:
    validate_uint64(value, path)
    return value


@dataclass(frozen=True, slots=True)
class DenseModelShape:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    dtype: DType
    tie_word_embeddings: bool
    rms_norm_epsilon: float
    rope_theta: float
    max_position_embeddings: int

    def validate(self, path: str = "model") -> None:
        for field_name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "rotary_dim",
            "max_position_embeddings",
        ):
            _positive(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.tie_word_embeddings) is not bool:
            raise SchemaError(
                "must be a bool", path=f"{path}.tie_word_embeddings"
            )
        if self.tie_word_embeddings:
            raise SchemaError(
                "Stage 2 supports untied embedding/head only",
                path=f"{path}.tie_word_embeddings",
            )
        for field_name in ("rms_norm_epsilon", "rope_theta"):
            value = getattr(self, field_name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise SchemaError(
                    "must be a finite positive float",
                    path=f"{path}.{field_name}",
                )
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        if self.dtype is not DType.FP16:
            raise SchemaError("Dense v1 supports fp16 only", path=f"{path}.dtype")
        hidden_from_heads = _derived_uint64(
            self.num_heads * self.head_dim,
            f"{path}.derived.hidden_size",
        )
        if self.hidden_size != hidden_from_heads:
            raise SchemaError(
                "must equal num_heads * head_dim", path=f"{path}.hidden_size"
            )
        if self.rotary_dim != self.head_dim:
            raise SchemaError(
                "Stage 2 requires rotary_dim == head_dim",
                path=f"{path}.rotary_dim",
            )
        if self.num_kv_heads > self.num_heads:
            raise SchemaError(
                "must not exceed num_heads", path=f"{path}.num_kv_heads"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise SchemaError(
                "num_heads must be divisible by num_kv_heads",
                path=f"{path}.num_kv_heads",
            )
        q_width = _derived_uint64(
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            f"{path}.derived.qkv_width",
        )
        qkv_params = _derived_uint64(
            self.hidden_size * q_width, f"{path}.derived.qkv_params"
        )
        output_params = _derived_uint64(
            self.num_heads * self.head_dim * self.hidden_size,
            f"{path}.derived.output_params",
        )
        gate_up_params = _derived_uint64(
            2 * self.hidden_size * self.intermediate_size,
            f"{path}.derived.gate_up_params",
        )
        down_params = _derived_uint64(
            self.intermediate_size * self.hidden_size,
            f"{path}.derived.down_params",
        )
        per_layer_params = _derived_uint64(
            qkv_params + output_params + gate_up_params + down_params,
            f"{path}.derived.per_layer_params",
        )
        per_layer_with_norm = _derived_uint64(
            per_layer_params + 2 * self.hidden_size,
            f"{path}.derived.per_layer_params_with_norm",
        )
        layers = _derived_uint64(
            per_layer_with_norm * self.num_layers,
            f"{path}.derived.total_layer_params",
        )
        embedding = _derived_uint64(
            self.vocab_size * self.hidden_size,
            f"{path}.derived.embedding_params",
        )
        lm_head = _derived_uint64(
            self.hidden_size * self.vocab_size,
            f"{path}.derived.lm_head_params",
        )
        _derived_uint64(
            embedding + layers + self.hidden_size + lm_head,
            f"{path}.derived.total_params",
        )

    def parameter_elements(self) -> int:
        self.validate("model")
        q_width = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        per_layer_linear = (
            self.hidden_size * q_width
            + self.hidden_size * self.hidden_size
            + 3 * self.hidden_size * self.intermediate_size
        )
        return (
            self.vocab_size * self.hidden_size
            + self.num_layers * (per_layer_linear + 2 * self.hidden_size)
            + self.hidden_size
            + self.hidden_size * self.vocab_size
        )

    def parameter_bytes(self) -> int:
        return self.parameter_elements() * 2


@dataclass(frozen=True, slots=True)
class DenseLayerTemplate:
    id: str
    kind: DenseLayerKind
    norm: NormKind
    activation: ElementwiseKind
    has_bias: bool

    def validate(self, path: str = "layer") -> None:
        validate_nonempty(self.id, f"{path}.id")
        if type(self.kind) is not DenseLayerKind:
            raise SchemaError("must be a DenseLayerKind", path=f"{path}.kind")
        if type(self.norm) is not NormKind:
            raise SchemaError("must be a NormKind", path=f"{path}.norm")
        if type(self.activation) is not ElementwiseKind:
            raise SchemaError(
                "must be an ElementwiseKind", path=f"{path}.activation"
            )
        if type(self.has_bias) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.has_bias")
        if self.has_bias:
            raise SchemaError(
                "Llama Dense v1 does not contain GEMM bias", path=f"{path}.has_bias"
            )


@dataclass(frozen=True, slots=True)
class SequenceParallelSpec:
    enabled: bool
    reuse_axis: MeshAxisName | None

    def validate(self, path: str = "sequence_parallel") -> None:
        if type(self.enabled) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.enabled")
        expected_axis = MeshAxisName.TP if self.enabled else None
        if self.reuse_axis is not expected_axis:
            expected = "tp" if self.enabled else "null"
            raise SchemaError(
                f"must be {expected} when enabled={self.enabled!r}",
                path=f"{path}.reuse_axis",
            )


@dataclass(frozen=True, slots=True)
class ProfileEntry:
    profile_id: str
    key: ProfileKey
    weight: float
    exact_profile: Stage3StaticProfile | None = None

    @classmethod
    def create(
        cls,
        *,
        key: ProfileKey,
        weight: float,
        exact_profile: Stage3StaticProfile | None = None,
    ) -> "ProfileEntry":
        return cls(
            profile_id=key.stable_id(),
            key=key,
            weight=weight,
            exact_profile=exact_profile,
        )

    def validate(self, path: str = "profile") -> None:
        self.key.validate(f"{path}.key")
        expected_id = self.key.stable_id()
        if self.profile_id != expected_id:
            raise SchemaError(
                f"unstable profile id; expected {expected_id!r}",
                path=f"{path}.profile_id",
            )
        if self.exact_profile is not None:
            if type(self.exact_profile) is not Stage3StaticProfile:
                raise SchemaError(
                    "must be a Stage3StaticProfile or null",
                    path=f"{path}.exact_profile",
                )
            self.exact_profile.validate(f"{path}.exact_profile")
            if self.exact_profile.key != self.key:
                raise SchemaError(
                    "key must equal exact_profile.key",
                    path=f"{path}.exact_profile.key",
                )
        if (
            type(self.weight) is not float
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise SchemaError(
                "must be a finite positive float", path=f"{path}.weight"
            )


def _validate_profile_manifest(
    profiles: tuple[ProfileEntry, ...], *, path: str
) -> None:
    if not profiles:
        raise SchemaError("must contain at least one profile", path=path)
    previous_id: str | None = None
    for index, profile in enumerate(profiles):
        profile.validate(f"{path}[{index}]")
        if previous_id is not None and profile.profile_id <= previous_id:
            raise SchemaError(
                "profile_id values must be strictly increasing",
                path=f"{path}[{index}].profile_id",
            )
        previous_id = profile.profile_id
    if abs(math.fsum(profile.weight for profile in profiles) - 1.0) > 1e-12:
        raise SchemaError("weights must sum to 1 within 1e-12", path=path)


@dataclass(frozen=True, slots=True)
class IR0Template:
    schema_version: str
    producer_pass: str
    id: str
    job: JobKind
    model: DenseModelShape
    instance: LogicalInstance
    sequence_parallel: SequenceParallelSpec
    layer: DenseLayerTemplate
    infer_output: InferOutput
    profiles: tuple[ProfileEntry, ...]

    @classmethod
    def create(
        cls,
        *,
        job: JobKind,
        model: DenseModelShape,
        instance: LogicalInstance,
        sequence_parallel: SequenceParallelSpec,
        layer: DenseLayerTemplate,
        infer_output: InferOutput,
        profiles: tuple[ProfileEntry, ...],
    ) -> "IR0Template":
        semantic_key = {
            "job": job,
            "model": model,
            "instance": instance,
            "sequence_parallel": sequence_parallel,
            "layer": layer,
            "infer_output": infer_output,
            "profiles": profiles,
        }
        return cls(
            schema_version=IR0_TEMPLATE_SCHEMA_VERSION,
            producer_pass="build_ir0",
            id=stable_artifact_id(
                "ir0_template",
                semantic_key,
                schema_version=IR0_TEMPLATE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "job": self.job,
            "model": self.model,
            "instance": self.instance,
            "sequence_parallel": self.sequence_parallel,
            "layer": self.layer,
            "infer_output": self.infer_output,
            "profiles": self.profiles,
        }

    def validate(self, path: str = "ir0_template") -> None:
        if self.schema_version != IR0_TEMPLATE_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "build_ir0":
            raise SchemaError("must be 'build_ir0'", path=f"{path}.producer_pass")
        if type(self.job) is not JobKind or self.job is not JobKind.INFER:
            raise SchemaError("Dense v1 supports infer only", path=f"{path}.job")
        expected_id = stable_artifact_id(
            "ir0_template",
            self._semantic_key(),
            schema_version=IR0_TEMPLATE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )
        self.model.validate(f"{path}.model")
        self.instance.validate(f"{path}.instance")
        if self.instance.role is LogicalRole.TRAIN:
            raise SchemaError("Dense v1 supports infer only", path=f"{path}.instance.role")
        if self.instance.replicas != 1:
            raise SchemaError("must equal 1", path=f"{path}.instance.replicas")
        parallel = self.instance.parallel
        for axis in ("dp", "pp", "ep"):
            if getattr(parallel, axis) != 1:
                raise SchemaError("must equal 1", path=f"{path}.instance.parallel.{axis}")
        self.sequence_parallel.validate(f"{path}.sequence_parallel")
        if parallel.sp is not self.sequence_parallel.enabled:
            raise SchemaError(
                "must agree with instance.parallel.sp",
                path=f"{path}.sequence_parallel.enabled",
            )
        if len(self.instance.meshes) != 1:
            raise SchemaError(
                "Dense v1 requires exactly one TP mesh",
                path=f"{path}.instance.meshes",
            )
        mesh = self.instance.meshes[0]
        if len(mesh.axes) != 1 or mesh.axes[0].name is not MeshAxisName.TP:
            raise SchemaError(
                "Dense v1 mesh must contain only the TP axis",
                path=f"{path}.instance.meshes[0].axes",
            )
        if mesh.axes[0].size != parallel.tp:
            raise SchemaError(
                "must equal instance.parallel.tp",
                path=f"{path}.instance.meshes[0].axes[0].size",
            )
        self.layer.validate(f"{path}.layer")
        if type(self.infer_output) is not InferOutput:
            raise SchemaError(
                "must be an InferOutput", path=f"{path}.infer_output"
            )
        _validate_profile_manifest(self.profiles, path=f"{path}.profiles")
        for index, profile in enumerate(self.profiles):
            if profile.key.expert_load is not None:
                raise SchemaError(
                    "must be null for a Dense template",
                    path=f"{path}.profiles[{index}].key.expert_load",
                )
            exact = profile.exact_profile
            if exact is None:
                continue
            allowed_roles = {
                Stage3ProfileMode.PREFILL: (
                    LogicalRole.PREFILL,
                    LogicalRole.BOTH,
                ),
                Stage3ProfileMode.DECODE: (
                    LogicalRole.DECODE,
                    LogicalRole.BOTH,
                ),
                Stage3ProfileMode.MIXED: (LogicalRole.BOTH,),
            }[exact.mode]
            if self.instance.role not in allowed_roles:
                raise SchemaError(
                    "exact profile mode is incompatible with instance role",
                    path=f"{path}.instance.role",
                )


@dataclass(frozen=True, slots=True)
class ExpandedProfileIR0:
    id: str
    source_template_id: str
    profile_id: str
    weight: float
    graph: IR0

    @classmethod
    def create(
        cls,
        *,
        source_template_id: str,
        weight: float,
        graph: IR0,
    ) -> "ExpandedProfileIR0":
        profile_id = graph.profile.stable_id()
        semantic_key = {
            "source_template_id": source_template_id,
            "profile_id": profile_id,
            "weight": weight,
            "graph_id": graph.id,
        }
        return cls(
            id=stable_artifact_id(
                "expanded_profile_ir0",
                semantic_key,
                schema_version=EXPANDED_IR0_BUNDLE_SCHEMA_VERSION,
            ),
            source_template_id=source_template_id,
            profile_id=profile_id,
            weight=weight,
            graph=graph,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_template_id": self.source_template_id,
            "profile_id": self.profile_id,
            "weight": self.weight,
            "graph_id": self.graph.id,
        }

    def validate(self, path: str = "entry") -> None:
        validate_nonempty(self.source_template_id, f"{path}.source_template_id")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "logical_expand":
            raise SchemaError(
                "must be produced by logical_expand",
                path=f"{path}.graph.producer_pass",
            )
        if not self.graph.nodes or not self.graph.values:
            raise SchemaError(
                "logical_expand must materialize a non-empty graph",
                path=f"{path}.graph.nodes",
            )
        expected_profile_id = self.graph.profile.stable_id()
        if self.profile_id != expected_profile_id:
            raise SchemaError(
                f"must equal graph profile id {expected_profile_id!r}",
                path=f"{path}.profile_id",
            )
        if (
            type(self.weight) is not float
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise SchemaError(
                "must be a finite positive float", path=f"{path}.weight"
            )
        expected_id = stable_artifact_id(
            "expanded_profile_ir0",
            self._semantic_key(),
            schema_version=EXPANDED_IR0_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable entry id; expected {expected_id!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class ExpandedIR0Bundle:
    schema_version: str
    producer_pass: str
    id: str
    source_template_id: str
    source_profiles: tuple[ProfileEntry, ...]
    entries: tuple[ExpandedProfileIR0, ...]

    @classmethod
    def create(
        cls,
        *,
        source_template: IR0Template,
        entries: tuple[ExpandedProfileIR0, ...],
    ) -> "ExpandedIR0Bundle":
        source_template.validate("source_template")
        for index, entry in enumerate(entries):
            if entry.graph.job is not source_template.job:
                raise SchemaError(
                    "graph job must match source template job",
                    path=f"entries[{index}].graph.job",
                )
        semantic_key = {
            "source_template_id": source_template.id,
            "source_profiles": source_template.profiles,
            "entries": entries,
        }
        return cls(
            schema_version=EXPANDED_IR0_BUNDLE_SCHEMA_VERSION,
            producer_pass="logical_expand",
            id=stable_artifact_id(
                "expanded_ir0_bundle",
                semantic_key,
                schema_version=EXPANDED_IR0_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_template_id": self.source_template_id,
            "source_profiles": self.source_profiles,
            "entries": self.entries,
        }

    def validate(self, path: str = "expanded_ir0_bundle") -> None:
        if self.schema_version != EXPANDED_IR0_BUNDLE_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "logical_expand":
            raise SchemaError(
                "must be 'logical_expand'", path=f"{path}.producer_pass"
            )
        validate_nonempty(self.source_template_id, f"{path}.source_template_id")
        _validate_profile_manifest(
            self.source_profiles, path=f"{path}.source_profiles"
        )
        if len(self.entries) != len(self.source_profiles):
            raise SchemaError(
                "must contain exactly one graph per source profile",
                path=f"{path}.entries",
            )
        previous_id: str | None = None
        graph_ids: set[str] = set()
        for index, (source_profile, entry) in enumerate(
            zip(self.source_profiles, self.entries)
        ):
            entry.validate(f"{path}.entries[{index}]")
            if previous_id is not None and entry.profile_id <= previous_id:
                raise SchemaError(
                    "profile_id values must be strictly increasing",
                    path=f"{path}.entries[{index}].profile_id",
                )
            previous_id = entry.profile_id
            if entry.source_template_id != self.source_template_id:
                raise SchemaError(
                    "must match bundle source_template_id",
                    path=f"{path}.entries[{index}].source_template_id",
                )
            if entry.profile_id != source_profile.profile_id:
                raise SchemaError(
                    "must match the corresponding template profile",
                    path=f"{path}.entries[{index}].profile_id",
                )
            if entry.graph.profile != source_profile.key:
                raise SchemaError(
                    "graph profile must match the corresponding template profile",
                    path=f"{path}.entries[{index}].graph.profile",
                )
            if entry.weight != source_profile.weight:
                raise SchemaError(
                    "must match the corresponding template profile weight",
                    path=f"{path}.entries[{index}].weight",
                )
            if entry.graph.id in graph_ids:
                raise SchemaError(
                    "each profile must own an independent IR0 graph",
                    path=f"{path}.entries[{index}].graph.id",
                )
            graph_ids.add(entry.graph.id)
        expected_id = stable_artifact_id(
            "expanded_ir0_bundle",
            self._semantic_key(),
            schema_version=EXPANDED_IR0_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )
