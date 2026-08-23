"""Typed C0-C4 scale truth for the 4-Die LiteMoE Swizzle workload."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .lite_moe import LiteMoeStaticTrace
from .lite_moe_dp4 import LiteMoeDp4Spec, LiteMoeDp4Topology
from .serde import canonical_digest


MOE_SWIZZLE_SCALE_SPEC_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_scale_spec/v1alpha1"
)
MOE_SWIZZLE_SCALE_ORACLE_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_scale_oracle/v1alpha1"
)


class MoeSwizzleTraceFamily(str, Enum):
    BALANCED = "balanced"
    SKEWED = "skewed"


class MoeSwizzleScaleRole(str, Enum):
    CONTROL = "control"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    CAPACITY = "capacity"


class MoeSwizzleExecutionStatus(str, Enum):
    PRODUCTION_READY = "production_ready"
    CAPACITY_PROBE = "capacity_probe"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be positive", path=path)


def _dtype_bytes(dtype: DType, path: str) -> int:
    if dtype is not DType.FP16:
        raise SchemaError("V2 scale truth currently requires production FP16", path=path)
    return 2


@dataclass(frozen=True, slots=True)
class MoeSwizzleScaleSpec:
    """One immutable trace/profile point derived from production DP4 truth."""

    schema_version: str
    producer_pass: str
    id: str
    name: str
    role: MoeSwizzleScaleRole
    trace_family: MoeSwizzleTraceFamily
    execution_status: MoeSwizzleExecutionStatus
    source_c0_spec_id: str
    source_c0_spec_digest: str
    source_c0_topology_id: str
    source_c0_topology_digest: str
    tokens: int
    hidden_size: int
    intermediate_size: int
    expert_count: int
    top_k: int
    capacity_per_expert: int
    mesh_rows: int
    mesh_columns: int
    dtype: DType
    trace: LiteMoeStaticTrace
    token_source_die_ids: tuple[int, ...]
    expert_home_die_ids: tuple[int, ...]

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleScaleSpec":
        result = cls(
            schema_version=MOE_SWIZZLE_SCALE_SPEC_SCHEMA_VERSION,
            producer_pass="moe_swizzle_scale_spec",
            id=stable_artifact_id(
                "moe_swizzle_scale_spec",
                semantic,
                schema_version=MOE_SWIZZLE_SCALE_SPEC_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "moe_swizzle_scale_spec") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_SCALE_SPEC_SCHEMA_VERSION
            or self.producer_pass != "moe_swizzle_scale_spec"
        ):
            raise SchemaError("unsupported scale spec schema/producer", path=path)
        validate_nonempty(self.name, f"{path}.name")
        for name in (
            "source_c0_spec_id",
            "source_c0_spec_digest",
            "source_c0_topology_id",
            "source_c0_topology_digest",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in (
            "tokens", "hidden_size", "intermediate_size", "expert_count",
            "top_k", "capacity_per_expert", "mesh_rows", "mesh_columns",
        ):
            _positive(getattr(self, name), f"{path}.{name}")
        if type(self.role) is not MoeSwizzleScaleRole:
            raise SchemaError("must use a typed scale role", path=f"{path}.role")
        if type(self.trace_family) is not MoeSwizzleTraceFamily:
            raise SchemaError("must use a typed trace family", path=f"{path}.trace_family")
        if type(self.execution_status) is not MoeSwizzleExecutionStatus:
            raise SchemaError("must use a typed execution status", path=f"{path}.execution_status")
        if (
            self.expert_count != 4
            or self.top_k != 1
            or (self.mesh_rows, self.mesh_columns) != (2, 2)
            or self.hidden_size != 16
            or self.intermediate_size != 32
        ):
            raise SchemaError("must preserve production EP4/H16/I32 geometry", path=path)
        _dtype_bytes(self.dtype, f"{path}.dtype")
        if type(self.trace) is not LiteMoeStaticTrace:
            raise SchemaError("must carry a LiteMoeStaticTrace", path=f"{path}.trace")
        self.trace.validate(f"{path}.trace")
        if self.trace.token_count != self.tokens:
            raise SchemaError("trace/token count drifted", path=f"{path}.trace")
        if (
            len(self.token_source_die_ids) != self.tokens
            or len(self.expert_home_die_ids) != self.expert_count
            or any(item >= 4 for item in self.token_source_die_ids)
            or any(item >= 4 for item in self.expert_home_die_ids)
        ):
            raise SchemaError("source/home die coverage is not exact EP4", path=path)
        histogram = self.trace.expert_histogram
        if max(histogram) != self.capacity_per_expert:
            raise SchemaError("capacity must equal the concrete trace peak", path=f"{path}.capacity_per_expert")
        if self.trace_family is MoeSwizzleTraceFamily.BALANCED:
            if len(set(histogram)) != 1:
                raise SchemaError("balanced trace must have equal expert counts", path=f"{path}.trace")
        elif len(set(histogram)) == 1:
            raise SchemaError("skewed trace must be observably nonuniform", path=f"{path}.trace")
        expected_status = (
            MoeSwizzleExecutionStatus.CAPACITY_PROBE
            if self.role is MoeSwizzleScaleRole.CAPACITY
            else MoeSwizzleExecutionStatus.PRODUCTION_READY
        )
        if self.execution_status is not expected_status:
            raise SchemaError(
                "C0-C3 require the generalized production carrier and C4 is a capacity probe",
                path=f"{path}.execution_status",
            )
        expected = stable_artifact_id(
            "moe_swizzle_scale_spec",
            self._semantic_key(),
            schema_version=MOE_SWIZZLE_SCALE_SPEC_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(
        self,
        c0_spec: LiteMoeDp4Spec,
        c0_topology: LiteMoeDp4Topology,
        path: str = "moe_swizzle_scale_spec",
    ) -> None:
        self.validate(path)
        c0_spec.validate(f"{path}.c0_spec")
        c0_topology.validate_against(c0_spec, f"{path}.c0_topology")
        if (
            self.source_c0_spec_id,
            self.source_c0_spec_digest,
            self.source_c0_topology_id,
            self.source_c0_topology_digest,
        ) != (
            c0_spec.id,
            canonical_digest(c0_spec),
            c0_topology.id,
            canonical_digest(c0_topology),
        ):
            raise SchemaError("C0 production provenance drifted", path=path)
        if (
            self.hidden_size,
            self.intermediate_size,
            self.expert_count,
            self.top_k,
            self.dtype,
            self.expert_home_die_ids,
        ) != (
            c0_spec.hidden_size,
            c0_spec.intermediate_size,
            c0_spec.expert_count,
            c0_spec.top_k,
            c0_spec.dtype,
            c0_topology.expert_home_die_ids,
        ):
            raise SchemaError("scale geometry is not production-derived", path=path)
        expected_sources = tuple(
            c0_topology.token_source_die_ids[index % c0_spec.trace.token_count]
            for index in range(self.tokens)
        )
        if self.token_source_die_ids != expected_sources:
            raise SchemaError("token sources are not a production-trace extension", path=f"{path}.token_source_die_ids")
        if self.name == "C0" and (
            self.trace != c0_spec.trace
            or self.capacity_per_expert != c0_spec.capacity_per_expert
            or self.token_source_die_ids != c0_topology.token_source_die_ids
        ):
            raise SchemaError("C0 must reuse the exact production trace", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleScaleOracle:
    """Same-work totals rebuilt exclusively from one typed scale spec."""

    schema_version: str
    producer_pass: str
    id: str
    source_scale_spec_id: str
    source_scale_spec_digest: str
    trace_digest: str
    assignment_count: int
    contributor_count: int
    expert_token_counts: tuple[int, ...]
    expert_gemm_flops: tuple[int, ...]
    total_expert_gemm_flops: int
    remote_token_indices: tuple[int, ...]
    dispatch_logical_bytes: int
    combine_logical_bytes: int
    logical_p2p_bytes: int
    data_packets: int
    combined_terminal_count: int
    combined_terminal_bytes: int
    train_tape_terminal_count: int
    train_tape_terminal_bytes: int

    @classmethod
    def create(cls, spec: MoeSwizzleScaleSpec) -> "MoeSwizzleScaleOracle":
        spec.validate()
        dtype_bytes = _dtype_bytes(spec.dtype, "spec.dtype")
        token_bytes = spec.hidden_size * dtype_bytes
        remote = tuple(
            assignment.token_index
            for assignment in spec.trace.assignments
            if spec.token_source_die_ids[assignment.token_index]
            != spec.expert_home_die_ids[assignment.expert_index]
        )
        counts = spec.trace.expert_histogram
        flops = tuple(
            6 * count * spec.hidden_size * spec.intermediate_size
            for count in counts
        )
        dispatch = len(remote) * token_bytes
        combine = dispatch
        semantic = {
            "source_scale_spec_id": spec.id,
            "source_scale_spec_digest": canonical_digest(spec),
            "trace_digest": canonical_digest(spec.trace),
            "assignment_count": spec.tokens,
            "contributor_count": spec.tokens * spec.top_k,
            "expert_token_counts": counts,
            "expert_gemm_flops": flops,
            "total_expert_gemm_flops": sum(flops),
            "remote_token_indices": remote,
            "dispatch_logical_bytes": dispatch,
            "combine_logical_bytes": combine,
            "logical_p2p_bytes": dispatch + combine,
            "data_packets": (dispatch + combine) // 16,
            "combined_terminal_count": spec.tokens,
            "combined_terminal_bytes": spec.tokens * token_bytes,
            "train_tape_terminal_count": spec.tokens,
            "train_tape_terminal_bytes": (
                spec.tokens * spec.intermediate_size * dtype_bytes
            ),
        }
        result = cls(
            schema_version=MOE_SWIZZLE_SCALE_ORACLE_SCHEMA_VERSION,
            producer_pass="moe_swizzle_scale_oracle",
            id=stable_artifact_id(
                "moe_swizzle_scale_oracle",
                semantic,
                schema_version=MOE_SWIZZLE_SCALE_ORACLE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate_against(spec)
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "moe_swizzle_scale_oracle") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_SCALE_ORACLE_SCHEMA_VERSION
            or self.producer_pass != "moe_swizzle_scale_oracle"
        ):
            raise SchemaError("unsupported scale oracle schema/producer", path=path)
        for name in (
            "source_scale_spec_id", "source_scale_spec_digest", "trace_digest",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in (
            "assignment_count", "contributor_count", "total_expert_gemm_flops",
            "dispatch_logical_bytes", "combine_logical_bytes", "logical_p2p_bytes",
            "data_packets", "combined_terminal_count", "combined_terminal_bytes",
            "train_tape_terminal_count", "train_tape_terminal_bytes",
        ):
            _positive(getattr(self, name), f"{path}.{name}")
        if len(self.expert_token_counts) != 4 or len(self.expert_gemm_flops) != 4:
            raise SchemaError("oracle must cover four experts", path=path)
        if self.total_expert_gemm_flops != sum(self.expert_gemm_flops):
            raise SchemaError("expert FLOP total drifted", path=f"{path}.total_expert_gemm_flops")
        if self.logical_p2p_bytes != self.dispatch_logical_bytes + self.combine_logical_bytes:
            raise SchemaError("P2P logical bytes do not close", path=f"{path}.logical_p2p_bytes")
        expected = stable_artifact_id(
            "moe_swizzle_scale_oracle",
            self._semantic_key(),
            schema_version=MOE_SWIZZLE_SCALE_ORACLE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(
        self,
        spec: MoeSwizzleScaleSpec,
        path: str = "moe_swizzle_scale_oracle",
    ) -> None:
        self.validate(path)
        spec.validate(f"{path}.spec")
        if (
            self.source_scale_spec_id != spec.id
            or self.source_scale_spec_digest != canonical_digest(spec)
            or self.trace_digest != canonical_digest(spec.trace)
        ):
            raise SchemaError("oracle/spec provenance drifted", path=path)
        dtype_bytes = _dtype_bytes(spec.dtype, f"{path}.spec.dtype")
        token_bytes = spec.hidden_size * dtype_bytes
        remote = tuple(
            assignment.token_index
            for assignment in spec.trace.assignments
            if spec.token_source_die_ids[assignment.token_index]
            != spec.expert_home_die_ids[assignment.expert_index]
        )
        counts = spec.trace.expert_histogram
        flops = tuple(6 * count * spec.hidden_size * spec.intermediate_size for count in counts)
        rebuilt = (
            spec.tokens,
            spec.tokens * spec.top_k,
            counts,
            flops,
            sum(flops),
            remote,
            len(remote) * token_bytes,
            len(remote) * token_bytes,
            2 * len(remote) * token_bytes,
            2 * len(remote) * token_bytes // 16,
            spec.tokens,
            spec.tokens * token_bytes,
            spec.tokens,
            spec.tokens * spec.intermediate_size * dtype_bytes,
        )
        actual = (
            self.assignment_count,
            self.contributor_count,
            self.expert_token_counts,
            self.expert_gemm_flops,
            self.total_expert_gemm_flops,
            self.remote_token_indices,
            self.dispatch_logical_bytes,
            self.combine_logical_bytes,
            self.logical_p2p_bytes,
            self.data_packets,
            self.combined_terminal_count,
            self.combined_terminal_bytes,
            self.train_tape_terminal_count,
            self.train_tape_terminal_bytes,
        )
        if actual != rebuilt:
            raise SchemaError("oracle does not exactly rebuild typed trace work", path=path)


__all__ = [name for name in globals() if name.startswith("MoeSwizzle") or name.startswith("MOE_SWIZZLE")]
