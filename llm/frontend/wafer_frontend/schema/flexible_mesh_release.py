"""Strict, versioned release evidence for the 100 rectangular Mesh shapes.

This module intentionally models observations, not compiler success.  A release
case is runtime verified only when two independently identified executions are
bound to the same allowlisted tool/config profile and both carry exact runtime
closure evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .flexible_mesh_workload import FlexibleMeshWorkloadKind
from .rect_mesh import RectMeshSpec
from .serde import canonical_digest


FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_release/v1alpha1"
)
FLEXIBLE_MESH_RELEASE_CAPACITY_POLICY_VERSION = (
    "wafer_frontend.flexible_mesh_release_capacity/v2"
)
FLEXIBLE_MESH_RELEASE_MAX_RECORDS = 1_048_576
FLEXIBLE_MESH_RELEASE_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
# Linked JSON is a pre-finalizer carrier, not the executable artifact. The
# executable remains subject to the independent 64 MiB hard limit above.
FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES = 256 * 1024 * 1024
FLEXIBLE_MESH_RELEASE_MAX_SESSIONS_PER_CORE_PER_WAVE = 3
FLEXIBLE_MESH_RELEASE_MAX_RUNTIME_CORE_ID = 0xFFFF
FLEXIBLE_MESH_RELEASE_MAX_TRANSPORT_TAGS = 65_535
FLEXIBLE_MESH_RELEASE_SHAPE_COUNT = 100
FLEXIBLE_MESH_RELEASE_CASE_COUNT = 600
FLEXIBLE_MESH_RELEASE_EXECUTION_COUNT = 1_200


def _validate_sha256(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _validate_exact_enum_tuple(value: object, enum_type: type[Enum], path: str) -> None:
    expected = tuple(enum_type)
    if type(value) is not tuple or value != expected:
        raise SchemaError("must contain every value once in canonical order", path=path)


class FlexibleMeshReleaseFamily(str, Enum):
    DENSE_TRAIN = "dense_train"
    MOE_INFERENCE = "moe_inference"
    MOE_TRAIN = "moe_train"
    MESHSLICE_AG = "meshslice_ag"
    MESHSLICE_RS_FALLBACK = "meshslice_rs_fallback"
    MESHSLICE_AR_FALLBACK = "meshslice_ar_fallback"


class FlexibleMeshReleaseOperation(str, Enum):
    DENSE_TRAIN_STEP = "dense_train_step"
    MOE_INFERENCE_STEP = "moe_inference_step"
    MOE_TRAIN_STEP = "moe_train_step"
    AG_GEMM = "ag_gemm"
    GEMM_RS = "gemm_rs"
    GEMM_AR = "gemm_ar"


class FlexibleMeshReleaseBaseline(str, Enum):
    DENSE_TIMING_BASELINE = "dense_timing_baseline"
    MOE_DIRECT_XY = "moe_direct_xy"
    MESHSLICE_STANDARD = "meshslice_standard"
    UNFUSED_FALLBACK = "unfused_fallback"


class FlexibleMeshReleaseToolKind(str, Enum):
    FINALIZER = "finalizer"
    RESOLVER = "resolver"
    NPUSIM = "npusim"


class FlexibleMeshReleaseRuntimeStage(str, Enum):
    FINALIZER = "finalizer"
    RESOLVER = "resolver"
    NPUSIM = "npusim"


class FlexibleMeshProgramIOPhase(str, Enum):
    RESOLVED = "resolved"
    APPLIED_ACTUAL_ARTIFACT_SHA = "applied_actual_artifact_sha"
    VERIFY_PASS = "verify_pass"


class FlexibleMeshCompletionMarker(str, Enum):
    COMPLETE_STEP_STATE = "complete_step_state"
    DISPATCH_COMBINE = "dispatch_combine"
    FOUR_WAY_ROUTE = "four_way_route"
    GRADIENT = "gradient"
    OPTIMIZER = "optimizer"
    STATE = "state"
    MESHSLICE_OPERATION = "meshslice_operation"


_FAMILY_PROFILE: dict[
    FlexibleMeshReleaseFamily,
    tuple[
        FlexibleMeshWorkloadKind,
        FlexibleMeshReleaseOperation,
        FlexibleMeshReleaseBaseline,
        tuple[FlexibleMeshCompletionMarker, ...],
    ],
] = {
    FlexibleMeshReleaseFamily.DENSE_TRAIN: (
        FlexibleMeshWorkloadKind.DENSE_TRAIN,
        FlexibleMeshReleaseOperation.DENSE_TRAIN_STEP,
        FlexibleMeshReleaseBaseline.DENSE_TIMING_BASELINE,
        (FlexibleMeshCompletionMarker.COMPLETE_STEP_STATE,),
    ),
    FlexibleMeshReleaseFamily.MOE_INFERENCE: (
        FlexibleMeshWorkloadKind.MOE_INFER,
        FlexibleMeshReleaseOperation.MOE_INFERENCE_STEP,
        FlexibleMeshReleaseBaseline.MOE_DIRECT_XY,
        (FlexibleMeshCompletionMarker.DISPATCH_COMBINE,),
    ),
    FlexibleMeshReleaseFamily.MOE_TRAIN: (
        FlexibleMeshWorkloadKind.MOE_TRAIN,
        FlexibleMeshReleaseOperation.MOE_TRAIN_STEP,
        FlexibleMeshReleaseBaseline.MOE_DIRECT_XY,
        (
            FlexibleMeshCompletionMarker.FOUR_WAY_ROUTE,
            FlexibleMeshCompletionMarker.GRADIENT,
            FlexibleMeshCompletionMarker.OPTIMIZER,
            FlexibleMeshCompletionMarker.STATE,
        ),
    ),
    FlexibleMeshReleaseFamily.MESHSLICE_AG: (
        FlexibleMeshWorkloadKind.DENSE_INFER,
        FlexibleMeshReleaseOperation.AG_GEMM,
        FlexibleMeshReleaseBaseline.MESHSLICE_STANDARD,
        (FlexibleMeshCompletionMarker.MESHSLICE_OPERATION,),
    ),
    FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK: (
        FlexibleMeshWorkloadKind.DENSE_INFER,
        FlexibleMeshReleaseOperation.GEMM_RS,
        FlexibleMeshReleaseBaseline.UNFUSED_FALLBACK,
        (FlexibleMeshCompletionMarker.MESHSLICE_OPERATION,),
    ),
    FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK: (
        FlexibleMeshWorkloadKind.DENSE_INFER,
        FlexibleMeshReleaseOperation.GEMM_AR,
        FlexibleMeshReleaseBaseline.UNFUSED_FALLBACK,
        (FlexibleMeshCompletionMarker.MESHSLICE_OPERATION,),
    ),
}


def expected_completion_markers(
    family: FlexibleMeshReleaseFamily,
) -> tuple[FlexibleMeshCompletionMarker, ...]:
    if type(family) is not FlexibleMeshReleaseFamily:
        raise SchemaError("must be a release family", path="family")
    return _FAMILY_PROFILE[family][3]


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseTool:
    kind: FlexibleMeshReleaseToolKind
    binary_path: str
    version: str
    sha256: str
    allowlisted_sha256: tuple[str, ...]

    def validate(self, path: str = "flexible_mesh_release_tool") -> None:
        if type(self.kind) is not FlexibleMeshReleaseToolKind:
            raise SchemaError("must be a release tool kind", path=f"{path}.kind")
        validate_nonempty(self.binary_path, f"{path}.binary_path")
        validate_nonempty(self.version, f"{path}.version")
        _validate_sha256(self.sha256, f"{path}.sha256")
        if (
            type(self.allowlisted_sha256) is not tuple
            or not self.allowlisted_sha256
            or self.allowlisted_sha256
            != tuple(sorted(set(self.allowlisted_sha256)))
        ):
            raise SchemaError(
                "allowlist must be non-empty, unique, and sorted",
                path=f"{path}.allowlisted_sha256",
            )
        for index, digest in enumerate(self.allowlisted_sha256):
            _validate_sha256(digest, f"{path}.allowlisted_sha256[{index}]")
        if self.sha256 not in self.allowlisted_sha256:
            raise SchemaError("tool SHA is not allowlisted", path=f"{path}.sha256")


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseBinding:
    schema_version: str
    id: str
    runtime_profile_version: str
    environment_profile_version: str
    tools: tuple[FlexibleMeshReleaseTool, ...]
    hardware_config_sha256: str
    simulation_config_sha256: str
    mapping_config_sha256: str

    @classmethod
    def create(
        cls,
        *,
        runtime_profile_version: str,
        environment_profile_version: str,
        tools: tuple[FlexibleMeshReleaseTool, ...],
        hardware_config_sha256: str,
        simulation_config_sha256: str,
        mapping_config_sha256: str,
    ) -> "FlexibleMeshReleaseBinding":
        semantic = {
            "runtime_profile_version": runtime_profile_version,
            "environment_profile_version": environment_profile_version,
            "tools": tools,
            "hardware_config_sha256": hardware_config_sha256,
            "simulation_config_sha256": simulation_config_sha256,
            "mapping_config_sha256": mapping_config_sha256,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_release_binding",
                semantic,
                schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "flexible_mesh_release_binding") -> None:
        if self.schema_version != FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.runtime_profile_version, f"{path}.runtime_profile_version")
        validate_nonempty(
            self.environment_profile_version,
            f"{path}.environment_profile_version",
        )
        if type(self.tools) is not tuple or tuple(tool.kind for tool in self.tools) != tuple(
            FlexibleMeshReleaseToolKind
        ):
            raise SchemaError(
                "tools must contain finalizer, resolver, and npusim in canonical order",
                path=f"{path}.tools",
            )
        for index, tool in enumerate(self.tools):
            if type(tool) is not FlexibleMeshReleaseTool:
                raise SchemaError("must be a release tool", path=f"{path}.tools[{index}]")
            tool.validate(f"{path}.tools[{index}]")
        for name in (
            "hardware_config_sha256",
            "simulation_config_sha256",
            "mapping_config_sha256",
        ):
            _validate_sha256(getattr(self, name), f"{path}.{name}")
        expected_id = stable_artifact_id(
            "flexible_mesh_release_binding",
            self._semantic_key(),
            schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable binding id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseCase:
    schema_version: str
    id: str
    family: FlexibleMeshReleaseFamily
    mesh: RectMeshSpec
    workload_kind: FlexibleMeshWorkloadKind
    trace_model_digest: str
    selected_baseline: FlexibleMeshReleaseBaseline
    operation: FlexibleMeshReleaseOperation
    runtime_profile_version: str

    @classmethod
    def create(
        cls,
        *,
        family: FlexibleMeshReleaseFamily,
        mesh: RectMeshSpec,
        trace_model_digest: str,
        runtime_profile_version: str,
    ) -> "FlexibleMeshReleaseCase":
        if type(family) is not FlexibleMeshReleaseFamily:
            raise SchemaError("must be a release family", path="family")
        workload, operation, baseline, _ = _FAMILY_PROFILE[family]
        semantic_id_key = {
            "mesh_digest": mesh.digest,
            "workload_kind": workload,
            "trace_model_digest": trace_model_digest,
            "selected_baseline": baseline,
            "operation": operation,
            "runtime_profile_version": runtime_profile_version,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_release_case",
                semantic_id_key,
                schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
            ),
            family=family,
            mesh=mesh,
            workload_kind=workload,
            trace_model_digest=trace_model_digest,
            selected_baseline=baseline,
            operation=operation,
            runtime_profile_version=runtime_profile_version,
        )
        result.validate()
        return result

    def _id_key(self) -> dict[str, object]:
        return {
            "mesh_digest": self.mesh.digest,
            "workload_kind": self.workload_kind,
            "trace_model_digest": self.trace_model_digest,
            "selected_baseline": self.selected_baseline,
            "operation": self.operation,
            "runtime_profile_version": self.runtime_profile_version,
        }

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "flexible_mesh_release_case") -> None:
        if self.schema_version != FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.family) is not FlexibleMeshReleaseFamily:
            raise SchemaError("must be a release family", path=f"{path}.family")
        if type(self.mesh) is not RectMeshSpec:
            raise SchemaError("must be a RectMeshSpec", path=f"{path}.mesh")
        self.mesh.validate(f"{path}.mesh")
        workload, operation, baseline, _ = _FAMILY_PROFILE[self.family]
        if (
            self.workload_kind is not workload
            or self.operation is not operation
            or self.selected_baseline is not baseline
        ):
            raise SchemaError("family profile drifted", path=path)
        _validate_sha256(self.trace_model_digest, f"{path}.trace_model_digest")
        validate_nonempty(self.runtime_profile_version, f"{path}.runtime_profile_version")
        expected_id = stable_artifact_id(
            "flexible_mesh_release_case",
            self._id_key(),
            schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable release case id", path=f"{path}.id")


def generate_flexible_mesh_release_cases(
    *,
    trace_model_digests: tuple[tuple[FlexibleMeshReleaseFamily, str], ...],
    runtime_profile_version: str,
) -> tuple[FlexibleMeshReleaseCase, ...]:
    if (
        type(trace_model_digests) is not tuple
        or tuple(family for family, _ in trace_model_digests)
        != tuple(FlexibleMeshReleaseFamily)
    ):
        raise SchemaError(
            "trace/model digests must cover all six families in canonical order",
            path="trace_model_digests",
        )
    validate_nonempty(runtime_profile_version, "runtime_profile_version")
    for index, (_, digest) in enumerate(trace_model_digests):
        _validate_sha256(digest, f"trace_model_digests[{index}][1]")
    digest_by_family = dict(trace_model_digests)
    result = tuple(
        FlexibleMeshReleaseCase.create(
            family=family,
            mesh=RectMeshSpec(rows=rows, columns=columns),
            trace_model_digest=digest_by_family[family],
            runtime_profile_version=runtime_profile_version,
        )
        for family in FlexibleMeshReleaseFamily
        for rows in range(1, 11)
        for columns in range(1, 11)
    )
    validate_flexible_mesh_release_cases(result, runtime_profile_version)
    return result


def validate_flexible_mesh_release_cases(
    cases: tuple[FlexibleMeshReleaseCase, ...],
    runtime_profile_version: str,
    path: str = "release_cases",
) -> None:
    if type(cases) is not tuple or len(cases) != FLEXIBLE_MESH_RELEASE_CASE_COUNT:
        raise SchemaError("must contain exactly 600 release cases", path=path)
    ids: set[str] = set()
    observed: set[tuple[FlexibleMeshReleaseFamily, int, int]] = set()
    family_digests: dict[FlexibleMeshReleaseFamily, str] = {}
    expected_order = tuple(
        (family, rows, columns)
        for family in FlexibleMeshReleaseFamily
        for rows in range(1, 11)
        for columns in range(1, 11)
    )
    actual_order: list[tuple[FlexibleMeshReleaseFamily, int, int]] = []
    for index, case in enumerate(cases):
        if type(case) is not FlexibleMeshReleaseCase:
            raise SchemaError("must be a release case", path=f"{path}[{index}]")
        case.validate(f"{path}[{index}]")
        if case.runtime_profile_version != runtime_profile_version:
            raise SchemaError("runtime profile drifted", path=f"{path}[{index}]")
        if case.id in ids:
            raise SchemaError("duplicate release case id", path=f"{path}[{index}].id")
        ids.add(case.id)
        key = (case.family, case.mesh.rows, case.mesh.columns)
        if key in observed:
            raise SchemaError("duplicate family/Mesh case", path=f"{path}[{index}]")
        observed.add(key)
        actual_order.append(key)
        previous = family_digests.setdefault(case.family, case.trace_model_digest)
        if previous != case.trace_model_digest:
            raise SchemaError("family trace/model digest drifted", path=f"{path}[{index}]")
    if tuple(actual_order) != expected_order:
        raise SchemaError("cases do not exactly cover the canonical 100-shape matrix", path=path)


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseCapacityEvidence:
    rank_count: int
    peak_sessions_per_core_per_wave: int
    symbolic_record_count: int
    exact_record_count: int
    linked_manifest_file_bytes: int
    artifact_file_bytes: int
    max_runtime_core_id: int
    transport_tag_count: int

    def validate(
        self,
        case: FlexibleMeshReleaseCase | str | None = None,
        path: str = "flexible_mesh_release_capacity",
    ) -> None:
        if type(case) is str:
            path = case
            case = None
        elif case is not None and type(case) is not FlexibleMeshReleaseCase:
            raise SchemaError("must be a release case", path=path)
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.rank_count > 100 or (
            case is not None and self.rank_count != case.mesh.rank_count
        ):
            raise SchemaError("rank count drifted or exceeded 100", path=f"{path}.rank_count")
        if self.peak_sessions_per_core_per_wave > FLEXIBLE_MESH_RELEASE_MAX_SESSIONS_PER_CORE_PER_WAVE:
            raise SchemaError("session limit exceeded", path=f"{path}.peak_sessions_per_core_per_wave")
        if (
            self.symbolic_record_count == 0
            or self.symbolic_record_count > FLEXIBLE_MESH_RELEASE_MAX_RECORDS
            or self.exact_record_count == 0
            or self.exact_record_count > self.symbolic_record_count
        ):
            raise SchemaError("record capacity is not closed", path=path)
        if (
            self.linked_manifest_file_bytes == 0
            or self.linked_manifest_file_bytes
            > FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES
        ):
            raise SchemaError(
                "linked manifest byte limit exceeded: "
                f"observed={self.linked_manifest_file_bytes} "
                f"limit={FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES}",
                path=f"{path}.linked_manifest_file_bytes",
            )
        if self.artifact_file_bytes == 0 or self.artifact_file_bytes > FLEXIBLE_MESH_RELEASE_MAX_ARTIFACT_BYTES:
            raise SchemaError("artifact byte limit exceeded", path=f"{path}.artifact_file_bytes")
        if self.max_runtime_core_id > FLEXIBLE_MESH_RELEASE_MAX_RUNTIME_CORE_ID:
            raise SchemaError("runtime core id limit exceeded", path=f"{path}.max_runtime_core_id")
        if self.transport_tag_count > FLEXIBLE_MESH_RELEASE_MAX_TRANSPORT_TAGS:
            raise SchemaError("transport tag limit exceeded", path=f"{path}.transport_tag_count")


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseResidual:
    active_endpoints: int = 0
    active_sessions: int = 0
    outstanding_tags: int = 0
    incomplete_barriers: int = 0
    pending_state_writes: int = 0
    proto_wait_count: int = 0
    lsu_residual: int = 0
    dte_residual: int = 0
    router_residual: int = 0
    credit_residual: int = 0

    def validate(self, path: str = "flexible_mesh_release_residual") -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            validate_uint64(value, f"{path}.{name}")
            if value != 0:
                raise SchemaError("runtime residual must be zero", path=f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class FlexibleMeshIndependentRun:
    materialization_id: str
    finalizer_run_id: str
    resolver_run_id: str
    npusim_run_id: str

    def validate(self, path: str = "flexible_mesh_independent_run") -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        for name, value in zip(self.__dataclass_fields__, values):
            validate_nonempty(value, f"{path}.{name}")
        if len(set(values)) != len(values):
            raise SchemaError("stage run identities must be distinct", path=path)

    @property
    def identities(self) -> tuple[str, ...]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseExecutionEvidence:
    schema_version: str
    id: str
    execution_index: int
    case_id: str
    case_digest: str
    binding_id: str
    binding_digest: str
    run: FlexibleMeshIndependentRun
    spec_digest: str
    plan_digest: str
    manifest_digest: str
    hardware_config_sha256: str
    simulation_config_sha256: str
    mapping_config_sha256: str
    artifact_sha256: str
    artifact_file_bytes: int
    program_io_artifact_sha256: str
    program_io_digest: str
    resolver_digest: str
    makespan_cycles: int
    marker_digest: str
    capacity: FlexibleMeshReleaseCapacityEvidence
    residual: FlexibleMeshReleaseResidual
    rank_coverage: tuple[int, ...]
    core_coverage: tuple[int, ...]
    completion_markers: tuple[FlexibleMeshCompletionMarker, ...]
    stage_exit_codes: tuple[tuple[FlexibleMeshReleaseRuntimeStage, int], ...]
    program_io_phases: tuple[FlexibleMeshProgramIOPhase, ...]
    timing_execution: bool
    functional_execution: bool

    @classmethod
    def create(
        cls,
        *,
        case: FlexibleMeshReleaseCase,
        binding: FlexibleMeshReleaseBinding,
        execution_index: int,
        run: FlexibleMeshIndependentRun,
        spec_digest: str,
        plan_digest: str,
        manifest_digest: str,
        hardware_config_sha256: str,
        simulation_config_sha256: str,
        mapping_config_sha256: str,
        artifact_sha256: str,
        artifact_file_bytes: int,
        program_io_artifact_sha256: str,
        program_io_digest: str,
        resolver_digest: str,
        makespan_cycles: int,
        marker_digest: str,
        capacity: FlexibleMeshReleaseCapacityEvidence,
        residual: FlexibleMeshReleaseResidual,
        rank_coverage: tuple[int, ...],
        core_coverage: tuple[int, ...],
        completion_markers: tuple[FlexibleMeshCompletionMarker, ...],
        stage_exit_codes: tuple[tuple[FlexibleMeshReleaseRuntimeStage, int], ...],
        program_io_phases: tuple[FlexibleMeshProgramIOPhase, ...],
        timing_execution: bool = True,
        functional_execution: bool = False,
    ) -> "FlexibleMeshReleaseExecutionEvidence":
        semantic = {
            "execution_index": execution_index,
            "case_id": case.id,
            "case_digest": case.digest,
            "binding_id": binding.id,
            "binding_digest": binding.digest,
            "run": run,
            "spec_digest": spec_digest,
            "plan_digest": plan_digest,
            "manifest_digest": manifest_digest,
            "hardware_config_sha256": hardware_config_sha256,
            "simulation_config_sha256": simulation_config_sha256,
            "mapping_config_sha256": mapping_config_sha256,
            "artifact_sha256": artifact_sha256,
            "artifact_file_bytes": artifact_file_bytes,
            "program_io_artifact_sha256": program_io_artifact_sha256,
            "program_io_digest": program_io_digest,
            "resolver_digest": resolver_digest,
            "makespan_cycles": makespan_cycles,
            "marker_digest": marker_digest,
            "capacity": capacity,
            "residual": residual,
            "rank_coverage": rank_coverage,
            "core_coverage": core_coverage,
            "completion_markers": completion_markers,
            "stage_exit_codes": stage_exit_codes,
            "program_io_phases": program_io_phases,
            "timing_execution": timing_execution,
            "functional_execution": functional_execution,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_release_execution",
                semantic,
                schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate_against(case, binding)
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate_against(
        self,
        case: FlexibleMeshReleaseCase,
        binding: FlexibleMeshReleaseBinding,
        path: str = "flexible_mesh_release_execution",
    ) -> None:
        binding.validate(f"{path}.binding")
        self._validate_payload(case, path)
        if self.binding_id != binding.id or self.binding_digest != binding.digest:
            raise SchemaError("tool/config binding drifted", path=path)
        if (
            self.simulation_config_sha256 != binding.simulation_config_sha256
            or self.mapping_config_sha256 != binding.mapping_config_sha256
        ):
            raise SchemaError(
                "execution static config drifted from release binding", path=path
            )

    def _validate_payload(
        self,
        case: FlexibleMeshReleaseCase,
        path: str = "flexible_mesh_release_execution",
    ) -> None:
        if self.schema_version != FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        case.validate(f"{path}.case")
        if type(self.execution_index) is not int or self.execution_index not in (0, 1):
            raise SchemaError("execution index must be zero or one", path=f"{path}.execution_index")
        if self.case_id != case.id or self.case_digest != case.digest:
            raise SchemaError("case provenance drifted", path=path)
        validate_nonempty(self.binding_id, f"{path}.binding_id")
        if type(self.run) is not FlexibleMeshIndependentRun:
            raise SchemaError("must be an independent run", path=f"{path}.run")
        self.run.validate(f"{path}.run")
        for name in (
            "case_digest",
            "binding_digest",
            "spec_digest",
            "plan_digest",
            "manifest_digest",
            "hardware_config_sha256",
            "simulation_config_sha256",
            "mapping_config_sha256",
            "artifact_sha256",
            "program_io_artifact_sha256",
            "program_io_digest",
            "resolver_digest",
            "marker_digest",
        ):
            _validate_sha256(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.artifact_file_bytes, f"{path}.artifact_file_bytes")
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if self.program_io_artifact_sha256 != self.artifact_sha256:
            raise SchemaError(
                "ProgramIO is not bound to the actual artifact SHA",
                path=f"{path}.program_io_artifact_sha256",
            )
        if self.makespan_cycles == 0:
            raise SchemaError("makespan must be positive", path=f"{path}.makespan_cycles")
        if type(self.capacity) is not FlexibleMeshReleaseCapacityEvidence:
            raise SchemaError("must be capacity evidence", path=f"{path}.capacity")
        self.capacity.validate(case, f"{path}.capacity")
        if self.artifact_file_bytes != self.capacity.artifact_file_bytes:
            raise SchemaError("artifact bytes drifted from exact audit", path=path)
        if type(self.residual) is not FlexibleMeshReleaseResidual:
            raise SchemaError("must be a runtime residual", path=f"{path}.residual")
        self.residual.validate(f"{path}.residual")
        expected_coverage = tuple(range(case.mesh.rank_count))
        if self.rank_coverage != expected_coverage or self.core_coverage != expected_coverage:
            raise SchemaError("rank/core coverage is not exact", path=path)
        if self.completion_markers != expected_completion_markers(case.family):
            raise SchemaError("workload completion markers are not exact", path=f"{path}.completion_markers")
        expected_exits = tuple((stage, 0) for stage in FlexibleMeshReleaseRuntimeStage)
        if self.stage_exit_codes != expected_exits:
            raise SchemaError("runtime stage receipts are not exact zero exits", path=f"{path}.stage_exit_codes")
        _validate_exact_enum_tuple(self.program_io_phases, FlexibleMeshProgramIOPhase, f"{path}.program_io_phases")
        if self.timing_execution is not True or self.functional_execution is not False:
            raise SchemaError("release evidence is timing-only", path=path)
        expected_id = stable_artifact_id(
            "flexible_mesh_release_execution",
            self._semantic_key(),
            schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable execution evidence id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseCaseEvidence:
    schema_version: str
    id: str
    case: FlexibleMeshReleaseCase
    binding_id: str
    binding_digest: str
    executions: tuple[FlexibleMeshReleaseExecutionEvidence, ...]

    @classmethod
    def create(
        cls,
        *,
        case: FlexibleMeshReleaseCase,
        binding: FlexibleMeshReleaseBinding,
        executions: tuple[FlexibleMeshReleaseExecutionEvidence, ...],
    ) -> "FlexibleMeshReleaseCaseEvidence":
        semantic = {
            "case": case,
            "binding_id": binding.id,
            "binding_digest": binding.digest,
            "executions": executions,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_release_case_evidence",
                semantic,
                schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate_against(binding)
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    @property
    def runtime_verified(self) -> bool:
        return len(self.executions) == 2 and self._safe_validate(require_repeat=False)

    @property
    def repeatability_verified(self) -> bool:
        if not self.runtime_verified:
            return False
        first, second = self.executions
        repeat_fields = (
            "spec_digest",
            "plan_digest",
            "manifest_digest",
            "hardware_config_sha256",
            "simulation_config_sha256",
            "mapping_config_sha256",
            "artifact_sha256",
            "artifact_file_bytes",
            "program_io_artifact_sha256",
            "program_io_digest",
            "resolver_digest",
            "makespan_cycles",
            "marker_digest",
            "capacity",
            "residual",
            "rank_coverage",
            "core_coverage",
            "completion_markers",
            "stage_exit_codes",
            "program_io_phases",
        )
        return all(getattr(first, name) == getattr(second, name) for name in repeat_fields)

    def _safe_validate(self, *, require_repeat: bool) -> bool:
        try:
            self._validate_structure(None, require_repeat=require_repeat)
            return True
        except (SchemaError, AttributeError, TypeError, ValueError):
            return False

    def _validate_structure(
        self,
        binding: FlexibleMeshReleaseBinding | None,
        *,
        require_repeat: bool,
        path: str = "flexible_mesh_release_case_evidence",
    ) -> None:
        if self.schema_version != FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.case.validate(f"{path}.case")
        _validate_sha256(self.binding_digest, f"{path}.binding_digest")
        validate_nonempty(self.binding_id, f"{path}.binding_id")
        if binding is not None and (self.binding_id != binding.id or self.binding_digest != binding.digest):
            raise SchemaError("tool/config binding drifted", path=path)
        if type(self.executions) is not tuple or len(self.executions) != 2:
            raise SchemaError("exactly two executions are required", path=f"{path}.executions")
        if tuple(execution.execution_index for execution in self.executions) != (0, 1):
            raise SchemaError("executions must be indexed zero then one", path=f"{path}.executions")
        all_run_ids: list[str] = []
        for index, execution in enumerate(self.executions):
            if type(execution) is not FlexibleMeshReleaseExecutionEvidence:
                raise SchemaError("must be execution evidence", path=f"{path}.executions[{index}]")
            if binding is not None:
                execution.validate_against(self.case, binding, f"{path}.executions[{index}]")
            else:
                execution._validate_payload(self.case, f"{path}.executions[{index}]")
                if (
                    execution.binding_id != self.binding_id
                    or execution.binding_digest != self.binding_digest
                ):
                    raise SchemaError(
                        "execution binding drifted from case evidence",
                        path=f"{path}.executions[{index}]",
                    )
            all_run_ids.extend(execution.run.identities)
        if len(set(all_run_ids)) != 8:
            raise SchemaError("two executions are not independent", path=f"{path}.executions")
        expected_id = stable_artifact_id(
            "flexible_mesh_release_case_evidence",
            self._semantic_key(),
            schema_version=FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable case evidence id", path=f"{path}.id")
        if require_repeat and not self.repeatability_verified:
            raise SchemaError("repeatability evidence drifted", path=f"{path}.executions")

    def validate_against(
        self,
        binding: FlexibleMeshReleaseBinding,
        path: str = "flexible_mesh_release_case_evidence",
    ) -> None:
        binding.validate(f"{path}.binding")
        self._validate_structure(binding, require_repeat=False, path=path)
        if not self.repeatability_verified:
            raise SchemaError("repeatability evidence drifted", path=f"{path}.executions")


__all__ = [
    "FLEXIBLE_MESH_RELEASE_CAPACITY_POLICY_VERSION",
    "FLEXIBLE_MESH_RELEASE_CASE_COUNT",
    "FLEXIBLE_MESH_RELEASE_EXECUTION_COUNT",
    "FLEXIBLE_MESH_RELEASE_MAX_ARTIFACT_BYTES",
    "FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES",
    "FLEXIBLE_MESH_RELEASE_MAX_RECORDS",
    "FLEXIBLE_MESH_RELEASE_MAX_RUNTIME_CORE_ID",
    "FLEXIBLE_MESH_RELEASE_MAX_SESSIONS_PER_CORE_PER_WAVE",
    "FLEXIBLE_MESH_RELEASE_MAX_TRANSPORT_TAGS",
    "FLEXIBLE_MESH_RELEASE_SCHEMA_VERSION",
    "FLEXIBLE_MESH_RELEASE_SHAPE_COUNT",
    "FlexibleMeshCompletionMarker",
    "FlexibleMeshIndependentRun",
    "FlexibleMeshProgramIOPhase",
    "FlexibleMeshReleaseBaseline",
    "FlexibleMeshReleaseBinding",
    "FlexibleMeshReleaseCapacityEvidence",
    "FlexibleMeshReleaseCase",
    "FlexibleMeshReleaseCaseEvidence",
    "FlexibleMeshReleaseExecutionEvidence",
    "FlexibleMeshReleaseFamily",
    "FlexibleMeshReleaseOperation",
    "FlexibleMeshReleaseResidual",
    "FlexibleMeshReleaseRuntimeStage",
    "FlexibleMeshReleaseTool",
    "FlexibleMeshReleaseToolKind",
    "expected_completion_markers",
    "generate_flexible_mesh_release_cases",
    "validate_flexible_mesh_release_cases",
]
