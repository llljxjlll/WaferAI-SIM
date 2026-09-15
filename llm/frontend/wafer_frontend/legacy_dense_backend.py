"""Strict bridge from unified workload preflight to the legacy Dense compiler.

The current ``compile_rect_mesh`` fallback is executable, but its capability
report explicitly says that the standard whole-Dense chain is incomplete.
This module therefore exposes that path only as a static-profile motif canary.
It must never be used as full-model readiness evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .compiler import compile_rect_mesh
from .errors import SchemaError, UnsupportedFeatureError
from .schema.common import stable_artifact_id, validate_nonempty
from .schema.experiment import (
    ExperimentSpec,
    InferSource,
    PlacementStrategy,
    WorkloadMode,
)
from .schema.ir1 import PhysicalFabric
from .schema.memory_plan import MemoryPlanExecution
from .schema.persistent_state import HbmAddressSpace
from .schema.rect_mesh import RectMeshSpec
from .schema.rect_mesh_compile import (
    RectMeshCompileChain,
    RectMeshCompileMode,
)
from .schema.serde import canonical_digest, canonical_json
from .schema.workload_materialization import (
    WorkloadMaterializationManifest,
    WorkloadMaterializationStatus,
)
from .schema.workload_run import (
    WorkloadExecutionStrategy,
    WorkloadFamily,
    WorkloadMemoryMode,
    WorkloadModelArchitecture,
    WorkloadRunRequest,
)
from .workload_runner import (
    WorkloadRunnerStage,
    WorkloadStageArtifact,
    WorkloadStageContext,
)


LEGACY_DENSE_BACKEND_COMPATIBILITY_SCHEMA_VERSION = (
    "wafer_frontend.legacy_dense_backend_compatibility/v1alpha1"
)


class LegacyDenseBackendScope(str, Enum):
    UNSUPPORTED = "unsupported"
    LEGACY_STATIC_PROFILE_MOTIF = "legacy_static_profile/motif"


@dataclass(frozen=True, slots=True)
class LegacyDenseBackendCompatibility:
    """Fail-closed compatibility decision for one exact unified case."""

    schema_version: str
    id: str
    request_digest: str
    manifest_digest: str
    legacy_spec_digest: str
    fabric_digest: str
    scope: LegacyDenseBackendScope
    reasons: tuple[str, ...]
    legacy_profile_id: str | None
    selected_chain: RectMeshCompileChain | None
    full_model_ready: bool

    @classmethod
    def create(
        cls,
        *,
        request: WorkloadRunRequest,
        manifest: WorkloadMaterializationManifest,
        legacy_spec: ExperimentSpec,
        fabric: PhysicalFabric,
        reasons: tuple[str, ...],
        legacy_profile_id: str | None,
    ) -> "LegacyDenseBackendCompatibility":
        reasons = tuple(sorted(set(reasons)))
        compatible = not reasons
        key = {
            "request_digest": request.digest,
            "manifest_digest": manifest.digest,
            "legacy_spec_digest": canonical_digest(legacy_spec),
            "fabric_digest": canonical_digest(fabric),
            "scope": (
                LegacyDenseBackendScope.LEGACY_STATIC_PROFILE_MOTIF
                if compatible
                else LegacyDenseBackendScope.UNSUPPORTED
            ),
            "reasons": reasons,
            "legacy_profile_id": legacy_profile_id if compatible else None,
            "selected_chain": (
                RectMeshCompileChain.NAIVE_FIXED_V1 if compatible else None
            ),
            # compile_rect_mesh reports dense_workload_complete=False for this
            # fallback.  Keep the negative assertion in the persisted contract.
            "full_model_ready": False,
        }
        result = cls(
            schema_version=LEGACY_DENSE_BACKEND_COMPATIBILITY_SCHEMA_VERSION,
            id=stable_artifact_id(
                "legacy_dense_backend_compatibility",
                key,
                schema_version=LEGACY_DENSE_BACKEND_COMPATIBILITY_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    @property
    def compatible(self) -> bool:
        return self.scope is LegacyDenseBackendScope.LEGACY_STATIC_PROFILE_MOTIF

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "legacy_dense_backend_compatibility") -> None:
        if self.schema_version != LEGACY_DENSE_BACKEND_COMPATIBILITY_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in (
            "request_digest",
            "manifest_digest",
            "legacy_spec_digest",
            "fabric_digest",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SchemaError("must be a lowercase SHA-256 digest", path=f"{path}.{name}")
        if type(self.scope) is not LegacyDenseBackendScope:
            raise SchemaError("must be a LegacyDenseBackendScope", path=f"{path}.scope")
        if self.reasons != tuple(sorted(set(self.reasons))):
            raise SchemaError("must be sorted and unique", path=f"{path}.reasons")
        for index, reason in enumerate(self.reasons):
            validate_nonempty(reason, f"{path}.reasons[{index}]")
        if self.compatible:
            if self.reasons:
                raise SchemaError("compatible scope cannot carry rejection reasons", path=path)
            if not self.legacy_profile_id:
                raise SchemaError("compatible scope requires a profile id", path=path)
            if self.selected_chain is not RectMeshCompileChain.NAIVE_FIXED_V1:
                raise SchemaError("compatible scope requires naive_fixed/v1", path=path)
        elif not self.reasons:
            raise SchemaError("unsupported scope requires reasons", path=f"{path}.reasons")
        if type(self.full_model_ready) is not bool or self.full_model_ready:
            raise SchemaError(
                "legacy static-profile bridge can never assert full-model readiness",
                path=f"{path}.full_model_ready",
            )
        expected = stable_artifact_id(
            "legacy_dense_backend_compatibility",
            self._key(),
            schema_version=LEGACY_DENSE_BACKEND_COMPATIBILITY_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable compatibility id", path=f"{path}.id")


def _compatibility_reasons(
    request: WorkloadRunRequest,
    manifest: WorkloadMaterializationManifest,
    legacy_spec: ExperimentSpec,
    fabric: PhysicalFabric,
) -> tuple[tuple[str, ...], str | None]:
    reasons: list[str] = []
    if manifest.request != request:
        reasons.append("manifest.request_mismatch")
    if manifest.status is not WorkloadMaterializationStatus.PARTIAL:
        reasons.append("manifest.unsupported")
    if request.family is not WorkloadFamily.DENSE_INFERENCE:
        reasons.append("request.family_dense_inference_required")
    if request.model.architecture is not WorkloadModelArchitecture.LLAMA_DENSE:
        reasons.append("request.model_dense_required")
    if request.memory.mode is not WorkloadMemoryMode.RESIDENT_HBM:
        reasons.append("request.memory_resident_hbm_required")
    if manifest.memory_plan.execution is not MemoryPlanExecution.NO_EXTERNAL_TRANSPORT_REQUIRED:
        reasons.append("manifest.external_memory_not_executable")
    if not request.execution.timing or request.execution.functional:
        reasons.append("request.execution_timing_only_required")
    if request.execution.strategy is not WorkloadExecutionStrategy.BASELINE:
        reasons.append("request.execution_baseline_required")
    if (request.parallel.dp, request.parallel.ep, request.parallel.pp) != (1, 1, 1):
        reasons.append("request.parallel_dp_ep_pp_must_equal_one")
    expected_dies = tuple(range(request.mesh.rank_count))
    active_dies = request.parallel.active_die_ids or expected_dies
    if request.parallel.tp > request.mesh.rank_count:
        reasons.append("request.parallel_tp_exceeds_mesh")
    if request.parallel.tp < request.mesh.rank_count and not request.parallel.active_die_ids:
        reasons.append("request.explicit_active_die_ids_required")
    if len(active_dies) != request.parallel.tp:
        reasons.append("request.active_die_ids_tp_mismatch")
    if manifest.placement.active_die_ids != active_dies:
        reasons.append("manifest.placement_active_die_ids_mismatch")
    if fabric.die_grid != (request.mesh.columns, request.mesh.rows):
        reasons.append("fabric.mesh_shape_mismatch")
    if len(fabric.dies) != request.mesh.rank_count:
        reasons.append("fabric.complete_rectangular_mesh_required")

    profile_id: str | None = None
    if legacy_spec.workload.mode is not WorkloadMode.INFER:
        reasons.append("legacy.workload_inference_required")
    else:
        infer = legacy_spec.workload.infer
        assert infer is not None
        if infer.source is not InferSource.STATIC_PROFILE or infer.profile is None:
            reasons.append("legacy.single_static_profile_required")
        else:
            profile = infer.profile
            profile_id = profile.stable_id()
            if profile.prefill_tokens > 0 and profile.decode_tokens > 0:
                reasons.append("legacy.mixed_prefill_decode_unsupported")
            steps = request.steps.inference
            if steps is None or (
                profile.prefill_tokens,
                profile.decode_tokens,
                profile.num_seqs,
            ) != (
                steps.prefill_tokens if steps is not None else -1,
                steps.decode_steps if steps is not None else -1,
                steps.request_count if steps is not None else -1,
            ):
                reasons.append("legacy.profile_shape_mismatch")

    model = legacy_spec.model
    unified_model = request.model
    if (
        model.V,
        model.H,
        model.I,
        model.L,
        model.NH,
        model.KVH,
        model.DH,
        model.max_position_embeddings,
        model.dtype,
    ) != (
        unified_model.vocabulary_size,
        unified_model.hidden_size,
        unified_model.intermediate_size,
        unified_model.num_layers,
        unified_model.num_attention_heads,
        unified_model.num_kv_heads,
        unified_model.head_dim,
        unified_model.max_sequence_length,
        unified_model.dtype,
    ):
        reasons.append("legacy.model_mismatch")
    instances = legacy_spec.parallel.instances
    if len(instances) != 1:
        reasons.append("legacy.single_instance_required")
    elif (
        instances[0].tp,
        instances[0].dp,
        instances[0].ep,
        instances[0].pp,
    ) != (request.parallel.tp, 1, 1, 1):
        reasons.append("legacy.parallel_mismatch")
    if legacy_spec.placement.strategy is PlacementStrategy.COMPACT:
        if active_dies != expected_dies:
            reasons.append("legacy.explicit_active_group_required")
    elif len(instances) == 1:
        expected_key = (instances[0].id, f"{instances[0].id}.mesh.tp")
        groups = legacy_spec.placement.groups
        if (len(groups) != 1
                or (groups[0].instance_id, groups[0].mesh_ref) != expected_key
                or groups[0].die_ids != active_dies):
            reasons.append("legacy.active_group_mismatch")
    return tuple(reasons), profile_id


def assess_legacy_dense_backend(
    request: WorkloadRunRequest,
    manifest: WorkloadMaterializationManifest,
    legacy_spec: ExperimentSpec,
    fabric: PhysicalFabric,
) -> LegacyDenseBackendCompatibility:
    """Assess the exact legacy subset without implying runtime readiness."""

    if type(request) is not WorkloadRunRequest:
        raise SchemaError("must be a WorkloadRunRequest", path="request")
    if type(manifest) is not WorkloadMaterializationManifest:
        raise SchemaError("must be a WorkloadMaterializationManifest", path="manifest")
    if type(legacy_spec) is not ExperimentSpec:
        raise SchemaError("must be an ExperimentSpec", path="legacy_spec")
    if type(fabric) is not PhysicalFabric:
        raise SchemaError("must be a PhysicalFabric", path="fabric")
    request.validate()
    manifest.validate()
    legacy_spec.validate()
    fabric.validate("fabric")
    reasons, profile_id = _compatibility_reasons(
        request, manifest, legacy_spec, fabric
    )
    return LegacyDenseBackendCompatibility.create(
        request=request,
        manifest=manifest,
        legacy_spec=legacy_spec,
        fabric=fabric,
        reasons=reasons,
        legacy_profile_id=profile_id,
    )


def require_legacy_dense_backend(
    request: WorkloadRunRequest,
    manifest: WorkloadMaterializationManifest,
    legacy_spec: ExperimentSpec,
    fabric: PhysicalFabric,
) -> LegacyDenseBackendCompatibility:
    decision = assess_legacy_dense_backend(request, manifest, legacy_spec, fabric)
    if not decision.compatible:
        raise UnsupportedFeatureError(
            "request is outside legacy Dense static-profile subset: "
            + ", ".join(decision.reasons),
            path="legacy_dense_backend",
            code="legacy_dense_backend_unsupported",
        )
    return decision


@dataclass(frozen=True, slots=True)
class LegacyDenseCompileAdapter:
    """One real production compiler stage exposed through the runner artifact ABI."""

    request: WorkloadRunRequest
    manifest: WorkloadMaterializationManifest
    legacy_spec: ExperimentSpec
    fabric: PhysicalFabric
    hbm_address_spaces: tuple[HbmAddressSpace, ...]

    def __call__(self, context: WorkloadStageContext) -> WorkloadStageArtifact:
        if type(context) is not WorkloadStageContext:
            raise SchemaError("must be a WorkloadStageContext", path="context")
        if context.manifest != self.manifest:
            raise SchemaError("context manifest changed", path="context.manifest")
        decision = require_legacy_dense_backend(
            self.request, self.manifest, self.legacy_spec, self.fabric
        )
        compilation = compile_rect_mesh(
            self.legacy_spec,
            self.fabric,
            rect_mesh=RectMeshSpec(self.request.mesh.rows, self.request.mesh.columns),
            hbm_address_spaces=self.hbm_address_spaces,
            mode=RectMeshCompileMode.AUTO,
            producer_pass="legacy_dense_static_profile_adapter",
        )
        compilation.validate()
        report = compilation.capability_report
        if (
            report.selected_chain is not RectMeshCompileChain.NAIVE_FIXED_V1
            or not report.executable_baseline
            or report.standard_chain_complete
            or report.dense_workload_complete
        ):
            raise SchemaError(
                "compile_rect_mesh capability drifted from legacy motif scope",
                path="legacy_dense_backend.compile.capability_report",
            )
        sources = compilation.compilation.linked.entries
        if len(sources) != 1 or sources[0].profile_id != decision.legacy_profile_id:
            raise SchemaError(
                "compiler did not preserve the exact static profile",
                path="legacy_dense_backend.compile.profile",
            )

        relative_root = Path("adapter")
        target = context.work_dir / relative_root
        target.mkdir(parents=True, exist_ok=True)
        outputs = {
            "compatibility.json": decision,
            "rect_mesh_compilation.json": compilation,
            "linked_manifest.json": sources[0].manifest,
            "capability_report.json": report,
        }
        for name, value in outputs.items():
            (target / name).write_text(canonical_json(value) + "\n", encoding="utf-8")
        artifact = WorkloadStageArtifact.create(
            stage=WorkloadRunnerStage.ADAPTER,
            input_digest=context.input_digest,
            work_dir=context.work_dir,
            relative_paths=tuple(
                (relative_root / name).as_posix() for name in sorted(outputs)
            ),
        )
        artifact.verify_files(context.work_dir)
        return artifact


def run_legacy_dense_compiler_canary(
    legacy_spec: ExperimentSpec,
    fabric: PhysicalFabric,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
    *,
    work_dir: Path,
    input_digest: str,
) -> WorkloadStageArtifact:
    """Compile one legacy static profile and persist motif-scoped real artifacts.

    This canary intentionally has no unified manifest argument: current P3
    manifests require a mixed Prefill+Decode graph, which the legacy IR0 builder
    rejects.  The output proves only the older single-phase compiler path.
    """

    legacy_spec.validate()
    fabric.validate("fabric")
    if type(hbm_address_spaces) is not tuple or not hbm_address_spaces:
        raise SchemaError("must be a non-empty tuple", path="hbm_address_spaces")
    for index, address_space in enumerate(hbm_address_spaces):
        if type(address_space) is not HbmAddressSpace:
            raise SchemaError(
                "must be an HbmAddressSpace",
                path=f"hbm_address_spaces[{index}]",
            )
        address_space.validate(f"hbm_address_spaces[{index}]")
    if legacy_spec.workload.mode is not WorkloadMode.INFER:
        raise UnsupportedFeatureError(
            "legacy compiler canary requires inference",
            path="legacy_spec.workload.mode",
        )
    infer = legacy_spec.workload.infer
    assert infer is not None
    if infer.source is not InferSource.STATIC_PROFILE or infer.profile is None:
        raise UnsupportedFeatureError(
            "legacy compiler canary requires one static profile",
            path="legacy_spec.workload.infer",
        )
    if infer.profile.prefill_tokens > 0 and infer.profile.decode_tokens > 0:
        raise UnsupportedFeatureError(
            "legacy compiler cannot lower a mixed Prefill+Decode static profile",
            path="legacy_spec.workload.infer.profile",
        )
    width, height = fabric.die_grid
    compilation = compile_rect_mesh(
        legacy_spec,
        fabric,
        rect_mesh=RectMeshSpec(height, width),
        hbm_address_spaces=hbm_address_spaces,
        mode=RectMeshCompileMode.AUTO,
        producer_pass="legacy_dense_static_profile_compiler_canary",
    )
    compilation.validate()
    report = compilation.capability_report
    if (
        report.selected_chain is not RectMeshCompileChain.NAIVE_FIXED_V1
        or not report.executable_baseline
        or report.standard_chain_complete
        or report.dense_workload_complete
    ):
        raise SchemaError(
            "compiler canary escaped legacy motif scope",
            path="legacy_dense_backend.compile.capability_report",
        )
    sources = compilation.compilation.linked.entries
    if len(sources) != 1:
        raise SchemaError(
            "compiler canary requires exactly one linked profile",
            path="legacy_dense_backend.compile.profile",
        )
    target = work_dir / "adapter"
    target.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, object] = {
        "scope.json": {
            "scope": LegacyDenseBackendScope.LEGACY_STATIC_PROFILE_MOTIF,
            "full_model_ready": False,
            "unified_manifest_covered": False,
            "reason": "legacy single-phase static profile compiler canary",
        },
        "rect_mesh_compilation.json": compilation,
        "linked_manifest.json": sources[0].manifest,
        "capability_report.json": report,
    }
    for name, value in outputs.items():
        (target / name).write_text(canonical_json(value) + "\n", encoding="utf-8")
    artifact = WorkloadStageArtifact.create(
        stage=WorkloadRunnerStage.ADAPTER,
        input_digest=input_digest,
        work_dir=work_dir,
        relative_paths=tuple(f"adapter/{name}" for name in sorted(outputs)),
    )
    artifact.verify_files(work_dir)
    return artifact


__all__ = [
    "LEGACY_DENSE_BACKEND_COMPATIBILITY_SCHEMA_VERSION",
    "LegacyDenseBackendCompatibility",
    "LegacyDenseBackendScope",
    "LegacyDenseCompileAdapter",
    "assess_legacy_dense_backend",
    "require_legacy_dense_backend",
    "run_legacy_dense_compiler_canary",
]
