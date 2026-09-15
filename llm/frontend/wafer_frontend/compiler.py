"""Production orchestration for the fixed naive frontend pipeline.

This module deliberately stops at the linked, symbolic manifest. Final
runtime-ID assignment and ProgramArtifact encoding are performed by the C++
``ProgramArtifactFinalizer`` so the simulator ABI has one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

from .errors import SchemaError, UnsupportedFeatureError
from .passes import (
    PassManager,
    PipelinePhase,
    build_global_bundle,
    build_ir0,
    link_bundle,
    logical_expand,
    lower_bundle,
    partition_bundle,
    place_bundle,
    plan_bundle,
    project_bundle,
    schedule_bundle,
)
from .passes.pass_manager import PipelineSnapshot
from .passes.intra_die_refine import refine_bundle
from .policies.registry import PolicyRegistry, RegistryKind, production_registry
from .schema.common import (
    ValidationMode,
    stable_artifact_id,
    validate_nonempty,
)
from .schema.experiment import (
    ExperimentSpec,
    InterDiePolicyName,
    IntraDiePolicyName,
    PlacementStrategy,
    WorkloadMode,
)
from .schema.ir1 import PhysicalFabric
from .schema.n4 import (
    FusedInterDieContract,
    FusionPartitionContext,
    InterDiePlanningContext,
)
from .schema.intra_die_refine import (
    IntraDieOptimizationOptions,
    IntraDieRefineContext,
    IntraDieRefineContract,
    RefinedIR2Bundle,
    SplitKRefineOptions,
)
from .schema.n5 import (
    IntraDieSchedulingContext,
    IntraDieSchedulingContract,
    ProjectToIR2Context,
    ProjectedIR2Bundle,
    ProjectedProfileIR2,
    PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
)
from .schema.n6 import LinkedProgramBundle
from .schema.placement import PlacementContext, PersistentStateReservationPolicy
from .schema.policy import PolicySelection
from .schema.persistent_state import HbmAddressSpace
from .schema.serde import canonical_digest
from .schema.rect_mesh import RectMeshSpec
from .schema.rect_mesh_compile import (
    RectMeshCompileCapabilityReport,
    RectMeshCompileChain,
    RectMeshCompileMode,
    RectMeshFallbackReason,
)


RECT_MESH_COMPILATION_SCHEMA_VERSION = (
    "wafer_frontend.rect_mesh_compilation/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class NaiveCompilation:
    """One immutable trace of the fixed SPEC -> MANIFEST_LINKED pipeline."""

    spec: ExperimentSpec
    fabric: PhysicalFabric
    contexts: tuple[object, ...]
    policy_selections: tuple[PolicySelection, ...]
    artifacts: tuple[object, ...]
    snapshot: PipelineSnapshot
    linked: LinkedProgramBundle

    def validate(self, path: str = "naive_compilation") -> None:
        self.spec.validate(f"{path}.spec")
        self.fabric.validate(f"{path}.fabric")
        if len(self.contexts) != 6:
            raise SchemaError(
                "must contain the six fixed naive pass contexts",
                path=f"{path}.contexts",
            )
        for index, context in enumerate(self.contexts):
            validator = getattr(context, "validate", None)
            if not callable(validator):
                raise SchemaError(
                    "context must expose validate()",
                    path=f"{path}.contexts[{index}]",
                )
            validator(f"{path}.contexts[{index}]")
        if len(self.artifacts) != 12 or self.artifacts[0] != self.spec:
            raise SchemaError(
                "must contain the exact spec plus eleven ordered pass outputs",
                path=f"{path}.artifacts",
            )
        if self.artifacts[-1] != self.linked:
            raise SchemaError(
                "last artifact must be the linked bundle",
                path=f"{path}.linked",
            )
        self.snapshot.validate(f"{path}.snapshot")
        self.linked.validate(f"{path}.linked")
        if self.snapshot.phase is not PipelinePhase.MANIFEST_LINKED:
            raise SchemaError(
                "pipeline must stop at MANIFEST_LINKED",
                path=f"{path}.snapshot.phase",
            )
        if len(self.snapshot.receipts) != 11:
            raise SchemaError(
                "must contain eleven fixed pass receipts",
                path=f"{path}.snapshot.receipts",
            )
        for index, receipt in enumerate(self.snapshot.receipts):
            if (
                receipt.input_digest != canonical_digest(self.artifacts[index])
                or receipt.output_digest
                != canonical_digest(self.artifacts[index + 1])
            ):
                raise SchemaError(
                    "receipt does not identify its exact adjacent artifacts",
                    path=f"{path}.snapshot.receipts[{index}]",
                )
        expected_context_digests = (
            None,
            None,
            *(canonical_digest(context) for context in self.contexts),
            None,
            None,
            None,
        )
        if tuple(
            receipt.context_digest for receipt in self.snapshot.receipts
        ) != expected_context_digests:
            raise SchemaError(
                "receipt context digests disagree with the six fixed contexts",
                path=f"{path}.snapshot.receipts",
            )
        planning_context = self.contexts[2]
        scheduling_context = self.contexts[5]
        if type(planning_context) is not InterDiePlanningContext:
            raise SchemaError(
                "third context must be InterDiePlanningContext",
                path=f"{path}.contexts[2]",
            )
        if type(scheduling_context) is not IntraDieSchedulingContext:
            raise SchemaError(
                "sixth context must be IntraDieSchedulingContext",
                path=f"{path}.contexts[5]",
            )
        expected_summary = (
            planning_context.fused_policy,
            planning_context.standalone_policy,
            scheduling_context.policy,
        )
        if self.policy_selections != expected_summary:
            raise SchemaError(
                "policy selection summary disagrees with the fixed contexts",
                path=f"{path}.policy_selections",
            )
        expected_policy_selections = (
            (),
            (),
            (),
            (),
            (
                planning_context.fused_policy,
                planning_context.standalone_policy,
            ),
            (),
            (),
            (scheduling_context.policy,),
            (),
            (),
            (),
        )
        if tuple(
            receipt.policy_selections for receipt in self.snapshot.receipts
        ) != expected_policy_selections:
            raise SchemaError(
                "receipt policy selections disagree with the fixed contexts",
                path=f"{path}.snapshot.receipts",
            )


@dataclass(frozen=True, slots=True)
class RectMeshCompilation:
    """Explicit opt-in wrapper for one rectangular whole-workload compile."""

    schema_version: str
    producer_pass: str
    id: str
    requested_spec: ExperimentSpec
    mesh: RectMeshSpec
    requested_mode: RectMeshCompileMode
    compilation: NaiveCompilation
    capability_report: RectMeshCompileCapabilityReport

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        requested_spec: ExperimentSpec,
        mesh: RectMeshSpec,
        requested_mode: RectMeshCompileMode,
        compilation: NaiveCompilation,
        capability_report: RectMeshCompileCapabilityReport,
    ) -> "RectMeshCompilation":
        semantic = {
            "requested_spec": requested_spec,
            "mesh": mesh,
            "requested_mode": requested_mode,
            "compilation": compilation,
            "capability_report": capability_report,
        }
        result = cls(
            schema_version=RECT_MESH_COMPILATION_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "rect_mesh_compilation",
                semantic,
                schema_version=RECT_MESH_COMPILATION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "requested_spec": self.requested_spec,
            "mesh": self.mesh,
            "requested_mode": self.requested_mode,
            "compilation": self.compilation,
            "capability_report": self.capability_report,
        }

    def validate(self, path: str = "rect_mesh_compilation") -> None:
        if self.schema_version != RECT_MESH_COMPILATION_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.requested_spec) is not ExperimentSpec:
            raise SchemaError(
                "must be an ExperimentSpec", path=f"{path}.requested_spec"
            )
        if type(self.mesh) is not RectMeshSpec:
            raise SchemaError("must be a RectMeshSpec", path=f"{path}.mesh")
        if type(self.requested_mode) is not RectMeshCompileMode:
            raise SchemaError(
                "must be a RectMeshCompileMode", path=f"{path}.requested_mode"
            )
        if type(self.compilation) is not NaiveCompilation:
            raise SchemaError(
                "must be a NaiveCompilation", path=f"{path}.compilation"
            )
        _validate_rect_mesh_compile_inputs(
            self.requested_spec, self.compilation.fabric, self.mesh
        )
        self.compilation.validate(f"{path}.compilation")
        if self.compilation.spec != _rect_mesh_baseline_spec(self.requested_spec):
            raise SchemaError(
                "executable fallback must use the exact naive policy rewrite",
                path=f"{path}.compilation.spec.policy.inter_die",
            )
        if type(self.capability_report) is not RectMeshCompileCapabilityReport:
            raise SchemaError(
                "must be a RectMeshCompileCapabilityReport",
                path=f"{path}.capability_report",
            )
        self.capability_report.validate(f"{path}.capability_report")
        if (
            self.capability_report.producer_pass != self.producer_pass
            or self.capability_report.mesh != self.mesh
            or self.capability_report.requested_mode is not self.requested_mode
            or self.capability_report.selected_chain
            is not RectMeshCompileChain.NAIVE_FIXED_V1
        ):
            raise SchemaError(
                "capability report disagrees with compiler dispatch", path=path
            )
        expected = stable_artifact_id(
            "rect_mesh_compilation",
            self._semantic_key(),
            schema_version=RECT_MESH_COMPILATION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )


def _fused_inter_die_contract_for_policy(
    policy: PolicySelection,
) -> FusedInterDieContract:
    contracts = {
        "naive": FusedInterDieContract.DIRECT_NAIVE_V1,
        "swizzle_topo": FusedInterDieContract.SWIZZLE_TOPO_V1,
    }
    try:
        return contracts[policy.name]
    except KeyError as exc:
        raise SchemaError(
            "unsupported inter-die policy for compiler contract",
            path="spec.policy.inter_die",
        ) from exc


def _intra_die_contract_for_policy(
    policy: PolicySelection,
) -> IntraDieSchedulingContract:
    contracts = {
        "naive": (
            IntraDieSchedulingContract
            .NAIVE_COMPONENT_RR_XY_SEQUENTIAL_STATE_TRANSFER_V5
        ),
        "optimized": IntraDieSchedulingContract.OPTIMIZED_CRITICAL_PATH_XY_V1,
    }
    try:
        return contracts[policy.name]
    except KeyError as exc:
        raise SchemaError(
            "unsupported intra-die policy for compiler contract",
            path="spec.policy.intra_die",
        ) from exc


def _refine_contract_for_policy(
    policy: PolicySelection,
    options: SplitKRefineOptions | IntraDieOptimizationOptions | None,
) -> tuple[IntraDieRefineContract, SplitKRefineOptions | IntraDieOptimizationOptions]:
    if policy.name not in ("naive", "optimized"):
        raise SchemaError(
            "unsupported intra-die policy for refine contract",
            path="spec.policy.intra_die",
        )
    if options is not None and type(options) not in (
        SplitKRefineOptions, IntraDieOptimizationOptions,
    ):
        raise SchemaError(
            "must be SplitKRefineOptions or IntraDieOptimizationOptions",
            path="intra_die_refine_options",
        )
    if options is None or options == SplitKRefineOptions():
        return IntraDieRefineContract.IDENTITY_V1, SplitKRefineOptions()
    options.validate("intra_die_refine_options")
    if policy.name != "optimized":
        raise SchemaError(
            "split-K graph refinement requires intra_die=optimized",
            path="intra_die_refine_options",
        )
    return IntraDieRefineContract.SPLIT_K_REDUCE_DOUBLE_BUFFER_V2, options


def _refined_projected_schedule_view(source: RefinedIR2Bundle) -> ProjectedIR2Bundle:
    entries: list[ProjectedProfileIR2] = []
    for refined in source.entries:
        projection = (
            refined.split_k_refinement.projection
            if refined.split_k_refinement is not None
            else refined.projection
        )
        entry = replace(refined.source, id="", projection=projection)
        entry = replace(
            entry,
            id=stable_artifact_id(
                "projected_profile_ir2", entry._semantic_key(),
                schema_version=PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
            ),
        )
        entry.validate("refined_projected_schedule_view.entry")
        entries.append(entry)
    view = replace(source.source, id="", entries=tuple(entries))
    view = replace(
        view,
        id=stable_artifact_id(
            "projected_ir2_bundle", view._semantic_key(),
            schema_version=PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
        ),
    )
    view.validate("refined_projected_schedule_view")
    return view

def _schedule_refined_bundle(source: RefinedIR2Bundle, context: IntraDieSchedulingContext, policy: object):
    if type(source) is not RefinedIR2Bundle:
        raise SchemaError("must be a RefinedIR2Bundle", path="source")
    source.validate("source")
    return schedule_bundle(_refined_projected_schedule_view(source), context, policy)


def compile_naive(
    spec: ExperimentSpec,
    fabric: PhysicalFabric,
    *,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
    persistent_state_reservation_policy: PersistentStateReservationPolicy | None = None,
    producer_pass: str = "naive_frontend",
    registry: PolicyRegistry | None = None,
    intra_die_refine_options: SplitKRefineOptions | IntraDieOptimizationOptions | None = None,
    intra_die_wire_address_limit_bytes: int | None = None,
) -> NaiveCompilation:
    """Compile one validated experiment through the fixed naive Python path."""

    if type(spec) is not ExperimentSpec:
        raise SchemaError("must be an ExperimentSpec", path="spec")
    if type(fabric) is not PhysicalFabric:
        raise SchemaError("must be a PhysicalFabric", path="fabric")
    validate_nonempty(producer_pass, "producer_pass")
    spec.validate("spec")
    fabric.validate("fabric")
    if registry is None:
        registry = production_registry()
    elif type(registry) is not PolicyRegistry:
        raise SchemaError("must be a PolicyRegistry", path="registry")
    inter_die_policy = registry.instantiate(
        RegistryKind.INTER_DIE,
        spec.policy.inter_die.value,
    )
    standalone_policy = registry.instantiate(
        RegistryKind.STANDALONE_COLLECTIVE,
        "direct_all_gather",
    )
    intra_die_policy = registry.instantiate(
        RegistryKind.INTRA_DIE,
        spec.policy.intra_die.value,
    )
    schedule_implementation = intra_die_policy.implementation
    if intra_die_wire_address_limit_bytes is not None:
        if spec.policy.intra_die is not IntraDiePolicyName.NAIVE:
            raise SchemaError(
                "wire-addressable SRAM compaction requires the NAIVE scheduler",
                path="intra_die_wire_address_limit_bytes",
            )
        from .policies.naive_intra_die import NaiveIntraDiePolicy

        schedule_implementation = NaiveIntraDiePolicy(
            wire_address_limit_bytes=intra_die_wire_address_limit_bytes,
        )

    placement_context = PlacementContext.create(
        producer_pass=producer_pass,
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces,
        persistent_state_reservation_policy=persistent_state_reservation_policy,
    )
    partition_context = FusionPartitionContext.create(
        producer_pass=producer_pass
    )
    planning_context = InterDiePlanningContext.create(
        producer_pass=producer_pass,
        fused_policy=inter_die_policy.selection,
        standalone_policy=standalone_policy.selection,
        fused_contract=_fused_inter_die_contract_for_policy(
            inter_die_policy.selection
        ),
    )
    projection_context = ProjectToIR2Context.create(
        producer_pass=producer_pass,
        state_transfers=(),
    )
    refine_contract, refine_options = _refine_contract_for_policy(
        intra_die_policy.selection, intra_die_refine_options
    )
    refine_context = IntraDieRefineContext.create(
        producer_pass=producer_pass,
        policy=intra_die_policy.selection,
        contract=refine_contract,
        options=refine_options,
    )
    scheduling_context = IntraDieSchedulingContext.create(
        producer_pass=producer_pass,
        policy=intra_die_policy.selection,
        contract=_intra_die_contract_for_policy(intra_die_policy.selection),
    )
    contexts: tuple[object, ...] = (
        placement_context,
        partition_context,
        planning_context,
        projection_context,
        refine_context,
        scheduling_context,
    )

    selected_plan_bundle = partial(
        plan_bundle,
        fused_policy=inter_die_policy.implementation,
        standalone_policy=standalone_policy.implementation,
    )
    selected_schedule_bundle = partial(
        _schedule_refined_bundle,
        policy=schedule_implementation,
    )

    manager = PassManager()
    stages = (
        ("build_ir0", build_ir0, None, ()),
        ("logical_expand", logical_expand, None, ()),
        ("placement", place_bundle, placement_context, ()),
        ("fusion_partition", partition_bundle, partition_context, ()),
        (
            "inter_die_plan",
            selected_plan_bundle,
            planning_context,
            (inter_die_policy.selection, standalone_policy.selection),
        ),
        ("project_to_ir2", project_bundle, projection_context, ()),
        ("intra_die_refine", refine_bundle, refine_context, ()),
        (
            "intra_die_schedule",
            selected_schedule_bundle,
            scheduling_context,
            (intra_die_policy.selection,),
        ),
        ("global_action_dag", build_global_bundle, None, ()),
        ("lowering", lower_bundle, None, ()),
        ("manifest_link", link_bundle, None, ()),
    )
    artifacts: list[object] = [spec]
    value: object = spec
    for pass_name, transform, context, policy_selections in stages:
        value = manager.run_pass(
            pass_name,
            value,
            transform,
            context=context,
            policy_selections=policy_selections,
        )
        artifacts.append(value)
    if type(value) is not LinkedProgramBundle:
        raise SchemaError(
            "manifest_link did not produce a LinkedProgramBundle",
            path="linked",
        )
    result = NaiveCompilation(
        spec=spec,
        fabric=fabric,
        contexts=contexts,
        policy_selections=(
            inter_die_policy.selection,
            standalone_policy.selection,
            intra_die_policy.selection,
        ),
        artifacts=tuple(artifacts),
        snapshot=manager.snapshot,
        linked=value,
    )
    result.validate()
    return result


def _rect_mesh_baseline_spec(spec: ExperimentSpec) -> ExperimentSpec:
    """Return the exact policy-only rewrite used by executable fallback."""

    return replace(
        spec,
        policy=replace(spec.policy, inter_die=InterDiePolicyName.NAIVE),
    )


def _validate_rect_mesh_compile_inputs(
    spec: ExperimentSpec,
    fabric: PhysicalFabric,
    mesh: RectMeshSpec,
) -> None:
    """Fail closed before dispatching a whole-workload RectMesh compile."""

    if type(spec) is not ExperimentSpec:
        raise SchemaError("must be an ExperimentSpec", path="spec")
    if type(fabric) is not PhysicalFabric:
        raise SchemaError("must be a PhysicalFabric", path="fabric")
    if type(mesh) is not RectMeshSpec:
        raise SchemaError("must be a RectMeshSpec", path="rect_mesh")
    spec.validate("spec")
    fabric.validate("fabric")
    mesh.validate("rect_mesh")
    if spec.workload.mode is not WorkloadMode.INFER:
        raise UnsupportedFeatureError(
            "RectMesh v1 whole-workload dispatch is Dense inference only",
            path="spec.workload.mode",
            code="rect_mesh_dense_inference_only",
        )
    infer = spec.workload.infer
    assert infer is not None
    if len(infer.profiles()) != 1:
        raise UnsupportedFeatureError(
            "RectMesh v1 whole-workload dispatch requires one static profile",
            path="spec.workload.infer",
            code="rect_mesh_static_profile_only",
        )
    if (
        spec.backend.reduction_contract.validation
        is not ValidationMode.TIMING
    ):
        raise UnsupportedFeatureError(
            "RectMesh v1 supports timing validation only",
            path="spec.backend.reduction_contract.validation",
            code=(
                RectMeshFallbackReason
                .UNSUPPORTED_FUNCTIONAL_EXECUTION.value
            ),
        )
    if (
        fabric.die_grid != mesh.physical_shape
        or len(fabric.dies) != mesh.rank_count
        or len(fabric.links) != mesh.directed_link_count
    ):
        raise SchemaError(
            "fabric must be the exact complete rectangular Mesh",
            path="fabric.die_grid",
            code=RectMeshFallbackReason.INVALID_MESH.value,
        )
    actual_rank_coordinates = tuple(
        (die.id, die.coord) for die in fabric.dies
    )
    expected_rank_coordinates = tuple(
        (rank, mesh.coordinate(rank)) for rank in range(mesh.rank_count)
    )
    if actual_rank_coordinates != expected_rank_coordinates:
        raise SchemaError(
            "fabric dies must be hole-free row-major rank coordinates",
            path="fabric.dies",
            code=RectMeshFallbackReason.INVALID_MESH.value,
        )
    if len(spec.parallel.instances) != 1:
        raise UnsupportedFeatureError(
            "RectMesh v1 requires exactly one Dense instance",
            path="spec.parallel.instances",
            code=RectMeshFallbackReason.INCOMPATIBLE_SHARDING.value,
        )
    instance = spec.parallel.instances[0]
    if instance.tp > mesh.rank_count:
        raise SchemaError(
            f"TP cannot exceed rectangular Mesh rank count {mesh.rank_count}",
            path="spec.parallel.instances[0].tp",
            code=RectMeshFallbackReason.INCOMPATIBLE_SHARDING.value,
        )
    if spec.placement.strategy is PlacementStrategy.COMPACT:
        if instance.tp != mesh.rank_count:
            raise SchemaError(
                "a partial TP group requires explicit physical Die placement",
                path="spec.placement.strategy",
                code=RectMeshFallbackReason.INVALID_PLACEMENT.value,
            )
    elif spec.placement.strategy is PlacementStrategy.EXPLICIT:
        expected_key = (instance.id, f"{instance.id}.mesh.tp")
        if (
            len(spec.placement.groups) != 1
            or (
                spec.placement.groups[0].instance_id,
                spec.placement.groups[0].mesh_ref,
            )
            != expected_key
            or len(spec.placement.groups[0].die_ids) != instance.tp
            or any(
                die_id >= mesh.rank_count
                for die_id in spec.placement.groups[0].die_ids
            )
        ):
            raise SchemaError(
                "explicit placement must bind exactly TP ranks to physical Mesh Dies",
                path="spec.placement.groups",
                code=RectMeshFallbackReason.INVALID_PLACEMENT.value,
            )


def compile_rect_mesh(
    spec: ExperimentSpec,
    fabric: PhysicalFabric,
    *,
    rect_mesh: RectMeshSpec,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
    persistent_state_reservation_policy: PersistentStateReservationPolicy | None = None,
    mode: RectMeshCompileMode = RectMeshCompileMode.AUTO,
    producer_pass: str = "rect_mesh_frontend",
    registry: PolicyRegistry | None = None,
    intra_die_refine_options: (
        SplitKRefineOptions | IntraDieOptimizationOptions | None
    ) = None,
    intra_die_wire_address_limit_bytes: int | None = None,
) -> RectMeshCompilation:
    """Compile an H by W Dense workload with diagnosable dispatch.

    AUTO currently selects the existing whole-workload chain as an executable
    fallback. The independent standard Swizzle chain cannot yet replace
    selected regions while retaining ordinary and standalone regions in the
    same artifact. Forced STANDARD therefore fails before any compiler pass
    instead of emitting a partial manifest.
    """

    validate_nonempty(producer_pass, "producer_pass")
    if type(mode) is not RectMeshCompileMode:
        raise SchemaError("must be a RectMeshCompileMode", path="mode")
    _validate_rect_mesh_compile_inputs(spec, fabric, rect_mesh)
    if mode is RectMeshCompileMode.STANDARD:
        raise UnsupportedFeatureError(
            (
                "RectMesh standard chain does not yet provide whole-workload "
                "ordinary/fusion/standalone replacement"
            ),
            path="mode",
            code=RectMeshFallbackReason.STANDARD_CHAIN_UNAVAILABLE.value,
            hint="use mode=AUTO for an executable naive fallback",
        )
    baseline_spec = _rect_mesh_baseline_spec(spec)
    compilation = compile_naive(
        baseline_spec,
        fabric,
        hbm_address_spaces=hbm_address_spaces,
        persistent_state_reservation_policy=persistent_state_reservation_policy,
        producer_pass=producer_pass,
        registry=registry,
        intra_die_refine_options=intra_die_refine_options,
        intra_die_wire_address_limit_bytes=intra_die_wire_address_limit_bytes,
    )
    manifests = tuple(entry.manifest for entry in compilation.linked.entries)
    report = RectMeshCompileCapabilityReport.create(
        producer_pass=producer_pass,
        mesh=rect_mesh,
        requested_mode=mode,
        selected_chain=RectMeshCompileChain.NAIVE_FIXED_V1,
        fallback_reasons=(
            ()
            if mode is RectMeshCompileMode.NAIVE
            else (RectMeshFallbackReason.STANDARD_CHAIN_UNAVAILABLE,)
        ),
        profile_count=len(compilation.linked.source_profiles),
        manifest_count=len(manifests),
        fragment_count=sum(len(manifest.fragments) for manifest in manifests),
        symbolic_record_count=sum(
            len(stream.records)
            for manifest in manifests
            for stream in manifest.core_streams
        ),
    )
    return RectMeshCompilation.create(
        producer_pass=producer_pass,
        requested_spec=spec,
        mesh=rect_mesh,
        requested_mode=mode,
        compilation=compilation,
        capability_report=report,
    )
