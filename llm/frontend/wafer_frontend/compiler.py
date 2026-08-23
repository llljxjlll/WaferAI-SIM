"""Production orchestration for the fixed naive frontend pipeline.

This module deliberately stops at the linked, symbolic manifest. Final
runtime-ID assignment and ProgramArtifact encoding are performed by the C++
``ProgramArtifactFinalizer`` so the simulator ABI has one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

from .errors import SchemaError
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
from .schema.common import stable_artifact_id, validate_nonempty
from .schema.experiment import ExperimentSpec
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
from .schema.placement import PlacementContext
from .schema.policy import PolicySelection
from .schema.persistent_state import HbmAddressSpace
from .schema.serde import canonical_digest


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
    producer_pass: str = "naive_frontend",
    registry: PolicyRegistry | None = None,
    intra_die_refine_options: SplitKRefineOptions | IntraDieOptimizationOptions | None = None,
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

    placement_context = PlacementContext.create(
        producer_pass=producer_pass,
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces,
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
        policy=intra_die_policy.implementation,
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
