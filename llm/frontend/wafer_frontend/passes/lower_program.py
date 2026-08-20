"""N6 leaf-fragment lowering producer orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..lowering.context import LoweringContext
from ..lowering.interfaces import (
    CoarseLowering,
    IsaRegionLowering,
    StateDmaLowering,
    StateTransferLowering,
    StandaloneCollectiveLowering,
)
from ..lowering.lifecycle import add_fixed_sram_lifecycle
from ..schema.artifact_manifest import (
    CommandFragment,
    RegionManifest,
)
from ..schema.ir2 import (
    FusedNodeOrigin,
    OrdinaryNodeOrigin,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
    StandaloneNodeOrigin,
)
from ..schema.n5 import (
    GlobalActionBundle,
    GlobalActionProfile,
    Stage4GlobalAction,
)
from ..schema.n6 import (
    LoweredProgramBundle,
    LoweredProgramProfile,
    Stage4LoweredProgram,
)
from ..schema.state_transfer import (
    SegmentedKvStateTransferContract,
    SlicedKvStateTransferContract,
)


@dataclass(frozen=True, slots=True)
class _LoweringDependencies:
    coarse: CoarseLowering
    isa: IsaRegionLowering
    standalone: StandaloneCollectiveLowering
    state: StateDmaLowering
    state_transfer: StateTransferLowering
    validate_intermediates: bool


def _resolve_dependencies(
    coarse_lowerer: CoarseLowering | None,
    isa_lowerer: IsaRegionLowering | None,
    standalone_lowerer: StandaloneCollectiveLowering | None,
    state_dma_lowerer: StateDmaLowering | None,
    state_transfer_lowerer: StateTransferLowering | None,
) -> _LoweringDependencies:
    use_fast_path = (
        coarse_lowerer is None
        and isa_lowerer is None
        and standalone_lowerer is None
        and state_dma_lowerer is None
        and state_transfer_lowerer is None
    )
    if coarse_lowerer is None:
        from ..lowering.coarse import NaiveCoarseLowering

        coarse_lowerer = NaiveCoarseLowering(validate_output=False)
    if isa_lowerer is None:
        from ..lowering.isa_region import NaiveIsaRegionLowering

        isa_lowerer = NaiveIsaRegionLowering(validate_output=False)
    if standalone_lowerer is None:
        from ..lowering.standalone import NaiveStandaloneCollectiveLowering

        standalone_lowerer = NaiveStandaloneCollectiveLowering(
            validate_output=False
        )
    if state_dma_lowerer is None:
        from ..lowering.state import NaiveStateDmaLowering

        state_dma_lowerer = NaiveStateDmaLowering(validate_output=False)
    if state_transfer_lowerer is None:
        from ..lowering.state_transfer import NaiveStateTransferLowering

        state_transfer_lowerer = NaiveStateTransferLowering(
            validate_output=False
        )
    return _LoweringDependencies(
        coarse=coarse_lowerer,
        isa=isa_lowerer,
        standalone=standalone_lowerer,
        state=state_dma_lowerer,
        state_transfer=state_transfer_lowerer,
        validate_intermediates=not use_fast_path,
    )


def _decorate_fragment(
    fragment: CommandFragment,
    context: LoweringContext,
    *,
    path: str,
    validate: bool,
) -> CommandFragment:
    if type(fragment) is not CommandFragment:
        raise SchemaError(
            "lowerer must return a CommandFragment",
            path=path,
        )
    if validate:
        decorated = add_fixed_sram_lifecycle(fragment, context)
    else:
        decorated = add_fixed_sram_lifecycle(fragment, context, validate=False)
    if type(decorated) is not CommandFragment:
        raise SchemaError(
            "lifecycle decorator must return a CommandFragment",
            path=path,
        )
    return decorated


def _decorate_region(
    region: RegionManifest,
    context: LoweringContext,
    *,
    path: str,
    validate: bool,
) -> RegionManifest:
    if type(region) is not RegionManifest:
        raise SchemaError(
            "ISA lowerer must return RegionManifest values",
            path=path,
        )
    decorated = RegionManifest.create(
        producer_pass=region.producer_pass,
        region_id=region.region_id,
        fusion_plan_id=region.fusion_plan_id,
        target_dies=region.target_dies,
        fragment=_decorate_fragment(
            region.fragment,
            context,
            path=f"{path}.fragment",
            validate=validate,
        ),
    )
    return decorated


def _lower_fragments(
    context: LoweringContext,
    dependencies: _LoweringDependencies,
) -> tuple[CommandFragment | RegionManifest, ...]:
    actions = context.global_dag.actions
    consumed: set[str] = set()
    fragments: list[CommandFragment | RegionManifest] = []

    # Per-action leaves retain their relative GlobalActionDAG tuple order.
    for action_index, action in enumerate(actions):
        if isinstance(action.origin_ref, StateIoOrigin):
            fragment = dependencies.state.lower(action, context)
            fragments.append(
                _decorate_fragment(
                    fragment,
                    context,
                    path=f"state_fragments[{len(fragments)}]",
                    validate=dependencies.validate_intermediates,
                )
            )
            consumed.add(action.id)
            continue
        if not isinstance(action.origin_ref, OrdinaryNodeOrigin):
            continue
        if action.task_kind is SemanticTaskKind.TRANSIT:
            raise SchemaError(
                "TRANSIT is only legal inside a standalone plan action tuple",
                path=f"source.global_dag.actions[{action_index}]",
            )
        fragment = dependencies.coarse.lower(action, context)
        fragments.append(
            _decorate_fragment(
                fragment,
                context,
                path=f"ordinary_fragments[{len(fragments)}]",
                validate=dependencies.validate_intermediates,
            )
        )
        consumed.add(action.id)

    # One transfer lowering call per contract endpoint die.  Coreless route
    # TRANSIT actions remain provenance witnesses: they are consumed here but
    # never passed to a leaf lowerer and never emit records.
    route_catalog = {
        route.id: route
        for group in context.ir1.groups
        for route in group.embedding.routes
    }
    for route in context.ir1.cross_routes:
        if route.id in route_catalog:
            raise SchemaError(
                "physical route ids must be globally unique",
                path="source.graph.cross_routes",
            )
        route_catalog[route.id] = route
    for transfer_index, contract in enumerate(
        context.projection.state_transfers
    ):
        route_ref = (
            contract.cross_group_route_ref
            if isinstance(
                contract,
                (
                    SegmentedKvStateTransferContract,
                    SlicedKvStateTransferContract,
                ),
            )
            else contract.pair_route_ref
        )
        route = route_catalog.get(route_ref)
        if route is None:
            raise SchemaError(
                "state transfer references an unknown physical route",
                path=(
                    "source.projection.state_transfers"
                    f"[{transfer_index}].route_ref"
                ),
            )
        contract_actions = tuple(
            action
            for action in actions
            if isinstance(action.origin_ref, StateTransferOrigin)
            and action.origin_ref.state_transfer_ref == contract.id
        )
        for endpoint_index, die_id in enumerate(
            (route.die_path[0], route.die_path[-1])
        ):
            endpoint_actions = tuple(
                action
                for action in contract_actions
                if action.logical_core is not None
                and action.logical_core.die_id == die_id
            )
            fragment = dependencies.state_transfer.lower(
                endpoint_actions, context
            )
            fragments.append(
                _decorate_fragment(
                    fragment,
                    context,
                    path=(
                        f"state_transfer_fragments[{transfer_index}]"
                        f"[{endpoint_index}]"
                    ),
                    validate=dependencies.validate_intermediates,
                )
            )
            consumed.update(action.id for action in endpoint_actions)
        consumed.update(
            action.id
            for action in contract_actions
            if action.task_kind is SemanticTaskKind.TRANSIT
        )

    # One ISA lowering call per plan, in the exact context plan order.  Transit
    # tasks are not executable ISA-region leaves.
    for plan_index, plan in enumerate(context.fusion_plans):
        plan_actions = tuple(
            action
            for action in actions
            if isinstance(action.origin_ref, FusedNodeOrigin)
            and action.origin_ref.plan_id == plan.id
            and action.task_kind is not SemanticTaskKind.TRANSIT
        )
        plan_transit = tuple(
            action
            for action in actions
            if isinstance(action.origin_ref, FusedNodeOrigin)
            and action.origin_ref.plan_id == plan.id
            and action.task_kind is SemanticTaskKind.TRANSIT
        )
        # Fused TRANSIT actions are route witnesses on intermediate dies.  The
        # ISA leaf is emitted only for executable endpoint actions, exactly as
        # the standalone lowerer filters its own coreless transit witnesses.
        # They still participate in the wrapper's exact action coverage below.
        regions = dependencies.isa.lower(plan, plan_actions, context)
        if type(regions) is not tuple:
            raise SchemaError(
                "ISA lowerer must return a tuple of RegionManifest",
                path=f"fusion_regions[{plan_index}]",
            )
        for region_index, region in enumerate(regions):
            fragments.append(
                _decorate_region(
                    region,
                    context,
                    path=(
                        f"fusion_regions[{plan_index}]"
                        f"[{region_index}]"
                    ),
                    validate=dependencies.validate_intermediates,
                )
            )
        consumed.update(action.id for action in plan_actions + plan_transit)

    # Standalone lowering consumes the complete plan action tuple, including
    # coreless TRANSIT actions used to prove the physical route.
    for plan_index, plan in enumerate(context.standalone_plans):
        plan_actions = tuple(
            action
            for action in actions
            if isinstance(action.origin_ref, StandaloneNodeOrigin)
            and action.origin_ref.collective_plan_id == plan.id
        )
        fragment = dependencies.standalone.lower(plan_actions, context)
        fragments.append(
            _decorate_fragment(
                fragment,
                context,
                path=f"standalone_fragments[{plan_index}]",
                validate=dependencies.validate_intermediates,
            )
        )
        consumed.update(action.id for action in plan_actions)

    action_ids = tuple(action.id for action in actions)
    if consumed != set(action_ids):
        missing = tuple(
            action_id for action_id in action_ids if action_id not in consumed
        )
        raise SchemaError(
            f"lowering groups must cover every action exactly; missing {missing!r}",
            path="source.global_dag.actions",
        )

    return tuple(fragments)


def _lower_profile(
    source: GlobalActionProfile,
    dependencies: _LoweringDependencies,
) -> LoweredProgramProfile:
    source.validate("source")
    context = source.lowering_context()
    result = LoweredProgramProfile.create(
        source=source,
        lowering_context=context,
        fragments=_lower_fragments(context, dependencies),
    )
    result.validate_against(source)
    return result


def lower_profile(
    source: GlobalActionProfile,
    coarse_lowerer: CoarseLowering | None = None,
    isa_lowerer: IsaRegionLowering | None = None,
    standalone_lowerer: StandaloneCollectiveLowering | None = None,
    state_dma_lowerer: StateDmaLowering | None = None,
    state_transfer_lowerer: StateTransferLowering | None = None,
) -> LoweredProgramProfile:
    """Lower one exact N5 global-action profile into canonical leaves."""

    if type(source) is not GlobalActionProfile:
        raise SchemaError(
            "must be a GlobalActionProfile",
            path="source",
        )
    dependencies = _resolve_dependencies(
        coarse_lowerer,
        isa_lowerer,
        standalone_lowerer,
        state_dma_lowerer,
        state_transfer_lowerer,
    )
    return _lower_profile(source, dependencies)


def lower_stage4(source: Stage4GlobalAction) -> Stage4LoweredProgram:
    """Lower one formal Stage 4 graph through production lowerers."""

    if type(source) is not Stage4GlobalAction:
        raise SchemaError(
            "must be a Stage4GlobalAction",
            path="source",
        )
    source.validate("source")
    context = LoweringContext(
        ir1=source.graph,
        fusion_plans=source.fusion_plans,
        standalone_plans=source.standalone_plans,
        projection=source.projection,
        schedule_set=source.schedule_set,
        global_dag=source.global_dag,
    )
    context.validate("lowering_context")
    dependencies = _resolve_dependencies(None, None, None, None, None)
    result = Stage4LoweredProgram.create(
        source=source,
        fragments=_lower_fragments(context, dependencies),
    )
    result.validate_against(source)
    return result


def lower_bundle(
    source: GlobalActionBundle,
    coarse_lowerer: CoarseLowering | None = None,
    isa_lowerer: IsaRegionLowering | None = None,
    standalone_lowerer: StandaloneCollectiveLowering | None = None,
    state_dma_lowerer: StateDmaLowering | None = None,
    state_transfer_lowerer: StateTransferLowering | None = None,
) -> LoweredProgramBundle:
    """Lower every source profile once in canonical bundle order."""

    if type(source) is not GlobalActionBundle:
        raise SchemaError(
            "must be a GlobalActionBundle",
            path="source",
        )
    source.validate("source")
    dependencies = _resolve_dependencies(
        coarse_lowerer,
        isa_lowerer,
        standalone_lowerer,
        state_dma_lowerer,
        state_transfer_lowerer,
    )
    entries = tuple(
        _lower_profile(entry, dependencies)
        for entry in source.entries
    )
    result = LoweredProgramBundle.create(source=source, entries=entries)
    result.validate_against(source)
    return result


__all__ = ["lower_bundle", "lower_profile", "lower_stage4"]
