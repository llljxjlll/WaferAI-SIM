"""Frozen lowering interfaces; lowering implementations arrive in N6."""

from __future__ import annotations

from typing import Protocol

from ..schema.action import FusionPlan
from ..schema.artifact_manifest import (
    CommandFragment,
    LinkedProgramManifest,
    RegionManifest,
)
from ..schema.global_action import GlobalAction
from .context import LoweringContext


class CoarseLowering(Protocol):
    def lower(
        self,
        action: GlobalAction,
        context: LoweringContext,
    ) -> CommandFragment: ...


class StateDmaLowering(Protocol):
    def lower(
        self,
        action: GlobalAction,
        context: LoweringContext,
    ) -> CommandFragment: ...


class StateTransferLowering(Protocol):
    def lower(
        self,
        actions: tuple[GlobalAction, ...],
        context: LoweringContext,
    ) -> CommandFragment: ...


class StandaloneCollectiveLowering(Protocol):
    def lower(
        self,
        actions: tuple[GlobalAction, ...],
        context: LoweringContext,
    ) -> CommandFragment: ...


class IsaRegionLowering(Protocol):
    """Return local region wrappers in canonical die order.

    Producers flatten these tuples before passing the resulting RegionManifest
    sequence to ManifestLinker.
    """

    def lower(
        self,
        plan: FusionPlan,
        actions: tuple[GlobalAction, ...],
        context: LoweringContext,
    ) -> tuple[RegionManifest, ...]: ...


class ManifestLinker(Protocol):
    """Consume an already-flattened canonical fragment/region tuple."""

    def link(
        self,
        context: LoweringContext,
        fragments: tuple[CommandFragment | RegionManifest, ...],
    ) -> LinkedProgramManifest: ...
