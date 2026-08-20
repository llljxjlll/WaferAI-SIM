from __future__ import annotations

import inspect
import unittest
from typing import get_type_hints

from llm.frontend.wafer_frontend.lowering.interfaces import (
    CoarseLowering,
    IsaRegionLowering,
    ManifestLinker,
    StandaloneCollectiveLowering,
)
from llm.frontend.wafer_frontend.lowering import (
    LoweringContext,
    NaiveIsaRegionLowering,
)
from llm.frontend.wafer_frontend.policies.interfaces import (
    FusionPartition,
    GlobalActionDAGBuilder,
    InterDiePolicy,
    IntraDiePolicy,
    ProjectToIR2,
    StandaloneCollectivePolicy,
)
from llm.frontend.wafer_frontend.policies import (
    StandaloneCollectivePolicy as PublicStandaloneCollectivePolicy,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistrationState,
    RegistryKind,
    default_registry,
)
from llm.frontend.wafer_frontend.schema import FusionAction, FusionActionKind
from llm.frontend.wafer_frontend.schema.artifact_manifest import RegionManifest


class InterfaceContractTest(unittest.TestCase):
    def test_fusion_action_is_exported_from_public_schema_package(self) -> None:
        self.assertEqual(FusionAction.__name__, "FusionAction")
        self.assertEqual(FusionActionKind.COMP.value, "comp")

    def test_standalone_policy_is_exported_from_public_policy_package(self) -> None:
        self.assertIs(PublicStandaloneCollectivePolicy, StandaloneCollectivePolicy)

    def test_policy_signatures_are_frozen(self) -> None:
        expected = {
            FusionPartition.run: ("self", "ir1"),
            InterDiePolicy.plan: ("self", "ir1", "fused_op", "profile"),
            StandaloneCollectivePolicy.plan: (
                "self",
                "ir1",
                "collective_op",
                "profile",
            ),
            ProjectToIR2.run: (
                "self",
                "ir1",
                "fusion_plans",
                "standalone_plans",
                "state_transfers",
            ),
            IntraDiePolicy.schedule: ("self", "projection", "ir1"),
            GlobalActionDAGBuilder.build: ("self", "ir1", "projection", "schedule_set"),
        }
        for method, parameters in expected.items():
            self.assertEqual(tuple(inspect.signature(method).parameters), parameters)

    def test_lowering_signatures_are_frozen(self) -> None:
        expected = {
            CoarseLowering.lower: ("self", "action", "context"),
            StandaloneCollectiveLowering.lower: ("self", "actions", "context"),
            IsaRegionLowering.lower: ("self", "plan", "actions", "context"),
            ManifestLinker.link: ("self", "context", "fragments"),
        }
        for method, parameters in expected.items():
            self.assertEqual(tuple(inspect.signature(method).parameters), parameters)
        self.assertEqual(
            get_type_hints(IsaRegionLowering.lower)["return"],
            tuple[RegionManifest, ...],
        )

    def test_naive_isa_region_lowering_is_public(self) -> None:
        self.assertEqual(NaiveIsaRegionLowering.__name__, "NaiveIsaRegionLowering")

    def test_stage0_activates_naive_without_aliasing_optimized(self) -> None:
        entries = default_registry().entries()
        active = {
            (entry.kind, entry.name)
            for entry in entries
            if entry.state is RegistrationState.ACTIVE
        }
        self.assertEqual(
            active,
            {
                (RegistryKind.INTER_DIE, "naive"),
                (RegistryKind.INTRA_DIE, "naive"),
                (RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"),
            },
        )
        declared = {
            (entry.kind, entry.name)
            for entry in entries
            if entry.state is RegistrationState.DECLARED
        }
        self.assertIn((RegistryKind.INTER_DIE, "swizzle_topo"), declared)
        self.assertIn((RegistryKind.INTRA_DIE, "optimized"), declared)

    def test_all_forward_annotations_resolve_at_runtime(self) -> None:
        methods = (
            FusionPartition.run,
            InterDiePolicy.plan,
            StandaloneCollectivePolicy.plan,
            ProjectToIR2.run,
            IntraDiePolicy.schedule,
            GlobalActionDAGBuilder.build,
            CoarseLowering.lower,
            StandaloneCollectiveLowering.lower,
            IsaRegionLowering.lower,
            ManifestLinker.link,
        )
        for method in methods:
            self.assertIn("return", get_type_hints(method))

    def test_lowering_context_is_public_and_immutable(self) -> None:
        self.assertEqual(LoweringContext.__dataclass_params__.frozen, True)
        self.assertIn("global_dag", LoweringContext.__slots__)


if __name__ == "__main__":
    unittest.main()
