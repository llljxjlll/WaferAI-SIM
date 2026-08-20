from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend import compile_naive
from llm.frontend.wafer_frontend import compiler as compiler_module
from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    StageNotImplementedError,
)
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.passes.pass_manager import (
    PIPELINE_SCHEMA_VERSION,
)
from llm.frontend.wafer_frontend.policies.registry import (
    PolicyRegistry,
    RegistryKind,
)
from llm.frontend.wafer_frontend.schema.action import (
    FusionPlan,
    StandaloneCollectivePlan,
)
from llm.frontend.wafer_frontend.schema.common import (
    ProfileKey,
    stable_artifact_id,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir1 import (
    FusedOpSkeleton,
    IR1,
    PhysicalNode,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    IR2ProjectionResult,
    IntraDieScheduleSet,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, from_data

from _fixtures import valid_spec
from test_n6_pipeline import _e1_compile_inputs


def _tiny_tp2_spec() -> ExperimentSpec:
    raw = valid_spec()
    model = raw["model"]
    assert isinstance(model, dict)
    model.update({"H": 8, "I": 16, "NH": 2, "KVH": 2, "DH": 4, "L": 1})
    infer = raw["workload"]["infer"]  # type: ignore[index]
    assert isinstance(infer, dict)
    profile = infer["profile"]
    assert isinstance(profile, dict)
    profile.update(
        {
            "prefill_tokens": 2,
            "context_sum": 2,
            "context_max": 2,
        }
    )
    return from_data(ExperimentSpec, raw, path="spec")


def _activate(
    registry: PolicyRegistry,
    kind: RegistryKind,
    name: str,
    factory: object,
) -> None:
    registry.activate(
        kind,
        name,
        factory,  # type: ignore[arg-type]
        implementation_id=f"test.compiler.{kind.value}.{name}",
        implementation_schema_version=f"test.compiler.{kind.value}/{name}/v1",
        capability_ids=("s1.gemm_collective.naive",),
    )


def _recording_registry(calls: list[tuple[str, str]]) -> PolicyRegistry:
    registry = PolicyRegistry()
    registry.declare(
        RegistryKind.INTER_DIE,
        "naive",
        interface="InterDiePolicy",
        available_stage="N4",
    )
    registry.declare(
        RegistryKind.STANDALONE_COLLECTIVE,
        "direct_all_gather",
        interface="StandaloneCollectivePolicy",
        available_stage="N4",
    )
    registry.declare(
        RegistryKind.INTRA_DIE,
        "naive",
        interface="IntraDiePolicy",
        available_stage="N5",
    )

    class Inter:
        def plan(
            self,
            ir1: IR1,
            fused_op: FusedOpSkeleton,
            profile: ProfileKey,
        ) -> FusionPlan:
            calls.append(("inter", fused_op.id))
            return NaiveInterDiePolicy().plan(ir1, fused_op, profile)

    class Standalone:
        def plan(
            self,
            ir1: IR1,
            collective_op: PhysicalNode,
            profile: ProfileKey,
        ) -> StandaloneCollectivePlan:
            calls.append(("standalone", collective_op.id))
            return DirectAllGatherPolicy().plan(ir1, collective_op, profile)

    class Intra:
        def schedule(
            self,
            projection: IR2ProjectionResult,
            ir1: IR1,
        ) -> IntraDieScheduleSet:
            calls.append(("intra", projection.id))
            return NaiveIntraDiePolicy().schedule(projection, ir1)

    _activate(registry, RegistryKind.INTER_DIE, "naive", Inter)
    _activate(
        registry,
        RegistryKind.STANDALONE_COLLECTIVE,
        "direct_all_gather",
        Standalone,
    )
    _activate(registry, RegistryKind.INTRA_DIE, "naive", Intra)
    return registry


class CompilerPolicyWiringTest(unittest.TestCase):
    def test_custom_registry_is_injected_and_semantically_equivalent(self) -> None:
        spec = _tiny_tp2_spec()
        fabric, hbm_address_spaces = _e1_compile_inputs()
        baseline = compile_naive(
            spec,
            fabric,
            hbm_address_spaces=hbm_address_spaces,
            producer_pass="compiler_policy_wiring",
        )
        calls: list[tuple[str, str]] = []
        selected = compile_naive(
            spec,
            fabric,
            hbm_address_spaces=hbm_address_spaces,
            producer_pass="compiler_policy_wiring",
            registry=_recording_registry(calls),
        )
        self.assertEqual(
            baseline.contexts[0].hbm_address_spaces, hbm_address_spaces
        )
        self.assertEqual(
            baseline.snapshot.receipts[2].context_digest,
            canonical_digest(baseline.contexts[0]),
        )
        self.assertEqual(selected.artifacts[:5], baseline.artifacts[:5])
        self.assertNotEqual(selected.artifacts[5:], baseline.artifacts[5:])
        for selected_entry, baseline_entry in zip(
            selected.artifacts[5].entries,
            baseline.artifacts[5].entries,
        ):
            self.assertEqual(
                selected_entry.fusion_plans,
                baseline_entry.fusion_plans,
            )
            self.assertEqual(
                selected_entry.standalone_plans,
                baseline_entry.standalone_plans,
            )
        for artifact_index, payload_name in (
            (6, "projection"),
            (7, "schedule_set"),
            (8, "global_dag"),
            (9, "fragments"),
            (10, "manifest"),
        ):
            self.assertEqual(
                tuple(
                    getattr(entry, payload_name)
                    for entry in selected.artifacts[artifact_index].entries
                ),
                tuple(
                    getattr(entry, payload_name)
                    for entry in baseline.artifacts[artifact_index].entries
                ),
            )
        self.assertNotEqual(selected.snapshot.id, baseline.snapshot.id)
        self.assertEqual(
            selected.policy_selections,
            (
                selected.contexts[2].fused_policy,
                selected.contexts[2].standalone_policy,
                selected.contexts[4].policy,
            ),
        )
        self.assertNotEqual(
            selected.policy_selections,
            baseline.policy_selections,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "policy selection summary disagrees",
        ):
            replace(
                selected,
                policy_selections=baseline.policy_selections,
            ).validate()
        selected_policy_ids = tuple(
            selection.implementation_id
            for receipt in selected.snapshot.receipts
            for selection in receipt.policy_selections
        )
        baseline_policy_ids = tuple(
            selection.implementation_id
            for receipt in baseline.snapshot.receipts
            for selection in receipt.policy_selections
        )
        self.assertNotEqual(selected_policy_ids, baseline_policy_ids)
        tampered_receipts = list(selected.snapshot.receipts)
        tampered_receipts[4] = replace(
            tampered_receipts[4],
            policy_selections=baseline.snapshot.receipts[4].policy_selections,
        )
        tampered_receipt_tuple = tuple(tampered_receipts)
        tampered_snapshot = replace(
            selected.snapshot,
            id=stable_artifact_id(
                "pipeline",
                {
                    "phase": selected.snapshot.phase.value,
                    "receipts": tampered_receipt_tuple,
                },
                schema_version=PIPELINE_SCHEMA_VERSION,
            ),
            receipts=tampered_receipt_tuple,
        )
        tampered_snapshot.validate()
        with self.assertRaisesRegex(
            SchemaError,
            "receipt policy selections disagree",
        ):
            replace(selected, snapshot=tampered_snapshot).validate()
        self.assertEqual(
            tuple(kind for kind, _identifier in calls),
            ("inter", "inter", "standalone", "standalone", "intra"),
        )

    def test_declared_optimized_fails_before_first_pass_without_mutation(self) -> None:
        raw = valid_spec()
        raw["policy"]["inter_die"] = "swizzle_topo"  # type: ignore[index]
        spec = from_data(ExperimentSpec, raw, path="spec")
        fabric, hbm_address_spaces = _e1_compile_inputs()
        before = (canonical_digest(spec), canonical_digest(fabric))
        with patch.object(
            compiler_module,
            "build_ir0",
            side_effect=AssertionError("first pass must not run"),
        ) as first_pass:
            with self.assertRaisesRegex(
                StageNotImplementedError, "swizzle_topo.*O1"
            ):
                compile_naive(
                    spec,
                    fabric,
                    hbm_address_spaces=hbm_address_spaces,
                )
        first_pass.assert_not_called()
        self.assertEqual(before, (canonical_digest(spec), canonical_digest(fabric)))

    def test_registry_type_is_exact(self) -> None:
        spec = _tiny_tp2_spec()
        fabric, hbm_address_spaces = _e1_compile_inputs()
        with self.assertRaisesRegex(SchemaError, "PolicyRegistry"):
            compile_naive(
                spec,
                fabric,
                hbm_address_spaces=hbm_address_spaces,
                registry=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
