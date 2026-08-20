from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from llm.frontend.wafer_frontend.errors import (
    InputMutationError,
    PassOrderError,
    SchemaError,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.passes.pass_manager import (
    PASS_SPECS,
    PassReceipt,
    PassManager,
    PipelinePhase,
    pipeline_description,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json


@dataclass(frozen=True, slots=True)
class ValidatedOutput:
    value: int
    accepted: bool

    def validate(self, path: str) -> None:
        if not self.accepted:
            raise SchemaError("output rejected by validator", path=path)


class PassManagerTest(unittest.TestCase):
    def test_fixed_order_and_receipt(self) -> None:
        manager = PassManager()
        output = manager.run_pass("build_ir0", {"spec": 1}, lambda value: {"ir0": value})
        self.assertEqual(output, {"ir0": {"spec": 1}})
        self.assertEqual(manager.snapshot.phase, PipelinePhase.IR0_READY)
        self.assertEqual(manager.snapshot.receipts[0].pass_name, "build_ir0")

    def test_skipping_a_pass_is_rejected_without_state_change(self) -> None:
        manager = PassManager()
        before = manager.snapshot
        with self.assertRaises(PassOrderError):
            manager.run_pass("placement", {}, lambda value: value)
        self.assertEqual(manager.snapshot, before)

    def test_input_mutation_is_detected_and_not_committed(self) -> None:
        manager = PassManager()
        input_value = {"items": []}

        def mutate(value: dict[str, list[int]]) -> dict[str, int]:
            value["items"].append(1)
            return {"result": 1}

        with self.assertRaises(InputMutationError):
            manager.run_pass("build_ir0", input_value, mutate)
        self.assertEqual(manager.snapshot.phase, PipelinePhase.SPEC_VALIDATED)

    def test_snapshot_is_reproducible(self) -> None:
        first = PassManager()
        second = PassManager()
        first.run_pass("build_ir0", {"x": 1}, lambda value: (value["x"],))
        second.run_pass("build_ir0", {"x": 1}, lambda value: (value["x"],))
        self.assertEqual(first.snapshot, second.snapshot)

    def test_dump_description_has_every_transition(self) -> None:
        description = pipeline_description()
        self.assertEqual(description.passes, PASS_SPECS)
        self.assertEqual(canonical_json(description), canonical_json(description))

    def test_new_manager_cannot_start_from_an_intermediate_phase(self) -> None:
        for phase in PipelinePhase:
            if phase is PipelinePhase.SPEC_VALIDATED:
                continue
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(PassOrderError, "must start"):
                    PassManager(phase)

    def test_next_pass_must_consume_the_previous_output_digest(self) -> None:
        manager = PassManager()
        output = manager.run_pass(
            "build_ir0", {"spec": 1}, lambda value: {"ir0": value}
        )
        before = manager.snapshot
        with self.assertRaisesRegex(PassOrderError, "previous pass output digest"):
            manager.run_pass(
                "logical_expand",
                {"unrelated": True},
                lambda value: value,
            )
        self.assertEqual(manager.snapshot, before)
        expanded = manager.run_pass(
            "logical_expand", output, lambda value: {"expanded": value}
        )
        self.assertEqual(expanded, {"expanded": output})

    def test_explicit_context_is_hashed_passed_and_immutable(self) -> None:
        manager = PassManager()
        context = {"fabric": {"dies": 2}, "placement": (0, 1)}
        output = manager.run_pass(
            "build_ir0",
            {"spec": 1},
            lambda value, passed_context: {
                "input": value,
                "context": passed_context,
            },
            context=context,
        )
        receipt = manager.snapshot.receipts[0]
        self.assertEqual(receipt.context_digest, canonical_digest(context))
        self.assertEqual(output["context"], context)

        first = PassManager()
        second = PassManager()
        first.run_pass(
            "build_ir0",
            {"spec": 1},
            lambda value, _context: value,
            context={"fabric": 1},
        )
        second.run_pass(
            "build_ir0",
            {"spec": 1},
            lambda value, _context: value,
            context={"fabric": 2},
        )
        self.assertNotEqual(first.snapshot.id, second.snapshot.id)

        mutable_context = {"dies": []}
        rejected = PassManager()
        before = rejected.snapshot

        def mutate_context(
            value: dict[str, int], passed_context: dict[str, list[int]]
        ) -> dict[str, int]:
            passed_context["dies"].append(0)
            return value

        with self.assertRaisesRegex(InputMutationError, "context"):
            rejected.run_pass(
                "build_ir0",
                {"spec": 1},
                mutate_context,
                context=mutable_context,
            )
        self.assertEqual(rejected.snapshot, before)

    def test_placement_cannot_read_untracked_closure_context(self) -> None:
        manager = PassManager()
        ir0 = manager.run_pass(
            "build_ir0", {"spec": 1}, lambda value: {"ir0": value}
        )
        expanded = manager.run_pass(
            "logical_expand", ir0, lambda value: {"expanded": value}
        )
        before = manager.snapshot
        with self.assertRaisesRegex(PassOrderError, "explicit immutable context"):
            manager.run_pass("placement", expanded, lambda value: value)
        self.assertEqual(manager.snapshot, before)

        placed = manager.run_pass(
            "placement",
            expanded,
            lambda value, passed_context: {
                "expanded": value,
                "fabric": passed_context,
            },
            context={"fabric": "tracked"},
        )
        self.assertEqual(placed["fabric"], {"fabric": "tracked"})
        self.assertIsNotNone(manager.snapshot.receipts[-1].context_digest)

    def test_n4_policy_passes_require_separate_tracked_contexts(self) -> None:
        registry = production_registry()
        planning_selections = (
            registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
            registry.instantiate(
                RegistryKind.STANDALONE_COLLECTIVE,
                "direct_all_gather",
            ).selection,
        )
        scheduling_selections = (
            registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection,
        )
        manager = PassManager()
        ir0 = manager.run_pass(
            "build_ir0", {"spec": 1}, lambda value: {"ir0": value}
        )
        expanded = manager.run_pass(
            "logical_expand", ir0, lambda value: {"expanded": value}
        )
        placed = manager.run_pass(
            "placement",
            expanded,
            lambda value, context: {"source": value, "placement": context},
            context={"placement": "compact"},
        )

        before_partition = manager.snapshot
        with self.assertRaisesRegex(PassOrderError, "explicit immutable context"):
            manager.run_pass("fusion_partition", placed, lambda value: value)
        self.assertEqual(manager.snapshot, before_partition)
        partitioned = manager.run_pass(
            "fusion_partition",
            placed,
            lambda value, context: {"source": value, "partition": context},
            context={"partition": "gemm_rs_all/v1"},
        )

        before_planning = manager.snapshot
        with self.assertRaisesRegex(PassOrderError, "explicit immutable context"):
            manager.run_pass("inter_die_plan", partitioned, lambda value: value)
        self.assertEqual(manager.snapshot, before_planning)
        for invalid_selections in ((), tuple(reversed(planning_selections))):
            with self.subTest(invalid_selections=invalid_selections):
                with self.assertRaisesRegex(PassOrderError, "exact kind order"):
                    manager.run_pass(
                        "inter_die_plan",
                        partitioned,
                        lambda value, context: value,
                        context={"inter_die": "direct_naive/v1"},
                        policy_selections=invalid_selections,
                    )
                self.assertEqual(manager.snapshot, before_planning)
        planned = manager.run_pass(
            "inter_die_plan",
            partitioned,
            lambda value, context: {"source": value, "planning": context},
            context={"inter_die": "direct_naive/v1"},
            policy_selections=planning_selections,
        )
        before_projection = manager.snapshot
        with self.assertRaisesRegex(PassOrderError, "explicit immutable context"):
            manager.run_pass("project_to_ir2", planned, lambda value: value)
        self.assertEqual(manager.snapshot, before_projection)
        projected = manager.run_pass(
            "project_to_ir2",
            planned,
            lambda value, context: {"source": value, "projection": context},
            context={"projection": "exact_naive/v1"},
        )
        before_schedule = manager.snapshot
        with self.assertRaisesRegex(PassOrderError, "explicit immutable context"):
            manager.run_pass("intra_die_schedule", projected, lambda value: value)
        self.assertEqual(manager.snapshot, before_schedule)
        manager.run_pass(
            "intra_die_schedule",
            projected,
            lambda value, context: {"source": value, "schedule": context},
            context={"schedule": "component_rr_xy/v1"},
            policy_selections=scheduling_selections,
        )
        partition_receipt, planning_receipt, projection_receipt, schedule_receipt = (
            manager.snapshot.receipts[-4:]
        )
        self.assertEqual(
            partition_receipt.context_digest,
            canonical_digest({"partition": "gemm_rs_all/v1"}),
        )
        self.assertEqual(
            planning_receipt.context_digest,
            canonical_digest({"inter_die": "direct_naive/v1"}),
        )
        self.assertEqual(
            projection_receipt.context_digest,
            canonical_digest({"projection": "exact_naive/v1"}),
        )
        self.assertEqual(
            schedule_receipt.context_digest,
            canonical_digest({"schedule": "component_rr_xy/v1"}),
        )
        self.assertEqual(
            planning_receipt.policy_selections,
            planning_selections,
        )
        self.assertEqual(
            schedule_receipt.policy_selections,
            scheduling_selections,
        )
        for receipt in (
            partition_receipt,
            planning_receipt,
            projection_receipt,
            schedule_receipt,
        ):
            with self.subTest(pass_name=receipt.pass_name):
                with self.assertRaisesRegex(
                    SchemaError,
                    "requires a recorded context digest",
                ):
                    replace(receipt, context_digest=None).validate()

    def test_reentrant_run_invalidates_the_outer_transaction(self) -> None:
        for catches_inner_error in (False, True):
            with self.subTest(catches_inner_error=catches_inner_error):
                manager = PassManager()
                before = manager.snapshot

                def transform(value: dict[str, int]) -> dict[str, int]:
                    try:
                        manager.run_pass("build_ir0", value, lambda item: item)
                    except PassOrderError:
                        if not catches_inner_error:
                            raise
                    return {"output": value["input"]}

                with self.assertRaisesRegex(PassOrderError, "reentrant"):
                    manager.run_pass("build_ir0", {"input": 1}, transform)
                self.assertEqual(manager.snapshot, before)

    def test_transform_validation_and_digest_failures_roll_back(self) -> None:
        def fail_transform(_value: object) -> object:
            raise RuntimeError("transform failed")

        cases = (
            (fail_transform, RuntimeError),
            (lambda _value: ValidatedOutput(1, False), SchemaError),
            (lambda _value: {"unsupported": object()}, SchemaError),
        )
        for transform, error_type in cases:
            with self.subTest(error_type=error_type.__name__):
                manager = PassManager()
                before = manager.snapshot
                with self.assertRaises(error_type):
                    manager.run_pass("build_ir0", {"spec": 1}, transform)
                self.assertEqual(manager.snapshot, before)

        manager = PassManager()
        accepted = manager.run_pass(
            "build_ir0",
            {"spec": 1},
            lambda _value: ValidatedOutput(1, True),
        )
        self.assertTrue(accepted.accepted)

    def test_receipt_and_snapshot_tampering_is_rejected(self) -> None:
        manager = PassManager()
        first_output = manager.run_pass(
            "build_ir0", {"spec": 1}, lambda value: {"ir0": value}
        )
        manager.run_pass(
            "logical_expand",
            first_output,
            lambda value: {"expanded": value},
        )
        snapshot = manager.snapshot
        snapshot.validate()
        first, second = snapshot.receipts

        tampered = (
            replace(snapshot, schema_version="wrong"),
            replace(snapshot, producer_pass="wrong"),
            replace(snapshot, id="pipeline_wrong"),
            replace(snapshot, phase=PipelinePhase.IR1_PLACED),
            replace(
                snapshot,
                receipts=(
                    replace(first, output_phase=PipelinePhase.LOGICAL_EXPANDED),
                    second,
                ),
            ),
            replace(
                snapshot,
                receipts=(first, replace(second, input_digest="0" * 64)),
            ),
            replace(
                snapshot,
                receipts=(first, replace(second, context_digest="bad")),
            ),
        )
        for changed in tampered:
            with self.subTest(changed=changed):
                with self.assertRaises(SchemaError):
                    changed.validate()

        with self.assertRaisesRegex(SchemaError, "unknown pass"):
            replace(first, pass_name="unknown").validate()
        with self.assertRaisesRegex(SchemaError, "SHA-256"):
            PassReceipt(
                first.pass_name,
                first.input_phase,
                first.output_phase,
                "not-a-digest",
                None,
                first.output_digest,
            ).validate()


if __name__ == "__main__":
    unittest.main()
