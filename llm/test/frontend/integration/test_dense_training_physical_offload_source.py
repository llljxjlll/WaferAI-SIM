"""Physical StateABI sources must drive signed HBM pinned DMA transfers."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_training_physical_offload_program import (
    build_dense_training_physical_offload_program,
)
from llm.frontend.wafer_frontend.passes.dense_training_physical_offload_source import (
    build_dense_training_physical_offload_source,
)
from llm.frontend.wafer_frontend.passes.offload import plan_offload_blocking
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from .run_dense_external_offload_runtime_canary import _build_offload_case
from .run_dense_training_sequence_runtime_canary import _offload_sequence


class DenseTrainingPhysicalOffloadSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sequence = _offload_sequence()
        cls.sequence.validate()
        cls.source = build_dense_training_physical_offload_source(
            cls.sequence, hbm_capacity_bytes=34880,
            external_capacity_bytes=32768,
        )
        cls.seeds, expected = build_deterministic_timing_state_overrides(
            cls.sequence.segments[0].linked_program,
        )
        if expected:
            raise AssertionError("physical state seeds have unresolved writes")

    def test_15_linked_states_become_individual_validated_source_allocations(self) -> None:
        manifest = self.source.external_manifest
        manifest.validate()
        source_allocs = {item.id for item in manifest.memory_plan.allocations}
        self.assertEqual(len(self.source.declarations), 15)
        self.assertEqual(len({item.inventory_ref for item in self.source.declarations}), 15)
        self.assertEqual(len({item.source_version_ref for item in self.source.declarations}), 15)
        self.assertEqual(len({item.external_allocation_ref for item in self.source.declarations}), 15)
        self.assertTrue(all(item.external_allocation_ref in source_allocs
                            for item in self.source.declarations))
        self.assertEqual(sum(item.logical_bytes for item in self.source.declarations), 17344)
        self.assertIn("memory_capacity_exceeded", self.source.resident_rejection)

    def test_real_planner_emits_exact_linked_address_per_state_both_phases(self) -> None:
        case = build_dense_training_physical_offload_program(self.source, self.seeds)
        self.assertEqual(len(case.program.descriptors), 30)
        self.assertEqual(len(case.program.external_seeds), 15)
        self.assertEqual(len(case.program.external_probes), 15)
        self.assertEqual(case.plan.stats.bring_in_count, 15)
        self.assertEqual(case.plan.stats.dirty_writeback_count, 15)
        first = {(item.external_address, item.hbm_address, item.size_bytes)
                 for item in case.program.descriptors if item.direction.value == "external_to_hbm"}
        second = {(item.external_address, item.hbm_address, item.size_bytes)
                  for item in case.program.descriptors if item.direction.value == "hbm_to_external"}
        self.assertEqual(first, second)
        self.assertEqual({item.hbm_address for item in case.program.descriptors},
                         {item.hbm_address for item in self.source.declarations})

    def test_missing_or_forged_external_seed_rejected_before_program(self) -> None:
        incomplete = dict(self.seeds)
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(SchemaError):
            build_dense_training_physical_offload_program(self.source, incomplete)
        wrong = dict(self.seeds)
        key = next(iter(wrong))
        wrong[key] = wrong[key][:-1]
        with self.assertRaises(SchemaError):
            build_dense_training_physical_offload_program(self.source, wrong)

    def test_original_unpinned_plan_is_unchanged_and_pinned_map_failfast(self) -> None:
        base, existing, _, _, _, _, _, _, _, _ = _build_offload_case(
            self.sequence.materialization,
            hbm_bytes_override=34880, dirty_writeback=True,
        )
        self.assertTrue(base.request.case_id)
        args = dict(
            request_digest=existing.request_digest,
            logical_graph_digest=existing.logical_graph_digest,
            source_memory_plan_digest=existing.source_memory_plan_digest,
            source_memory_plan=existing.source_memory_plan,
            state_mappings=existing.state_mappings,
            fabric=existing.fabric,
            chunks=existing.chunks, events=existing.events,
        )
        self.assertEqual(canonical_digest(plan_offload_blocking(**args)),
                         canonical_digest(existing))
        chunk = existing.chunks[0]
        pin = {chunk.id: 64}
        shifted = plan_offload_blocking(**args, pinned_hbm_addresses=pin)
        self.assertTrue(all(item.hbm_address == 64
                            for item in shifted.transfer_requests))
        for bad in ({}, {chunk.id: 1}, {chunk.id: 34880}):
            with self.subTest(bad=bad), self.assertRaises(SchemaError) as caught:
                plan_offload_blocking(**args, pinned_hbm_addresses=bad)
            self.assertEqual(caught.exception.code,
                             "offload_pinned_hbm_address_mismatch")


if __name__ == "__main__":
    unittest.main()
