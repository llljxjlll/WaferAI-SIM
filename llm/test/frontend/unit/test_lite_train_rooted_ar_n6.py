from __future__ import annotations

from dataclasses import replace
import unittest

from llm.test.frontend.unit.test_lite_train_dp2 import _chain

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_train_rooted_ar_lower_program import (
    lower_s2_lite_rooted_ar,
)
from llm.frontend.wafer_frontend.passes.lite_train_rooted_ar_link_program import (
    link_s2_lite_rooted_ar,
)
from llm.frontend.wafer_frontend.passes.lite_train_rooted_ar_n6 import (
    build_s2_lite_rooted_ar_n6_intent,
)
from llm.frontend.wafer_frontend.passes import (
    build_s2_lite_rooted_ar_n6_intent as public_build_rooted_ar_intent,
    link_s2_lite_rooted_ar as public_link_rooted_ar,
    lower_s2_lite_rooted_ar as public_lower_rooted_ar,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import FragmentKind, RecordOpcode
from llm.frontend.wafer_frontend.schema.lite_train_rooted_ar_n6 import (
    RootedArExecutableKind,
    RootedArExecutableUnit,
    S2LiteRootedArLinkedProgram,
    S2LiteRootedArLoweredProgram,
    S2LiteRootedArN6Intent,
)
from llm.frontend.wafer_frontend.schema import (
    RootedArExecutableKind as PublicRootedArExecutableKind,
    RootedArExecutableUnit as PublicRootedArExecutableUnit,
    S2LiteRootedArLinkedProgram as PublicRootedArLinkedProgram,
    S2LiteRootedArLoweredProgram as PublicRootedArLoweredProgram,
    S2LiteRootedArN6Intent as PublicRootedArN6Intent,
)


class S2LiteRootedArN6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _chain()[-1]
        cls.intent = build_s2_lite_rooted_ar_n6_intent(cls.source)
        cls.lowered = lower_s2_lite_rooted_ar(cls.intent)
        cls.linked = link_s2_lite_rooted_ar(cls.lowered)

    def test_exact_scratch_units_and_real_leaf_lowering(self) -> None:
        self.assertEqual(
            tuple((abi.region_offset_bytes, abi.size_bytes) for abi in self.intent.scratch_buffer_abis),
            ((32768, 2048), (34816, 2048)),
        )
        self.assertEqual((len(self.linked.manifest.fragments), len(self.linked.manifest.core_streams)), (98, 2))
        for replica_index, stream in enumerate(self.linked.manifest.core_streams):
            dag = self.source.local_dags[replica_index]
            wgrad = next(action.id for action in dag.actions if ".lm_head_wgrad" in getattr(action.source, "task_id", ""))
            sgd = next(action.id for action in dag.actions if ".sgd_update" in getattr(action.source, "task_id", ""))
            ids = tuple(record.source_global_action_id for record in stream.records)
            overlay = {unit.id for unit in self.intent.units if unit.logical_core == stream.logical_core}
            self.assertLess(max(index for index, item in enumerate(ids) if item == wgrad), min(index for index, item in enumerate(ids) if item in overlay))
            self.assertLess(max(index for index, item in enumerate(ids) if item in overlay), min(index for index, item in enumerate(ids) if item == sgd))
        self.assertEqual(tuple(unit.kind for unit in self.intent.units), tuple(RootedArExecutableKind))
        self.assertEqual((len(self.lowered.local_fragments), len(self.lowered.overlay_fragments)), (92, 6))
        self.assertEqual(sum(len(fragment.claimed_action_ids) for fragment in self.lowered.local_fragments), 92)
        opcodes = tuple(
            record.opcode
            for fragment in self.lowered.overlay_fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(
            {opcode: opcodes.count(opcode) for opcode in (
                RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.DTE_ISSUE, RecordOpcode.DTE_SEND,
                RecordOpcode.DTE_RECV, RecordOpcode.DTE_WAIT,
                RecordOpcode.LOCAL_REDUCE, RecordOpcode.SRAM_FREE,
            )},
            {
                RecordOpcode.SRAM_ALLOC_AT: 2,
                RecordOpcode.DTE_ISSUE: 1,
                RecordOpcode.DTE_SEND: 2,
                RecordOpcode.DTE_RECV: 2,
                RecordOpcode.DTE_WAIT: 3,
                RecordOpcode.LOCAL_REDUCE: 1,
                RecordOpcode.SRAM_FREE: 2,
            },
        )

    def test_public_exports_are_exact(self) -> None:
        self.assertIs(public_build_rooted_ar_intent, build_s2_lite_rooted_ar_n6_intent)
        self.assertIs(public_lower_rooted_ar, lower_s2_lite_rooted_ar)
        self.assertIs(public_link_rooted_ar, link_s2_lite_rooted_ar)
        self.assertIs(PublicRootedArExecutableKind, RootedArExecutableKind)
        self.assertIs(PublicRootedArExecutableUnit, RootedArExecutableUnit)
        self.assertIs(PublicRootedArN6Intent, S2LiteRootedArN6Intent)
        self.assertIs(PublicRootedArLoweredProgram, S2LiteRootedArLoweredProgram)
        self.assertIs(PublicRootedArLinkedProgram, S2LiteRootedArLinkedProgram)

    def test_noncontiguous_or_overlapping_scratch_fails_closed(self) -> None:
        first, second = self.intent.scratch_buffer_abis
        with self.subTest("noncontiguous"):
            with self.assertRaisesRegex(SchemaError, "contiguous rank-major"):
                replace(self.intent, scratch_buffer_abis=(first, replace(second, region_offset_bytes=second.region_offset_bytes + 64))).validate()
        with self.subTest("overlap-schedule"):
            forged_first = replace(first, region_offset_bytes=24128)
            forged_second = replace(second, region_offset_bytes=26176)
            with self.assertRaisesRegex(SchemaError, "overlaps a scheduled buffer"):
                replace(self.intent, scratch_buffer_abis=(forged_first, forged_second)).validate()

    def test_missing_wait_gate_or_overlay_claim_fails_closed(self) -> None:
        units = list(self.intent.units)
        units[4] = replace(units[4], deps=(units[0].id,))
        with self.assertRaisesRegex(SchemaError, "unstable unit id|scratch/dependency"):
            replace(self.intent, units=tuple(units)).validate()
        fragment = self.lowered.overlay_fragments[0]
        with self.assertRaisesRegex(SchemaError, "claimed action ids|all 8 executable units"):
            replace(self.lowered, overlay_fragments=(replace(fragment, claimed_action_ids=()), *self.lowered.overlay_fragments[1:])).validate()
        with self.assertRaisesRegex(SchemaError, "reserved for its exact dedicated producer"):
            replace(fragment, kind=FragmentKind.MOE_TRANSFER).validate()

    def test_restabilized_core_order_tamper_is_rejected(self) -> None:
        from llm.frontend.wafer_frontend.schema.artifact_manifest import LinkedCoreStream, LinkedProgramManifest
        manifest = self.linked.manifest
        stream = manifest.core_streams[0]
        dag = self.source.local_dags[0]
        sgd = next(action.id for action in dag.actions if ".sgd_update" in getattr(action.source, "task_id", ""))
        sgd_index = next(index for index, item in enumerate(stream.records) if item.source_global_action_id == sgd)
        forged_records = (*stream.records[:sgd_index], stream.records[sgd_index], *stream.records[sgd_index + 1:])
        overlay_index = next(index for index, item in enumerate(forged_records) if item.source_global_action_id in {unit.id for unit in self.intent.units})
        forged_records = list(forged_records)
        forged_records[overlay_index], forged_records[sgd_index] = forged_records[sgd_index], forged_records[overlay_index]
        semantic = manifest._semantic_key()
        semantic["core_streams"] = (LinkedCoreStream(stream.logical_core, stream.runtime_core_id, tuple(forged_records)), *manifest.core_streams[1:])
        forged = LinkedProgramManifest.create(producer_pass=manifest.producer_pass, **semantic)
        forged.validate("forged")
        with self.assertRaisesRegex(SchemaError, "exact rooted-AR production link quotient"):
            replace(self.linked, manifest=forged).validate()

    def test_actual_sha_program_io_exact_quotient(self) -> None:
        state_seeds, state_expected = (
            build_deterministic_timing_state_overrides(self.linked)
        )
        program_io = build_timing_program_io(
            self.linked,
            "a" * 64,
            state_seed_overrides=state_seeds,
            state_expected_overrides=state_expected,
        )
        program_io.validate_against(self.linked.manifest)
        self.assertEqual(
            (
                len(state_seeds),
                len(state_expected),
                len(program_io.blobs),
                len(program_io.initializations),
                len(program_io.output_probes),
            ),
            (15, 0, 24, 126, 2),
        )

    def test_actual_sha_program_io_closes_both_replicas(self) -> None:
        seeds, expected = build_deterministic_timing_state_overrides(
            self.linked
        )
        program_io = build_timing_program_io(
            self.linked,
            "a" * 64,
            state_seed_overrides=seeds,
            state_expected_overrides=expected,
        )
        program_io.validate_against(self.linked.manifest)
        self.assertEqual(
            (
                len(seeds),
                len(expected),
                len(program_io.blobs),
                len(program_io.initializations),
                len(program_io.output_probes),
            ),
            (15, 0, 24, 126, 2),
        )


if __name__ == "__main__":
    unittest.main()
