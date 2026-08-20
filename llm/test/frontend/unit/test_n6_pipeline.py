from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend import compile_naive
from llm.frontend.wafer_frontend.passes import (
    PipelinePhase,
    load_physical_fabric_and_hbm_address_spaces,
)
from llm.frontend.wafer_frontend.passes.pass_manager import PipelineSnapshot
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir1 import PhysicalFabric, SramAllocator
from llm.frontend.wafer_frontend.schema.n5 import (
    GlobalActionBundle,
    ScheduledIR2Bundle,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    LinkedProgramBundle,
    LoweredProgramBundle,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, from_data
from llm.frontend.wafer_frontend.schema.persistent_state import HbmAddressSpace

from _fixtures import valid_spec


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_E1_SRAM_BYTES = 65536


def _e1_spec() -> ExperimentSpec:
    raw = valid_spec()
    model = raw["model"]
    assert isinstance(model, dict)
    model.update(
        {
            "V": 128,
            "H": 32,
            "I": 64,
            "NH": 4,
            "KVH": 2,
            "DH": 8,
            "rotary_dim": 8,
            "L": 1,
        }
    )
    workload = raw["workload"]
    assert isinstance(workload, dict)
    infer = workload["infer"]
    assert isinstance(infer, dict)
    profile = infer["profile"]
    assert isinstance(profile, dict)
    profile.update(
        {
            "prefill_tokens": 4,
            "context_sum": 4,
            "context_max": 4,
        }
    )
    return from_data(ExperimentSpec, raw, path="spec")


def _e1_compile_inputs() -> tuple[
    PhysicalFabric, tuple[HbmAddressSpace, ...]
]:
    fabric, hbm_address_spaces = load_physical_fabric_and_hbm_address_spaces(
        _HARDWARE, _MAPPING
    )
    profiles = []
    for profile in fabric.sram_profiles:
        comm = next(region for region in profile.regions if region.name == "comm")
        profiles.append(
            replace(
                profile,
                capacity_bytes=_E1_SRAM_BYTES,
                regions=(
                    replace(
                        comm,
                        base_bytes=0,
                        size_bytes=_E1_SRAM_BYTES,
                        allocator=SramAllocator.BLOCK,
                    ),
                ),
            )
        )
    result = replace(fabric, sram_profiles=tuple(profiles))
    result.validate("e1_fabric")
    return result, hbm_address_spaces


def _e1_fabric() -> PhysicalFabric:
    return _e1_compile_inputs()[0]


@dataclass(frozen=True, slots=True)
class _E1PipelineRun:
    snapshot: PipelineSnapshot
    spec: ExperimentSpec
    fabric: PhysicalFabric
    contexts: tuple[object, ...]
    context_digests: tuple[str, ...]
    stage_inputs: tuple[object, ...]
    input_digests: tuple[str, ...]
    scheduled: ScheduledIR2Bundle
    global_bundle: GlobalActionBundle
    lowered: LoweredProgramBundle
    linked: LinkedProgramBundle


def _compile_through_n6() -> _E1PipelineRun:
    spec = _e1_spec()
    fabric, hbm_address_spaces = _e1_compile_inputs()
    compilation = compile_naive(
        spec,
        fabric,
        hbm_address_spaces=hbm_address_spaces,
        producer_pass="n6_pipeline_fixture",
    )
    contexts = compilation.contexts
    context_digests = tuple(canonical_digest(context) for context in contexts)
    stage_inputs = compilation.artifacts[:-1]
    input_digests = tuple(canonical_digest(value) for value in stage_inputs)
    scheduled = compilation.artifacts[7]
    global_bundle = compilation.artifacts[8]
    lowered = compilation.artifacts[9]
    linked = compilation.linked
    assert isinstance(scheduled, ScheduledIR2Bundle)
    assert isinstance(global_bundle, GlobalActionBundle)
    assert isinstance(lowered, LoweredProgramBundle)
    assert isinstance(linked, LinkedProgramBundle)
    return _E1PipelineRun(
        snapshot=compilation.snapshot,
        spec=spec,
        fabric=fabric,
        contexts=contexts,
        context_digests=context_digests,
        stage_inputs=stage_inputs,
        input_digests=input_digests,
        scheduled=scheduled,
        global_bundle=global_bundle,
        lowered=lowered,
        linked=linked,
    )


class N6PipelineTest(unittest.TestCase):
    # The unmodified hardware_numa 4096-byte comm-region rejection remains
    # covered by N5PipelineTest.test_real_hardware_comm_capacity_failure_rolls_back_schedule_pass.
    def test_e1_real_tp2_pipeline_is_exact_reproducible_and_immutable(self) -> None:
        first = _compile_through_n6()
        self.assertEqual(
            (
                first.spec.model.H,
                first.spec.model.I,
                first.spec.model.V,
                first.spec.model.NH,
                first.spec.model.KVH,
                first.spec.model.DH,
                first.spec.model.L,
            ),
            (32, 64, 128, 4, 2, 8, 1),
        )
        profile = first.spec.workload.infer.profile
        assert profile is not None
        self.assertEqual(
            (
                profile.prefill_tokens,
                profile.context_sum,
                profile.context_max,
            ),
            (4, 4, 4),
        )
        instance = first.spec.parallel.instances[0]
        self.assertEqual((instance.tp, instance.sp), (2, True))

        snapshot = first.snapshot
        self.assertEqual(snapshot.phase, PipelinePhase.MANIFEST_LINKED)
        self.assertEqual(
            tuple(receipt.pass_name for receipt in snapshot.receipts),
            (
                "build_ir0",
                "logical_expand",
                "placement",
                "fusion_partition",
                "inter_die_plan",
                "project_to_ir2",
                "intra_die_schedule",
                "global_action_dag",
                "lowering",
                "manifest_link",
            ),
        )
        self.assertEqual(
            tuple(receipt.input_digest for receipt in snapshot.receipts),
            first.input_digests,
        )
        self.assertEqual(
            tuple(receipt.context_digest for receipt in snapshot.receipts),
            (None, None, *first.context_digests, None, None, None),
        )
        self.assertTrue(
            all(
                previous.output_digest == current.input_digest
                for previous, current in zip(
                    snapshot.receipts,
                    snapshot.receipts[1:],
                )
            )
        )
        self.assertEqual(
            snapshot.receipts[-1].output_digest,
            canonical_digest(first.linked),
        )
        self.assertEqual(
            tuple(canonical_digest(value) for value in first.stage_inputs),
            first.input_digests,
        )
        self.assertEqual(
            tuple(canonical_digest(context) for context in first.contexts),
            first.context_digests,
        )

        for profile in first.fabric.sram_profiles:
            self.assertEqual(profile.capacity_bytes, _E1_SRAM_BYTES)
            self.assertEqual(len(profile.regions), 1)
            region = profile.regions[0]
            self.assertEqual(
                (
                    region.name,
                    region.base_bytes,
                    region.size_bytes,
                    region.allocator,
                ),
                ("comm", 0, _E1_SRAM_BYTES, SramAllocator.BLOCK),
            )

        scheduled_entry = first.scheduled.entries[0]
        self.assertEqual(
            tuple(
                len(schedule.buffer_bindings)
                for schedule in scheduled_entry.schedule_set.schedules
            ),
            (38, 38),
        )
        self.assertEqual(
            tuple(
                len(schedule.task_state_uses)
                for schedule in scheduled_entry.schedule_set.schedules
            ),
            (11, 11),
        )
        self.assertEqual(
            max(
                binding.region_offset_bytes
                for schedule in scheduled_entry.schedule_set.schedules
                for binding in schedule.buffer_bindings
            ),
            30912,
        )
        self.assertEqual(
            max(
                binding.region_offset_bytes + binding.size_bytes
                for schedule in scheduled_entry.schedule_set.schedules
                for binding in schedule.buffer_bindings
            ),
            30976,
        )
        self.assertEqual(
            len(first.global_bundle.entries[0].global_dag.actions),
            86,
        )
        lowered_entry = first.lowered.entries[0]
        manifest = first.linked.entries[0].manifest
        self.assertEqual(len(lowered_entry.fragments), 52)
        self.assertEqual(len(manifest.fragments), 52)
        self.assertEqual(len(manifest.core_streams), 2)
        self.assertEqual(len(manifest.state_operand_bindings), 22)
        self.assertEqual(
            sum(len(stream.records) for stream in manifest.core_streams),
            278,
        )

        records = tuple(
            record
            for linked in manifest.fragments
            for stream in (
                linked.fragment.core_streams
                if isinstance(linked, RegionManifest)
                else linked.core_streams
            )
            for record in stream.records
        )
        self.assertEqual(
            Counter(record.opcode for record in records),
            Counter(
                {
                    RecordOpcode.SRAM_ALLOC_AT: 76,
                    RecordOpcode.SRAM_FREE: 76,
                    RecordOpcode.SRAM_BIND: 32,
                    RecordOpcode.MATMUL: 14,
                    RecordOpcode.ATTENTION_EXACT: 2,
                    RecordOpcode.EMBEDDING_LOOKUP: 2,
                    RecordOpcode.ROPE_QK_EXACT: 2,
                    RecordOpcode.RMSNORM: 6,
                    RecordOpcode.SWIGLU: 2,
                    RecordOpcode.RESIDUAL: 4,
                    RecordOpcode.DTE_SEND: 8,
                    RecordOpcode.DTE_RECV: 8,
                    RecordOpcode.DTE_WAIT: 8,
                    RecordOpcode.LOCAL_REDUCE: 4,
                    RecordOpcode.DTE_ISSUE: 4,
                    RecordOpcode.EVENT_SET: 4,
                    RecordOpcode.EVENT_WAIT: 4,
                    RecordOpcode.LSU_LOAD: 18,
                    RecordOpcode.LSU_STORE: 4,
                }
            ),
        )

        second = _compile_through_n6()
        self.assertEqual(second.snapshot, snapshot)
        self.assertEqual(second.linked, first.linked)
        self.assertEqual(
            canonical_digest(second.linked),
            canonical_digest(first.linked),
        )


if __name__ == "__main__":
    unittest.main()
