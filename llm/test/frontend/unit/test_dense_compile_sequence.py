from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    compile_dense_e2e_sequence,
)
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.dense_compile_sequence import (
    DenseCompileRuntimeStatus,
    DenseCompileSegment,
    DenseCompileSequence,
    DenseKvSegmentBinding,
)
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadInferenceSteps,
    WorkloadMeshSpec,
    WorkloadParallelSpec,
    WorkloadRunRequest,
    WorkloadStepSpec,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_legacy_dense_backend import (
    _capability,
    _legacy_spec,
    _request,
)


def _one_die_case():
    base = _request(layers=2, prefill=4, decode=2)
    request = WorkloadRunRequest.create(
        family=base.family,
        model=base.model,
        steps=base.steps,
        mesh=WorkloadMeshSpec(1, 1),
        parallel=WorkloadParallelSpec(tp=1),
        memory=base.memory,
        execution=base.execution,
    )
    capacity = (
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref="die:0",
            base_address=0,
            capacity_bytes=1 << 30,
            alignment_bytes=64,
        ),
    )
    manifest = materialize_workload_preflight(
        request,
        _capability(),
        capacities=capacity,
    )
    template = _legacy_spec(layers=2, prefill=4, decode=0)
    template = replace(
        template,
        parallel=replace(
            template.parallel,
            instances=(
                replace(template.parallel.instances[0], tp=1, sp=False),
            ),
        ),
    )
    template.validate()
    fabric = physical_fabric_from_data(
        minimal_hardware(1, 1, sram_bytes=65536)
    )
    return manifest, template, fabric


def _two_by_two_case():
    base = _request(layers=2, prefill=4, decode=2)
    model = replace(base.model, num_kv_heads=4)
    steps = WorkloadStepSpec(
        inference=WorkloadInferenceSteps(
            prefill_tokens=4,
            decode_steps=2,
            request_count=4,
        )
    )
    request = WorkloadRunRequest.create(
        family=base.family,
        model=model,
        steps=steps,
        mesh=WorkloadMeshSpec(2, 2),
        parallel=WorkloadParallelSpec(
            tp=4,
            active_die_ids=(0, 1, 2, 3),
        ),
        memory=base.memory,
        execution=base.execution,
    )
    capacity = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{rank}",
            base_address=0,
            capacity_bytes=1 << 30,
            alignment_bytes=64,
        )
        for rank in range(4)
    )
    manifest = materialize_workload_preflight(
        request,
        _capability(),
        capacities=capacity,
    )
    template = _legacy_spec(layers=2, prefill=4, decode=0)
    template = replace(
        template,
        model=replace(template.model, KVH=4),
        parallel=replace(
            template.parallel,
            instances=(
                replace(template.parallel.instances[0], tp=4, sp=True),
            ),
        ),
    )
    template.validate()
    fabric = physical_fabric_from_data(
        minimal_hardware(2, 2, sram_bytes=65536)
    )
    return manifest, template, fabric


def _replace_segment(
    sequence: DenseCompileSequence,
    index: int,
    **changes: object,
) -> tuple[DenseCompileSegment, ...]:
    original = sequence.segments[index]
    semantic = original._semantic()
    semantic.update(changes)
    replacement = DenseCompileSegment.create(**semantic)
    values = list(sequence.segments)
    values[index] = replacement
    return tuple(values)


class DenseCompileSequenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.template, cls.fabric = _one_die_case()
        cls.sequence = compile_dense_e2e_sequence(
            cls.manifest,
            cls.template,
            cls.fabric,
            hbm_address_spaces=valid_hbm_address_spaces(cls.fabric),
        )

    def test_real_three_segment_compiler_canary(self) -> None:
        sequence = self.sequence
        self.assertEqual(len(sequence.segments), 3)
        self.assertEqual(
            tuple((item.phase, item.step) for item in sequence.segments),
            (("prefill", 0), ("decode", 1), ("decode", 2)),
        )
        self.assertEqual(
            tuple(item.one_shot_workload_end for item in sequence.segments),
            (False, False, True),
        )
        self.assertIs(
            sequence.runtime_status,
            DenseCompileRuntimeStatus.RUNTIME_NOT_MATERIALIZED,
        )
        self.assertEqual(
            tuple(item.layer_bindings for item in sequence.segments),
            ((0, 1), (0, 1), (0, 1)),
        )
        self.assertTrue(
            all(
                item.linked_manifest.fragments and item.linked_manifest.core_streams
                for item in sequence.segments
            )
        )
        self.assertEqual(
            len({item.linked_manifest_digest for item in sequence.segments}),
            3,
        )
        physical_layouts = []
        physical_extents = []
        for segment in sequence.segments:
            abis = {
                abi.id: abi
                for fragment in segment.linked_manifest.fragments
                for abi in fragment.state_abi
                if abi.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
            }
            ordered = tuple(
                sorted(
                    abis.values(),
                    key=lambda abi: (abi.kind.value, abi.die_id, abi.address),
                )
            )
            physical_layouts.append(
                tuple((abi.kind, abi.die_id, abi.address) for abi in ordered)
            )
            physical_extents.append(tuple(abi.size_bytes for abi in ordered))
        self.assertEqual(len(set(physical_layouts)), 1)
        self.assertEqual(
            physical_extents,
            [(128,) * 4, (160,) * 4, (192,) * 4],
        )
        for layer in range(2):
            chain = tuple(
                next(item for item in segment.kv_bindings if item.layer == layer)
                for segment in sequence.segments
            )
            self.assertEqual(
                tuple(item.input_version for item in chain),
                (0, 1, 2),
            )
            self.assertEqual(
                tuple(item.output_version for item in chain),
                (1, 2, 3),
            )
            self.assertEqual(len({item.address for item in chain}), 1)
            self.assertEqual(len({item.reserved_bytes for item in chain}), 1)
            self.assertEqual(chain[0].output_state_ref, chain[1].input_state_ref)
            self.assertEqual(chain[1].output_state_ref, chain[2].input_state_ref)
        self.sequence.validate()
        self.assertEqual(self.sequence.digest, self.sequence.digest)

    def test_real_two_by_two_tp4_four_request_compiler_canary(self) -> None:
        manifest, template, fabric = _two_by_two_case()
        sequence = compile_dense_e2e_sequence(
            manifest,
            template,
            fabric,
            hbm_address_spaces=valid_hbm_address_spaces(fabric),
        )
        self.assertEqual(
            tuple(segment.legacy_spec.workload.infer.profile.num_seqs
                  for segment in sequence.segments),
            (4, 4, 4),
        )
        self.assertEqual(
            tuple(segment.legacy_spec.parallel.instances[0].tp
                  for segment in sequence.segments),
            (4, 4, 4),
        )
        self.assertEqual(
            tuple(segment.legacy_spec.parallel.instances[0].sp
                  for segment in sequence.segments),
            (True, True, True),
        )
        self.assertTrue(
            all(
                {binding.logical_rank for binding in segment.kv_bindings}
                == {0, 1, 2, 3}
                for segment in sequence.segments
            )
        )
        self.assertTrue(
            all(
                len(segment.linked_manifest.fragments) > 0
                for segment in sequence.segments
            )
        )
        sequence.validate()

    def test_sequence_strict_json_round_trip(self) -> None:
        decoded = loads_dataclass(
            DenseCompileSequence,
            canonical_json(self.sequence),
            path="sequence",
        )
        self.assertEqual(decoded, self.sequence)
        self.assertEqual(decoded.digest, self.sequence.digest)

    def test_old_schema_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                self.sequence,
                schema_version="wafer_frontend.dense_compile_sequence/v0",
            ).validate()

    def test_reset_kv_state_is_rejected(self) -> None:
        segment = self.sequence.segments[1]
        binding = segment.kv_bindings[0]
        reset = DenseKvSegmentBinding.create(
            **{
                **binding._semantic(),
                "input_state_ref": self.sequence.segments[0].kv_bindings[0].input_state_ref,
            }
        )
        bindings = (reset, *segment.kv_bindings[1:])
        segments = _replace_segment(self.sequence, 1, kv_bindings=bindings)
        with self.assertRaisesRegex(SchemaError, "KV state/version/address binding drifted"):
            DenseCompileSequence.create(
                materialization=self.manifest,
                segments=segments,
            )

    def test_intermediate_one_shot_end_is_rejected(self) -> None:
        segments = _replace_segment(
            self.sequence,
            1,
            one_shot_workload_end=True,
        )
        with self.assertRaisesRegex(SchemaError, "only the final segment"):
            DenseCompileSequence.create(
                materialization=self.manifest,
                segments=segments,
            )

    def test_linked_manifest_digest_drift_is_rejected(self) -> None:
        semantic = self.sequence.segments[2]._semantic()
        semantic["linked_manifest_digest"] = "0" * 64
        with self.assertRaisesRegex(SchemaError, "linked manifest digest mismatch"):
            DenseCompileSegment.create(**semantic)

    def test_tp2_single_request_is_rejected_before_compile(self) -> None:
        base = _request(layers=2, prefill=4, decode=2)
        manifest = materialize_workload_preflight(
            base,
            _capability(),
            capacities=tuple(
                MemoryTierCapacity.create(
                    tier=MemoryTier.HBM,
                    location_ref=f"die:{rank}",
                    base_address=0,
                    capacity_bytes=1 << 30,
                    alignment_bytes=64,
                )
                for rank in range(2)
            ),
        )
        fabric = physical_fabric_from_data(
            minimal_hardware(2, 1, sram_bytes=65536)
        )
        with self.assertRaisesRegex(
            UnsupportedFeatureError,
            "decode_tokens_must_be_divisible_by_tp",
        ):
            compile_dense_e2e_sequence(
                manifest,
                _legacy_spec(layers=2, prefill=4, decode=0),
                fabric,
                hbm_address_spaces=valid_hbm_address_spaces(fabric),
            )

    def test_two_by_two_active_subset_remains_fail_closed(self) -> None:
        base = _request(layers=2, prefill=4, decode=2)
        request = WorkloadRunRequest.create(
            family=base.family,
            model=replace(base.model, num_kv_heads=4),
            steps=WorkloadStepSpec(
                inference=WorkloadInferenceSteps(4, 2, 2)
            ),
            mesh=WorkloadMeshSpec(2, 2),
            parallel=WorkloadParallelSpec(tp=2, active_die_ids=(0, 1)),
            memory=base.memory,
            execution=base.execution,
        )
        capacities = tuple(
            MemoryTierCapacity.create(
                tier=MemoryTier.HBM,
                location_ref=f"die:{rank}",
                base_address=0,
                capacity_bytes=1 << 30,
                alignment_bytes=64,
            )
            for rank in range(4)
        )
        manifest = materialize_workload_preflight(
            request,
            _capability(),
            capacities=capacities,
        )
        template = _legacy_spec(layers=2, prefill=4, decode=0)
        template = replace(
            template,
            model=replace(template.model, KVH=4),
            parallel=replace(
                template.parallel,
                instances=(replace(template.parallel.instances[0], tp=2, sp=True),),
            ),
        )
        fabric = physical_fabric_from_data(
            minimal_hardware(2, 2, sram_bytes=65536)
        )
        with self.assertRaisesRegex(
            UnsupportedFeatureError,
            "full_row_major_mesh.*tp_must_cover_mesh",
        ):
            compile_dense_e2e_sequence(
                manifest,
                template,
                fabric,
                hbm_address_spaces=valid_hbm_address_spaces(fabric),
            )


if __name__ == "__main__":
    unittest.main()
