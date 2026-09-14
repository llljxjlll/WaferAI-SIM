from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_training_compile_sequence import (
    compile_dense_training_sequence,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.dense_training_compile_sequence import (
    DenseTrainingCompileSegment,
    DenseTrainingCompileSequence,
    DenseTrainingParameterBinding,
    DenseTrainingRuntimeStatus,
    DenseTrainingSyncLowering,
)
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryTier,
    MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily,
    WorkloadMeshSpec,
    WorkloadModelArchitecture,
    WorkloadModelSpec,
    WorkloadOptimizerKind,
    WorkloadOptimizerSpec,
    WorkloadParallelSpec,
    WorkloadRunRequest,
    WorkloadStepSpec,
    WorkloadTrainingSteps,
)
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec
from llm.test.frontend.unit.test_workload_materialization import _capability


@lru_cache(maxsize=2)
def _sequence(rows: int, columns: int) -> DenseTrainingCompileSequence:
    raw = _hardware(rows, columns)
    fabric = physical_fabric_from_data(raw)
    spaces = hbm_address_spaces_from_data(raw)
    request = WorkloadRunRequest.create(
        family=WorkloadFamily.DENSE_TRAINING,
        model=WorkloadModelSpec(
            architecture=WorkloadModelArchitecture.LLAMA_DENSE,
            vocabulary_size=8 * columns,
            hidden_size=4 * columns,
            intermediate_size=8 * columns,
            num_layers=2,
            num_attention_heads=columns,
            num_kv_heads=columns,
            head_dim=4,
            max_sequence_length=64,
            dtype=DType.FP16,
        ),
        steps=WorkloadStepSpec(
            training=WorkloadTrainingSteps(
                step_count=2,
                global_batch_size=rows,
                micro_batch_size=1,
                micro_batch_count=1,
                sequence_length=columns,
            )
        ),
        mesh=WorkloadMeshSpec(rows, columns),
        parallel=WorkloadParallelSpec(
            tp=columns,
            dp=rows,
            active_die_ids=tuple(range(rows * columns)),
        ),
        optimizer=WorkloadOptimizerSpec(
            WorkloadOptimizerKind.SGD,
            0.001,
        ),
    )
    capacities = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{space.die_id}",
            base_address=space.base_address,
            capacity_bytes=space.size_bytes,
            alignment_bytes=space.alignment_bytes,
        )
        for space in spaces
    )
    manifest = materialize_workload_preflight(
        request,
        _capability(supported=True),
        capacities=capacities,
    )
    return compile_dense_training_sequence(
        manifest,
        _spec(rows, columns),
        fabric,
        hbm_address_spaces=spaces,
    )


def _replace_binding(
    sequence: DenseTrainingCompileSequence,
    step: int,
    parameter_ref: str,
    **changes: object,
) -> tuple[DenseTrainingCompileSegment, ...]:
    segment = sequence.segments[step]
    bindings = list(segment.parameter_bindings)
    index = next(
        index
        for index, binding in enumerate(bindings)
        if binding.parameter_ref == parameter_ref
    )
    semantic = bindings[index]._semantic()
    semantic.update(changes)
    bindings[index] = DenseTrainingParameterBinding.create(**semantic)
    segment_semantic = segment._semantic()
    segment_semantic["parameter_bindings"] = tuple(bindings)
    replacement = DenseTrainingCompileSegment.create(**segment_semantic)
    segments = list(sequence.segments)
    segments[step] = replacement
    return tuple(segments)


class DenseTrainingCompileSequenceTest(unittest.TestCase):
    def test_one_by_one_builds_trainable_state_program_io(self) -> None:
        linked = _sequence(1, 1).segments[0].linked_program
        seeds, expected = build_deterministic_timing_state_overrides(linked)
        self.assertEqual(len(seeds), 15)
        self.assertFalse(expected)
        contract = build_timing_program_io(
            linked,
            "0" * 64,
            state_seed_overrides=seeds,
        )
        self.assertEqual(len(contract.initializations), 45)
        self.assertEqual(len(contract.output_probes), 15)

    def test_one_by_one_real_linked_program_covers_all_parameters(self) -> None:
        sequence = _sequence(1, 1)
        self.assertEqual(len(sequence.segments), 2)
        self.assertEqual(
            tuple(len(item.parameter_bindings) for item in sequence.segments),
            (17, 17),
        )
        linked = sequence.segments[0].linked_program
        self.assertTrue(linked.manifest.fragments)
        self.assertTrue(linked.manifest.core_streams)
        self.assertGreater(linked.record_count, 0)
        self.assertFalse(linked.runtime_verified)
        self.assertIs(
            sequence.runtime_status,
            DenseTrainingRuntimeStatus.RUNTIME_NOT_MATERIALIZED,
        )
        self.assertTrue(all(
            shard.sync_lowering is DenseTrainingSyncLowering.SINGLETON_NOOP
            and not shard.sync_action_refs
            for binding in sequence.segments[0].parameter_bindings
            for shard in binding.legacy_shards
        ))

    def test_parameter_versions_and_fused_gate_up_slices_are_continuous(self) -> None:
        sequence = _sequence(1, 1)
        for parameter in (
            binding.parameter_ref
            for binding in sequence.segments[0].parameter_bindings
        ):
            first = next(
                item for item in sequence.segments[0].parameter_bindings
                if item.parameter_ref == parameter
            )
            second = next(
                item for item in sequence.segments[1].parameter_bindings
                if item.parameter_ref == parameter
            )
            self.assertEqual(
                (
                    first.input_parameter_version,
                    first.output_parameter_version,
                    second.input_parameter_version,
                    second.output_parameter_version,
                ),
                (0, 1, 1, 2),
            )
            self.assertEqual(
                first.output_parameter_state_ref,
                second.input_parameter_state_ref,
            )
            self.assertEqual(
                tuple(item.hbm_addresses for item in first.legacy_shards),
                tuple(item.hbm_addresses for item in second.legacy_shards),
            )
        gate = next(
            item for item in sequence.segments[0].parameter_bindings
            if item.parameter_ref == "layer.0.mlp_gate.weight"
        ).legacy_shards[0]
        up = next(
            item for item in sequence.segments[0].parameter_bindings
            if item.parameter_ref == "layer.0.mlp_up.weight"
        ).legacy_shards[0]
        self.assertEqual(gate.legacy_state_ref, up.legacy_state_ref)
        self.assertEqual(gate.legacy_byte_offset, 0)
        self.assertEqual(up.legacy_byte_offset, gate.logical_bytes)

    def test_two_by_two_binds_real_dp_tree_and_tp_shards(self) -> None:
        sequence = _sequence(2, 2)
        linked = sequence.segments[0].linked_program
        self.assertEqual(linked.plan.spec.tp_degree, 2)
        self.assertEqual(linked.plan.spec.dp_degree, 2)
        self.assertGreater(linked.record_count, 0)
        for binding in sequence.segments[0].parameter_bindings:
            self.assertEqual(len(binding.legacy_shards), 2)
            for shard in binding.legacy_shards:
                self.assertEqual(len(shard.owner_ranks), 2)
                self.assertIs(
                    shard.sync_lowering,
                    DenseTrainingSyncLowering.DP_TREE,
                )
                self.assertTrue(shard.sync_action_refs)

    def test_missing_wgrad_binding_fails_closed(self) -> None:
        sequence = _sequence(1, 1)
        victim = sequence.segments[0].parameter_bindings[0]
        segments = _replace_binding(
            sequence,
            0,
            victim.parameter_ref,
            wgrad_operation_ref=victim.load_operation_ref,
        )
        with self.assertRaisesRegex(SchemaError, "operation binding drifted"):
            DenseTrainingCompileSequence.create(
                materialization=sequence.materialization,
                segments=segments,
            )

    def test_missing_update_binding_fails_closed(self) -> None:
        sequence = _sequence(1, 1)
        victim = sequence.segments[0].parameter_bindings[0]
        segments = _replace_binding(
            sequence,
            0,
            victim.parameter_ref,
            sgd_operation_ref=victim.store_operation_ref,
        )
        with self.assertRaisesRegex(SchemaError, "operation binding drifted"):
            DenseTrainingCompileSequence.create(
                materialization=sequence.materialization,
                segments=segments,
            )

    def test_old_parameter_version_fails_closed(self) -> None:
        sequence = _sequence(1, 1)
        first = sequence.segments[0].parameter_bindings[0]
        second = sequence.segments[1].parameter_bindings[0]
        segments = _replace_binding(
            sequence,
            1,
            second.parameter_ref,
            input_parameter_state_ref=first.input_parameter_state_ref,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "state lineage drifted|version binding drifted",
        ):
            DenseTrainingCompileSequence.create(
                materialization=sequence.materialization,
                segments=segments,
            )

    def test_runtime_overclaim_is_rejected(self) -> None:
        sequence = _sequence(1, 1)
        with self.assertRaisesRegex(SchemaError, "overclaims"):
            replace(sequence, runtime_status="complete").validate()


if __name__ == "__main__":
    unittest.main()
