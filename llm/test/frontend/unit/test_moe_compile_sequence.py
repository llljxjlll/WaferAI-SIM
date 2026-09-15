from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.moe_compile_sequence import (
    compile_moe_sequence,
)
from llm.frontend.wafer_frontend.passes.validate_moe_training_backward_handoff import (
    validate_moe_training_backward_handoff,
)
from llm.frontend.wafer_frontend.passes.validate_moe_training_fp32_gradient_producers import (
    validate_moe_training_fp32_gradient_producers,
)
from llm.frontend.wafer_frontend.lowering.flexible_moe_multi_production import (
    expert_dgrad_action_ids, expert_wgrad_action_ids, gate_wgrad_cast_action_id,
)
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.flexible_moe import (
    MoeRectActionKind, MoeRectFlowStage,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryTier,
    MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.moe_compile_sequence import (
    MoeCompileCoverage,
    MoeCompileRuntimeStatus,
    MoeCompileSequence,
    MoeCompileUnit,
    MoeParameterLowering,
)
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily,
    WorkloadInferenceSteps,
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
from llm.test.frontend.unit.test_workload_materialization import _capability


def _request(family: WorkloadFamily, *, tp: int = 1) -> WorkloadRunRequest:
    training = family is WorkloadFamily.MOE_TRAINING
    ep = 2 if tp == 1 else 1
    return WorkloadRunRequest.create(
        family=family,
        model=WorkloadModelSpec(
            architecture=WorkloadModelArchitecture.LLAMA_MOE,
            vocabulary_size=16,
            hidden_size=4,
            intermediate_size=8,
            num_layers=2,
            num_attention_heads=2,
            num_kv_heads=2,
            head_dim=2,
            max_sequence_length=32,
            dtype=DType.FP16,
            num_experts=2,
            experts_per_token=1,
        ),
        steps=(
            WorkloadStepSpec(
                training=WorkloadTrainingSteps(
                    step_count=2,
                    global_batch_size=1,
                    micro_batch_size=1,
                    micro_batch_count=1,
                    sequence_length=4,
                )
            )
            if training
            else WorkloadStepSpec(
                inference=WorkloadInferenceSteps(
                    prefill_tokens=2,
                    decode_steps=2,
                    request_count=2,
                )
            )
        ),
        mesh=WorkloadMeshSpec(1, 2),
        parallel=WorkloadParallelSpec(
            tp=tp,
            ep=ep,
            active_die_ids=(0, 1),
        ),
        optimizer=(
            WorkloadOptimizerSpec(WorkloadOptimizerKind.SGD, 0.001)
            if training
            else None
        ),
    )


def _manifest(family: WorkloadFamily, *, tp: int = 1):
    capacities = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{rank}",
            base_address=0,
            capacity_bytes=1 << 24,
            alignment_bytes=16,
        )
        for rank in range(2)
    )
    return materialize_workload_preflight(
        _request(family, tp=tp),
        _capability(supported=True),
        capacities=capacities,
    )


@lru_cache(maxsize=2)
def _sequence(family: WorkloadFamily) -> MoeCompileSequence:
    return compile_moe_sequence(_manifest(family))


def _replace_unit(
    sequence: MoeCompileSequence,
    index: int,
    **changes: object,
) -> tuple[MoeCompileUnit, ...]:
    units = list(sequence.units)
    semantic = units[index]._semantic()
    semantic.update(changes)
    units[index] = MoeCompileUnit.create(**semantic)
    return tuple(units)


class MoeCompileSequenceTest(unittest.TestCase):
    def test_strict_training_fp32_gradient_projection_cast_and_sgd_faults(self) -> None:
        sequence = compile_moe_sequence(
            _manifest(WorkloadFamily.MOE_TRAINING),
            source_rank_policy="rank0_shared_spine",
        )
        self.assertIs(sequence.runtime_status, MoeCompileRuntimeStatus.RUNTIME_NOT_MATERIALIZED)
        for unit in sequence.units:
            validate_moe_training_fp32_gradient_producers(unit.plan, unit.spec,
                                                           unit.linked_manifest)
        unit = sequence.units[0]
        expert = next(item for item in unit.plan.actions
                      if item.kind is MoeRectActionKind.EXPERT_WGRAD and item.assignment_refs)
        gate = next(item for item in unit.plan.actions
                    if item.kind is MoeRectActionKind.GATE_WGRAD and item.assignment_refs)
        sgd = next(item for item in unit.plan.actions
                   if item.kind is MoeRectActionKind.EXPERT_SGD and item.rank == expert.rank)
        gate_reduce = next(item for item in unit.plan.actions
                           if item.kind is MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE
                           and item.rank == gate.rank)
        gate_sgd = next(item for item in unit.plan.actions
                        if item.kind is MoeRectActionKind.GATE_SGD and item.rank == gate.rank)
        tree_flows = tuple(item for item in unit.plan.flows
                           if item.stage is MoeRectFlowStage.GATE_ALL_REDUCE)
        gate_tree_records = tuple(item.id for item in unit.plan.actions
                                  if item.kind in (MoeRectActionKind.SEND, MoeRectActionKind.RECV)
                                  and item.flow_ref in {flow.id for flow in tree_flows})
        for dropped in (
            expert.id,
            *expert_wgrad_action_ids(unit.plan.id, expert.id),
            gate.id,
            gate_wgrad_cast_action_id(unit.plan.id, gate.id),
            gate_reduce.id,
            gate_sgd.id,
            *gate_tree_records,
            sgd.id,
        ):
            broken = replace(unit.linked_manifest, core_streams=tuple(replace(
                stream, records=tuple(ref for ref in stream.records
                                      if ref.source_global_action_id != dropped)
            ) for stream in unit.linked_manifest.core_streams))
            with self.assertRaisesRegex(SchemaError, "FP32 gradient lacks one physical record"):
                validate_moe_training_fp32_gradient_producers(unit.plan,
                                                                unit.spec, broken)

    def test_strict_training_backward_handoff_is_physical_and_fails_when_cut(self) -> None:
        sequence = compile_moe_sequence(
            _manifest(WorkloadFamily.MOE_TRAINING),
            source_rank_policy="rank0_shared_spine",
        )
        self.assertIs(sequence.coverage, MoeCompileCoverage.MOE_BLOCKS_ONLY)
        self.assertIs(sequence.runtime_status, MoeCompileRuntimeStatus.RUNTIME_NOT_MATERIALIZED)
        for unit in sequence.units:
            validate_moe_training_backward_handoff(unit.plan, unit.linked_manifest)
        unit = sequence.units[0]
        flow = next(item for item in unit.plan.flows
                    if item.stage is MoeRectFlowStage.BACKWARD_GRADIENT)
        recv_id = next(item.id for item in unit.plan.actions
                       if item.kind is MoeRectActionKind.RECV and item.flow_ref == flow.id)
        dgrad_id = next(item.id for item in unit.plan.actions
                        if item.kind is MoeRectActionKind.EXPERT_DGRAD and item.rank == flow.destination_rank)
        dx_flow = next(item for item in unit.plan.flows
                       if item.stage is MoeRectFlowStage.BACKWARD_DX)
        dx_send_id = next(item.id for item in unit.plan.actions
                          if item.kind is MoeRectActionKind.SEND and item.flow_ref == dx_flow.id)
        for dropped in (recv_id, dgrad_id,
                        *expert_dgrad_action_ids(unit.plan.id, dgrad_id),
                        dx_send_id):
            broken = replace(unit.linked_manifest, core_streams=tuple(replace(
                stream, records=tuple(ref for ref in stream.records
                                      if ref.source_global_action_id != dropped)
            ) for stream in unit.linked_manifest.core_streams))
            with self.assertRaisesRegex(SchemaError, "backward lacks one physical record"):
                validate_moe_training_backward_handoff(unit.plan, broken)

    def test_inference_compiles_every_prefill_decode_layer(self) -> None:
        sequence = _sequence(WorkloadFamily.MOE_INFERENCE)
        self.assertEqual(
            tuple((item.phase, item.step, item.layer) for item in sequence.units),
            (
                ("prefill", 0, 0),
                ("prefill", 0, 1),
                ("decode", 1, 0),
                ("decode", 1, 1),
                ("decode", 2, 0),
                ("decode", 2, 1),
            ),
        )
        self.assertIs(sequence.coverage, MoeCompileCoverage.MOE_BLOCKS_ONLY)
        self.assertIs(
            sequence.runtime_status,
            MoeCompileRuntimeStatus.RUNTIME_NOT_MATERIALIZED,
        )
        self.assertTrue(all(not item.parameter_bindings for item in sequence.units))
        for unit in sequence.units:
            self.assertTrue(unit.linked_manifest.fragments)
            self.assertFalse(unit.runtime_verified)
            action_kinds = {action.kind for action in unit.plan.actions}
            self.assertTrue({
                MoeRectActionKind.GATE,
                MoeRectActionKind.PACK,
                MoeRectActionKind.EXPERT_FORWARD,
                MoeRectActionKind.WEIGHTED_COMBINE,
            }.issubset(action_kinds))
            self.assertEqual(
                unit.spec.trace.expert_histogram,
                next(
                    trace.expert_token_counts
                    for trace in sequence.materialization.logical_graph.route_traces
                    if trace.id == unit.route_trace_ref
                ),
            )

    def test_training_compiles_two_steps_and_preserves_versions(self) -> None:
        sequence = _sequence(WorkloadFamily.MOE_TRAINING)
        self.assertEqual(
            tuple((item.step, item.layer) for item in sequence.units),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        )
        self.assertTrue(all(len(item.parameter_bindings) == 3 for item in sequence.units))
        first_layer = {
            binding.parameter_refs: binding
            for binding in sequence.units[0].parameter_bindings
        }
        second_step_layer = {
            binding.parameter_refs: binding
            for binding in sequence.units[2].parameter_bindings
        }
        self.assertEqual(set(first_layer), set(second_step_layer))
        for key, first in first_layer.items():
            second = second_step_layer[key]
            self.assertEqual(
                first.output_parameter_state_refs,
                second.input_parameter_state_refs,
            )
            self.assertTrue(first.production_wgrad_action_refs)
            self.assertTrue(first.production_sgd_action_refs)
            self.assertTrue(first.production_store_action_refs)
            if first.lowering is MoeParameterLowering.ROUTER_REPLICATED_GATE:
                self.assertEqual(len(first.production_parameter_state_refs), 2)
                self.assertEqual(len(first.production_sync_action_refs), 2)
            else:
                self.assertEqual(len(first.parameter_refs), 3)
                self.assertEqual(len(first.production_parameter_state_refs), 1)
                self.assertFalse(first.production_sync_action_refs)

    def test_route_trace_substitution_fails_closed(self) -> None:
        sequence = _sequence(WorkloadFamily.MOE_INFERENCE)
        units = _replace_unit(
            sequence,
            0,
            route_trace_ref=sequence.units[1].route_trace_ref,
            route_trace_digest=sequence.units[1].route_trace_digest,
        )
        with self.assertRaisesRegex(SchemaError, "another unit|route"):
            MoeCompileSequence.create(
                materialization=sequence.materialization,
                units=units,
            )

    def test_parameter_version_substitution_fails_closed(self) -> None:
        sequence = _sequence(WorkloadFamily.MOE_TRAINING)
        unit = sequence.units[2]
        bindings = list(unit.parameter_bindings)
        semantic = bindings[0]._semantic()
        semantic["input_parameter_state_refs"] = (
            sequence.units[0].parameter_bindings[0].input_parameter_state_refs
        )
        bindings[0] = type(bindings[0]).create(**semantic)
        units = _replace_unit(sequence, 2, parameter_bindings=tuple(bindings))
        with self.assertRaisesRegex(SchemaError, "version|continuous|lineage"):
            MoeCompileSequence.create(
                materialization=sequence.materialization,
                units=units,
            )

    def test_tp_path_is_rejected_instead_of_overclaimed(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedFeatureError,
            "tp_dp_pp_must_equal_one",
        ):
            compile_moe_sequence(
                _manifest(WorkloadFamily.MOE_INFERENCE, tp=2)
            )

    def test_sequence_is_repeatable_and_runtime_overclaim_is_rejected(self) -> None:
        first = _sequence(WorkloadFamily.MOE_INFERENCE)
        self.assertEqual(first, compile_moe_sequence(first.materialization))
        with self.assertRaisesRegex(SchemaError, "overclaim"):
            replace(first, runtime_status="complete").validate()


if __name__ == "__main__":
    unittest.main()
