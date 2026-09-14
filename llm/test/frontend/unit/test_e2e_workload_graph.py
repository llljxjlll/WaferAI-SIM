from __future__ import annotations

import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_e2e_workload_graph import (
    build_e2e_workload_graph,
)
from llm.frontend.wafer_frontend.passes.validate_e2e_workload_graph import (
    validate_e2e_workload_coverage,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.e2e_workload_graph import (
    E2EArtifactStatus,
    E2EOperationKind,
    E2ERouteTrace,
    E2EStateKind,
    E2EStateVersion,
    E2EWorkloadGraph,
    E2EWorkloadOperation,
)
from llm.frontend.wafer_frontend.schema.ir0 import ReduceOp
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


def _request(family: WorkloadFamily) -> WorkloadRunRequest:
    is_moe = family.is_moe
    model = WorkloadModelSpec(
        architecture=(
            WorkloadModelArchitecture.LLAMA_MOE
            if is_moe
            else WorkloadModelArchitecture.LLAMA_DENSE
        ),
        vocabulary_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        head_dim=4,
        max_sequence_length=64,
        dtype=DType.FP16,
        num_experts=4 if is_moe else 0,
        experts_per_token=1 if is_moe else 0,
    )
    if family.is_training:
        steps = WorkloadStepSpec(
            training=WorkloadTrainingSteps(
                step_count=2,
                global_batch_size=2,
                micro_batch_size=1,
                micro_batch_count=1,
                sequence_length=4,
            )
        )
        optimizer = WorkloadOptimizerSpec(
            kind=WorkloadOptimizerKind.SGD,
            learning_rate=0.01,
        )
        dp = 2
    else:
        steps = WorkloadStepSpec(
            inference=WorkloadInferenceSteps(
                prefill_tokens=4,
                decode_steps=2,
                request_count=1,
            )
        )
        optimizer = None
        dp = 1
    return WorkloadRunRequest.create(
        family=family,
        model=model,
        steps=steps,
        mesh=WorkloadMeshSpec(rows=2, columns=4 if is_moe else 2),
        parallel=WorkloadParallelSpec(tp=2, ep=2 if is_moe else 1, dp=dp),
        optimizer=optimizer,
    )


def _count(graph, kind: E2EOperationKind) -> int:
    return sum(operation.kind is kind for operation in graph.operations)


def _without_first(graph, kind: E2EOperationKind, predicate=lambda operation: True):
    removed = False
    operations = []
    for operation in graph.operations:
        if not removed and operation.kind is kind and predicate(operation):
            removed = True
            continue
        operations.append(operation)
    if not removed:
        raise AssertionError(f"fixture contains no removable {kind.value}")
    return replace(graph, operations=tuple(operations))


def _without_matching(graph, predicate):
    operations = tuple(
        operation for operation in graph.operations if not predicate(operation)
    )
    if len(operations) == len(graph.operations):
        raise AssertionError("fixture contains no removable operations")
    return replace(graph, operations=operations)


def _with_dependencies(graph, predicate, dependency_ids):
    """Rebuild a graph after replacing one operation's dependency edges."""

    id_map = {}
    operations = []
    replacement_count = 0
    for operation in graph.operations:
        source_dependencies = operation.deps
        if predicate(operation):
            source_dependencies = dependency_ids
            replacement_count += 1
        rebuilt = E2EWorkloadOperation.create(
            case_id=operation.case_id,
            sequence_index=operation.sequence_index,
            kind=operation.kind,
            phase=operation.phase,
            step=operation.step,
            layer=operation.layer,
            expert=operation.expert,
            parameter_ref=operation.parameter_ref,
            reads=operation.reads,
            writes=operation.writes,
            input_value_refs=operation.input_value_refs,
            output_value_refs=operation.output_value_refs,
            group_refs=operation.group_refs,
            reduce_op=operation.reduce_op,
            normalization_denominator=operation.normalization_denominator,
            deps=tuple(id_map[item] for item in source_dependencies),
        )
        id_map[operation.id] = rebuilt.id
        operations.append(rebuilt)
    if replacement_count != 1:
        raise AssertionError(f"expected one target operation, found {replacement_count}")
    states = tuple(
        E2EStateVersion.create(
            case_id=state.case_id,
            logical_name=state.logical_name,
            kind=state.kind,
            version=state.version,
            layer=state.layer,
            expert=state.expert,
            producer_op_id=(
                None
                if state.producer_op_id is None
                else id_map[state.producer_op_id]
            ),
            consumer_op_ids=tuple(id_map[item] for item in state.consumer_op_ids),
        )
        for state in graph.state_versions
    )
    values = tuple(
        replace(
            value,
            producer_op_id=(
                None
                if value.producer_op_id is None
                else id_map[value.producer_op_id]
            ),
            consumer_op_ids=tuple(id_map[item] for item in value.consumer_op_ids),
        )
        for value in graph.tensor_values
    )
    return E2EWorkloadGraph.create(
        request=graph.request,
        placement=graph.placement,
        operations=tuple(operations),
        state_versions=states,
        tensor_values=values,
        route_traces=graph.route_traces,
    )


class E2EWorkloadGraphTest(unittest.TestCase):
    def test_all_four_families_have_independently_verified_coverage(self) -> None:
        for family in WorkloadFamily:
            with self.subTest(family=family):
                graph = build_e2e_workload_graph(_request(family))
                report = validate_e2e_workload_coverage(graph)

                self.assertEqual(report.family, family.value)
                self.assertEqual(report.layer_count, 2)
                self.assertEqual(
                    report.logical_step_count,
                    2 if family.is_training else 3,
                )
                self.assertGreater(report.operation_count, 0)
                self.assertGreater(report.state_version_count, 0)
                self.assertGreater(report.tensor_value_count, 0)
                self.assertFalse(report.lowering_verified)
                self.assertFalse(report.runtime_verified)
                self.assertIs(
                    graph.lowering_status,
                    E2EArtifactStatus.NOT_MATERIALIZED,
                )
                self.assertIs(
                    graph.runtime_status,
                    E2EArtifactStatus.NOT_MATERIALIZED,
                )

    def test_dense_inference_covers_layers_head_and_kv_prefill_decode(self) -> None:
        graph = build_e2e_workload_graph(
            _request(WorkloadFamily.DENSE_INFERENCE)
        )
        self.assertEqual(_count(graph, E2EOperationKind.EMBEDDING), 3)
        self.assertEqual(_count(graph, E2EOperationKind.ATTENTION), 6)
        self.assertEqual(_count(graph, E2EOperationKind.MLP_DOWN), 6)
        self.assertEqual(_count(graph, E2EOperationKind.LM_HEAD), 3)
        self.assertEqual(_count(graph, E2EOperationKind.KV_APPEND), 6)

        for layer in range(2):
            versions = {
                state.version
                for state in graph.state_versions
                if state.kind is E2EStateKind.KV and state.layer == layer
            }
            self.assertEqual(versions, {0, 1, 2, 3})

        state_by_id = {state.id: state for state in graph.state_versions}
        kv_values = tuple(
            value
            for value in graph.tensor_values
            if state_by_id[value.state_ref].kind is E2EStateKind.KV
            and state_by_id[value.state_ref].layer == 0
        )
        bytes_by_version = {
            version: sum(
                value.size_bytes
                for value in kv_values
                if state_by_id[value.state_ref].version == version
            )
            for version in range(4)
        }
        self.assertEqual(bytes_by_version[0], 0)
        self.assertLess(bytes_by_version[1], bytes_by_version[2])
        self.assertLess(bytes_by_version[2], bytes_by_version[3])
        self.assertTrue(
            {E2EStateKind.ACTIVATION, E2EStateKind.LOGITS}.issubset(
                {state.kind for state in graph.state_versions}
            )
        )

    def test_dense_training_covers_two_full_update_steps(self) -> None:
        graph = build_e2e_workload_graph(
            _request(WorkloadFamily.DENSE_TRAINING)
        )
        parameter_count = sum(
            state.kind is E2EStateKind.PARAMETER and state.version == 0
            for state in graph.state_versions
        )
        self.assertEqual(_count(graph, E2EOperationKind.LOSS), 2)
        self.assertEqual(_count(graph, E2EOperationKind.DENSE_BACKWARD), 4)
        self.assertEqual(_count(graph, E2EOperationKind.WGRAD), 2 * parameter_count)
        self.assertEqual(
            _count(graph, E2EOperationKind.GRADIENT_SYNC),
            2 * parameter_count,
        )
        self.assertEqual(
            _count(graph, E2EOperationKind.SGD_UPDATE),
            2 * parameter_count,
        )
        self.assertEqual(
            _count(graph, E2EOperationKind.PARAMETER_STORE),
            2 * parameter_count,
        )
        self.assertEqual(_count(graph, E2EOperationKind.STEP_COMMIT), 2)
        for state in graph.state_versions:
            if state.kind is E2EStateKind.PARAMETER:
                self.assertIn(state.version, {0, 1, 2})
        self.assertIn(E2EStateKind.LOSS, {state.kind for state in graph.state_versions})
        gradient_states = {
            state.id
            for state in graph.state_versions
            if state.kind is E2EStateKind.GRADIENT
        }
        self.assertTrue(
            all(
                value.owner_domain_ref is not None
                for value in graph.tensor_values
                if value.state_ref in gradient_states
            )
        )
        groups = {group.id: group for group in graph.placement.groups}
        for operation in graph.operations:
            if operation.kind is E2EOperationKind.GRADIENT_SYNC:
                self.assertIs(operation.reduce_op, ReduceOp.SUM)
                self.assertTrue(operation.group_refs)
                self.assertEqual(
                    {len(groups[item].ranks) for item in operation.group_refs},
                    {operation.normalization_denominator},
                )

    def test_moe_inference_and_training_cover_routing_and_experts(self) -> None:
        inference = build_e2e_workload_graph(
            _request(WorkloadFamily.MOE_INFERENCE)
        )
        self.assertEqual(_count(inference, E2EOperationKind.ROUTER), 6)
        self.assertEqual(_count(inference, E2EOperationKind.DISPATCH), 6)
        self.assertEqual(_count(inference, E2EOperationKind.EXPERT_FORWARD), 24)
        self.assertEqual(_count(inference, E2EOperationKind.COMBINE), 6)

        training = build_e2e_workload_graph(
            _request(WorkloadFamily.MOE_TRAINING)
        )
        self.assertEqual(_count(training, E2EOperationKind.GRAD_DISPATCH), 4)
        self.assertEqual(_count(training, E2EOperationKind.EXPERT_BACKWARD), 16)
        self.assertEqual(_count(training, E2EOperationKind.DX_COMBINE), 4)
        self.assertEqual(_count(training, E2EOperationKind.SHARED_BACKWARD), 4)
        self.assertGreater(_count(training, E2EOperationKind.WGRAD), 0)
        self.assertGreater(_count(training, E2EOperationKind.EXPERT_GRADIENT), 0)
        self.assertGreater(_count(training, E2EOperationKind.ROUTER_GRADIENT), 0)
        validate_e2e_workload_coverage(training)

        decode_traces = tuple(
            trace for trace in inference.route_traces if trace.step == 1
        )
        self.assertEqual(len(decode_traces), 2)
        self.assertTrue(
            all(trace.zero_token_experts == (1, 2, 3) for trace in decode_traces)
        )
        values = {value.id: value for value in inference.tensor_values}
        for trace in decode_traces:
            payload_bytes = tuple(
                values[value_ref].size_bytes
                for value_ref in trace.dispatch_value_refs
            )
            self.assertIn(0, payload_bytes)
            self.assertTrue(any(size > 0 for size in payload_bytes))

    def test_fault_injection_rejects_changed_frozen_route_trace(self) -> None:
        graph = build_e2e_workload_graph(_request(WorkloadFamily.MOE_INFERENCE))
        source = graph.route_traces[0]
        changed_route = source.expert_by_token[1:] + source.expert_by_token[:1]
        changed = E2ERouteTrace.create(
            case_id=source.case_id,
            phase=source.phase,
            step=source.step,
            layer=source.layer,
            token_count=source.token_count,
            expert_by_token=changed_route,
            expert_token_counts=source.expert_token_counts,
            route_value_refs=source.route_value_refs,
            dispatch_value_refs=source.dispatch_value_refs,
            expert_output_value_refs=source.expert_output_value_refs,
            combine_value_refs=source.combine_value_refs,
        )
        broken = E2EWorkloadGraph.create(
            request=graph.request,
            placement=graph.placement,
            operations=graph.operations,
            state_versions=graph.state_versions,
            tensor_values=graph.tensor_values,
            route_traces=(changed, *graph.route_traces[1:]),
        )
        with self.assertRaisesRegex(SchemaError, "frozen token-to-expert"):
            validate_e2e_workload_coverage(broken)

    def test_fault_injection_rejects_missing_last_layer_and_kv_append(self) -> None:
        graph = build_e2e_workload_graph(
            _request(WorkloadFamily.DENSE_INFERENCE)
        )
        cases = (
            (
                _without_matching(
                    graph,
                    lambda operation: operation.layer == 1,
                ),
                "input_norm",
            ),
            (_without_first(graph, E2EOperationKind.KV_APPEND), "kv_append"),
        )
        for broken, expected in cases:
            with self.subTest(missing=expected):
                with self.assertRaisesRegex(SchemaError, expected):
                    validate_e2e_workload_coverage(broken)

    def test_fault_injection_rejects_training_semantic_gaps(self) -> None:
        dense = build_e2e_workload_graph(
            _request(WorkloadFamily.DENSE_TRAINING)
        )
        moe = build_e2e_workload_graph(_request(WorkloadFamily.MOE_TRAINING))
        cases = (
            (_without_first(dense, E2EOperationKind.LOSS), "loss"),
            (_without_first(dense, E2EOperationKind.WGRAD), "wgrad"),
            (_without_first(dense, E2EOperationKind.SGD_UPDATE), "sgd_update"),
            (
                _without_first(moe, E2EOperationKind.EXPERT_GRADIENT),
                "expert_gradient",
            ),
        )
        for broken, expected in cases:
            with self.subTest(missing=expected):
                with self.assertRaisesRegex(SchemaError, expected):
                    validate_e2e_workload_coverage(broken)

    def test_fault_injection_rejects_missing_inference_happens_before(self) -> None:
        graph = build_e2e_workload_graph(
            _request(WorkloadFamily.DENSE_INFERENCE)
        )
        previous_append = next(
            operation
            for operation in graph.operations
            if operation.kind is E2EOperationKind.KV_APPEND
            and operation.step == 0
            and operation.layer == 1
        )
        cases = (
            (
                _with_dependencies(
                    graph,
                    lambda operation: (
                        operation.kind is E2EOperationKind.EMBEDDING
                        and operation.step == 1
                    ),
                    (previous_append.id,),
                ),
                "prefill/decode",
            ),
            (
                _with_dependencies(
                    graph,
                    lambda operation: (
                        operation.kind is E2EOperationKind.KV_LOAD
                        and operation.step == 1
                        and operation.layer == 0
                    ),
                    (),
                ),
                "KV append must happen before",
            ),
        )
        for broken, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(SchemaError, expected):
                    validate_e2e_workload_coverage(broken)

    def test_fault_injection_rejects_training_happens_before_gaps(self) -> None:
        graph = build_e2e_workload_graph(
            _request(WorkloadFamily.DENSE_TRAINING)
        )
        loss = next(
            operation
            for operation in graph.operations
            if operation.kind is E2EOperationKind.LOSS and operation.step == 0
        )
        cases = (
            (
                _with_dependencies(
                    graph,
                    lambda operation: (
                        operation.kind is E2EOperationKind.DENSE_BACKWARD
                        and operation.step == 0
                        and operation.layer == 1
                    ),
                    (),
                ),
                "loss must happen before",
            ),
            (
                _with_dependencies(
                    graph,
                    lambda operation: (
                        operation.kind is E2EOperationKind.DENSE_BACKWARD
                        and operation.step == 0
                        and operation.layer == 0
                    ),
                    (loss.id,),
                ),
                "reverse layer order",
            ),
            (
                _with_dependencies(
                    graph,
                    lambda operation: (
                        operation.kind is E2EOperationKind.GRADIENT_SYNC
                        and operation.step == 0
                        and operation.parameter_ref == "embedding.weight"
                    ),
                    (loss.id,),
                ),
                "gradient must happen before gradient sync",
            ),
            (
                _with_dependencies(
                    graph,
                    lambda operation: (
                        operation.kind is E2EOperationKind.PARAMETER_LOAD
                        and operation.step == 1
                        and operation.parameter_ref == "embedding.weight"
                    ),
                    (),
                ),
                "store must happen before next-step load",
            ),
        )
        for broken, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(SchemaError, expected):
                    validate_e2e_workload_coverage(broken)


if __name__ == "__main__":
    unittest.main()
