from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from test_train_forward_n4 import _pipeline

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import project_train_forward
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_bundle
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir2 import (
    IR2ProjectionResult,
    IntraDieDAG,
    OrdinaryNodeOrigin,
    SemanticTaskKind,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    TRAIN_PROJECTED_IR2_SCHEMA_VERSION,
    ProjectToIR2Context,
    TrainProjectedIR2,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)


def _project():
    source = _pipeline()[-1]
    context = ProjectToIR2Context.create(
        producer_pass="n6_3a_test",
        state_transfers=(),
    )
    result = project_train_forward(source, context)
    return source, context, result


def _with_dag(replica, dag_index: int, changed: IntraDieDAG):
    dags = list(replica.projection.dags)
    dags[dag_index] = changed
    projection_key = replica.projection._semantic_key()
    projection_key["dags"] = tuple(dags)
    return replace(
        replica,
        projection=IR2ProjectionResult.create(
            producer_pass=replica.projection.producer_pass,
            **projection_key,
        ),
    )


def _rebuild_dag(dag: IntraDieDAG, **updates: object) -> IntraDieDAG:
    semantic_key = dag._semantic_key()
    semantic_key.update(updates)
    return IntraDieDAG.create(
        producer_pass=dag.producer_pass,
        **semantic_key,
    )


class TrainForwardProjectionTest(unittest.TestCase):
    def test_dp2_tp2_ce_projection_numeric_and_buffer_contract(self) -> None:
        source, context, result = _project()
        self.assertEqual(result.schema_version, TRAIN_PROJECTED_IR2_SCHEMA_VERSION)
        self.assertEqual(len(result.replicas), 2)
        expected_task_kinds = Counter(
            {
                "comp": 60,
                "dma_in": 30,
                "send": 16,
                "recv": 16,
                "local_copy": 8,
                "barrier": 8,
                "wait": 8,
                "reduce": 8,
            }
        )
        all_task_ids: list[set[str]] = []
        all_flow_ids: list[set[str]] = []
        for replica_index, replica in enumerate(result.replicas):
            local_dies = {
                placement.die_id
                for placement in replica.graph.groups[0].placements
            }
            self.assertEqual(
                tuple(len(dag.tasks) for dag in replica.projection.dags),
                ((77, 77, 0, 0) if replica_index == 0 else (0, 0, 77, 77)),
            )
            self.assertEqual(
                Counter(
                    task.kind.value
                    for dag in replica.projection.dags
                    for task in dag.tasks
                ),
                expected_task_kinds,
            )
            self.assertEqual(replica.projection.state_transfers, ())
            self.assertFalse(
                any(
                    task.kind is SemanticTaskKind.DMA_OUT
                    for dag in replica.projection.dags
                    for task in dag.tasks
                )
            )
            self.assertEqual(
                sum(len(dag.state_staging_values) for dag in replica.projection.dags),
                30,
            )
            ce_node = next(
                node for node in replica.graph.nodes
                if node.kind is OpKind.CE_FORWARD
            )
            value_index = {value.id: value for value in replica.graph.values}
            self.assertEqual(
                tuple(value_index[item].dtype for item in ce_node.inputs),
                (DType.FP16, DType.INT32),
            )
            self.assertIs(value_index[ce_node.outputs[0]].dtype, DType.FP32)
            ce_tasks = tuple(
                (dag, task)
                for dag in replica.projection.dags
                if dag.die_id in local_dies
                for task in dag.tasks
                if isinstance(task.origin_ref, OrdinaryNodeOrigin)
                and task.origin_ref.op_id == ce_node.id
            )
            self.assertEqual(len(ce_tasks), 2)
            for dag, task in ce_tasks:
                self.assertEqual(task.read_values, ce_node.inputs)
                self.assertEqual(task.write_values, ce_node.outputs)
                self.assertEqual(
                    tuple(item.role for item in task.compute.inputs),
                    ("logits", "labels"),
                )
                self.assertEqual(
                    tuple(item.role for item in task.compute.outputs),
                    ("loss",),
                )
                local_values = {value.id: value for value in dag.values}
                self.assertIs(local_values[ce_node.inputs[0]].dtype, DType.FP16)
                self.assertIs(local_values[ce_node.inputs[1]].dtype, DType.INT32)
                self.assertIs(local_values[ce_node.outputs[0]].dtype, DType.FP32)
                self.assertIn(task.id, local_values[ce_node.inputs[1]].consumer_tasks)
                self.assertIn(task.id, local_values[ce_node.outputs[0]].producer_tasks)
                self.assertEqual(len(task.deps), 1)
            all_task_ids.append(
                {
                    task.id
                    for dag in replica.projection.dags
                    for task in dag.tasks
                }
            )
            all_flow_ids.append(
                {
                    flow.id
                    for dag in replica.projection.dags
                    for flow in dag.flows
                }
            )
        self.assertTrue(all_task_ids[0].isdisjoint(all_task_ids[1]))
        self.assertTrue(all_flow_ids[0].isdisjoint(all_flow_ids[1]))
        result.validate_against(source, context)

    def test_determinism_strict_round_trip_and_old_version(self) -> None:
        source, context, result = _project()
        self.assertEqual(project_train_forward(source, context), result)
        decoded = loads_dataclass(
            TrainProjectedIR2,
            canonical_json(result),
            path="train_projected",
        )
        self.assertEqual(decoded, result)
        decoded.validate_against(source, context)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                result,
                schema_version="wafer_frontend.train_projected_ir2/v0",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "InterDiePlanBundle"):
            project_bundle(source, context)

    def test_replica_order_and_cross_replica_projection_fail_closed(self) -> None:
        _, _, result = _project()
        with self.assertRaisesRegex(SchemaError, "canonical DP order"):
            replace(result, replicas=tuple(reversed(result.replicas))).validate()
        forged = replace(
            result.replicas[1],
            projection=result.replicas[0].projection,
        )
        with self.assertRaisesRegex(SchemaError, "different IR-1|source_ir1"):
            replace(
                result,
                replicas=(result.replicas[0], forged),
            ).validate()

    def test_ce_role_dependency_and_buffer_tamper_fail_closed(self) -> None:
        _, _, result = _project()
        replica = result.replicas[0]
        dag_index = next(
            index for index, dag in enumerate(replica.projection.dags)
            if any(task.op_kind is OpKind.CE_FORWARD for task in dag.tasks)
        )
        dag = replica.projection.dags[dag_index]
        task_index = next(
            index for index, task in enumerate(dag.tasks)
            if task.op_kind is OpKind.CE_FORWARD
        )
        task = dag.tasks[task_index]
        assert task.compute is not None

        bad_tasks = list(dag.tasks)
        bad_tasks[task_index] = replace(
            task,
            compute=replace(
                task.compute,
                inputs=(
                    replace(task.compute.inputs[0], role="labels"),
                    task.compute.inputs[1],
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "operand roles/arity"):
            _with_dag(
                replica,
                dag_index,
                _rebuild_dag(dag, tasks=tuple(bad_tasks)),
            ).validate("replica")

        bad_tasks[task_index] = replace(task, deps=())
        with self.assertRaisesRegex(SchemaError, "dependency|deps"):
            _with_dag(
                replica,
                dag_index,
                _rebuild_dag(dag, tasks=tuple(bad_tasks)),
            ).validate("replica")

        label_id = task.read_values[1]
        bad_values = tuple(
            replace(value, dtype=DType.FP16)
            if value.id == label_id
            else value
            for value in dag.values
        )
        with self.assertRaisesRegex(SchemaError, "metadata disagrees"):
            _with_dag(
                replica,
                dag_index,
                _rebuild_dag(dag, values=bad_values),
            ).validate("replica")


if __name__ == "__main__":
    unittest.main()
