from __future__ import annotations

import math
import unittest
from collections import Counter
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_moe_execution import (
    build_lite_moe_global,
    project_lite_moe,
    schedule_lite_moe,
    validate_lite_moe_global,
    validate_lite_moe_projection,
    validate_lite_moe_schedule,
)
from llm.frontend.wafer_frontend.passes.lite_moe_n4 import build_lite_moe_n4
from llm.frontend.wafer_frontend.schema.lite_moe import LiteMoeTransferRole
from llm.frontend.wafer_frontend.schema.lite_moe_execution import (
    LiteMoeGlobalDag,
    LiteMoeProjectedDie,
    LiteMoeProjection,
    LiteMoeScheduled,
    LiteMoeTaskKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from llm.test.frontend.unit.test_lite_moe_n4 import LiteMoeN4Test


class LiteMoeExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = LiteMoeN4Test()
        fixture.setUp()
        self.n4 = build_lite_moe_n4(
            fixture.placed,
            fixture.partition_context,
            fixture.planning_context,
        )
        self.projection = project_lite_moe(self.n4)
        self.schedule = schedule_lite_moe(self.projection, self.n4)
        self.global_dag = build_lite_moe_global(
            self.schedule, self.projection, self.n4
        )

    def test_exact_projection_transport_and_swiglu(self) -> None:
        tasks = tuple(task for die in self.projection.dies for task in die.tasks)
        self.assertEqual(tuple(len(die.tasks) for die in self.projection.dies), (40, 40))
        self.assertEqual(
            Counter(task.kind for task in tasks),
            Counter(
                {
                    LiteMoeTaskKind.DMA_IN: 24,
                    LiteMoeTaskKind.GEMM: 24,
                    LiteMoeTaskKind.SWIGLU: 8,
                    LiteMoeTaskKind.SEND: 8,
                    LiteMoeTaskKind.RECV: 8,
                    LiteMoeTaskKind.WAIT: 8,
                }
            ),
        )
        self.assertEqual(sum(task.bytes for task in tasks if task.kind is LiteMoeTaskKind.DMA_IN), 24576)
        self.assertEqual(len(self.projection.flows), 8)
        self.assertEqual(sum(flow.bytes for flow in self.projection.flows), 256)
        task_index = {task.id: task for task in tasks}
        binding_index = {item.id: item for item in self.n4.p2p_bindings}
        node_index = {item.id: item for item in self.n4.graph.nodes}
        value_index = {item.id: item for item in self.n4.graph.values}
        for flow in self.projection.flows:
            binding = binding_index[flow.p2p_binding_ref]
            send = task_index[flow.send_task_ref]
            recv = task_index[flow.recv_task_ref]
            wait = task_index[flow.wait_task_ref]
            self.assertEqual((send.die_id, recv.die_id, wait.die_id), (binding.source_die_id, binding.destination_die_id, binding.destination_die_id))
            self.assertEqual(wait.deps, (recv.id,))
            if binding.role is LiteMoeTransferRole.MOE_DISPATCH:
                consumers = value_index[flow.destination_value_ref].consumers
                self.assertEqual(len(consumers), 2)
                self.assertTrue(all(wait.id in task_index[f"moe.task.{node}.comp"].deps for node in consumers))
            else:
                producer = value_index[flow.source_value_ref].producer
                self.assertEqual(send.deps, (f"moe.task.{producer}.comp",))
        for task in tasks:
            if task.kind is not LiteMoeTaskKind.SWIGLU:
                continue
            workload = task.workload
            assert workload is not None
            self.assertEqual(len(task.read_values), 1)
            self.assertEqual(len(task.deps), 2)
            producers = tuple(task_index[item] for item in task.deps)
            self.assertEqual(
                tuple(item.packed_output.offset_bytes for item in producers if item.packed_output is not None),
                (0, 64),
            )
            self.assertTrue(
                all(
                    item.packed_output is not None
                    and item.packed_output.root_value_ref == task.read_values[0]
                    and item.write_values == task.read_values
                    for item in producers
                )
            )
            self.assertEqual(
                sum(item.packed_output.size_bytes for item in producers if item.packed_output is not None),
                math.prod(workload.rank_input_shape) * 2,
            )

    def test_schedule_global_exact_deterministic_and_round_trip(self) -> None:
        self.assertEqual(len(self.schedule.placements), 80)
        self.assertEqual(len(self.schedule.buffers), 64)
        self.assertEqual(
            tuple(
                max(
                    item.address + item.size_bytes
                    for item in self.schedule.buffers
                    if item.die_id == die
                )
                for die in (0, 1)
            ),
            (13792, 13792),
        )
        self.assertEqual(len(self.global_dag.actions), 80)
        self.assertEqual(
            Counter(action.kind for action in self.global_dag.actions),
            Counter(task.kind for die in self.projection.dies for task in die.tasks),
        )
        self.assertEqual(
            canonical_digest(project_lite_moe(self.n4)),
            canonical_digest(self.projection),
        )
        self.assertEqual(
            canonical_digest(schedule_lite_moe(self.projection, self.n4)),
            canonical_digest(self.schedule),
        )
        self.assertEqual(
            canonical_digest(build_lite_moe_global(self.schedule, self.projection, self.n4)),
            canonical_digest(self.global_dag),
        )
        self.assertEqual(
            loads_dataclass(LiteMoeProjection, canonical_json(self.projection), path="projection"),
            self.projection,
        )
        self.assertEqual(
            loads_dataclass(LiteMoeScheduled, canonical_json(self.schedule), path="schedule"),
            self.schedule,
        )
        self.assertEqual(
            loads_dataclass(LiteMoeGlobalDag, canonical_json(self.global_dag), path="global"),
            self.global_dag,
        )

    def test_projection_dependency_route_and_provenance_tamper_fail_closed(self) -> None:
        swiglu_die = next(
            die
            for die in self.projection.dies
            if any(task.kind is LiteMoeTaskKind.SWIGLU for task in die.tasks)
        )
        ordinal = next(
            index
            for index, task in enumerate(swiglu_die.tasks)
            if task.kind is LiteMoeTaskKind.SWIGLU
        )
        task = next(
            item
            for item in swiglu_die.tasks
            if item.id in swiglu_die.tasks[ordinal].deps
            and item.packed_output is not None
        )
        task_ordinal = swiglu_die.tasks.index(task)
        assert task.packed_output is not None
        wrong_task = replace(
            task,
            packed_output=replace(task.packed_output, offset_bytes=32),
        )
        wrong_die = replace(
            swiglu_die,
            tasks=swiglu_die.tasks[:task_ordinal]
            + (wrong_task,)
            + swiglu_die.tasks[task_ordinal + 1 :],
        )
        wrong_dies = tuple(wrong_die if die.die_id == wrong_die.die_id else die for die in self.projection.dies)
        wrong_projection = LiteMoeProjection.create(
            **(self.projection._semantic_key() | {"dies": wrong_dies})
        )
        with self.assertRaisesRegex(SchemaError, "exactly quotient"):
            validate_lite_moe_projection(wrong_projection, self.n4)

        flow = self.projection.flows[0]
        with self.assertRaisesRegex(SchemaError, "flow task closure"):
            LiteMoeProjection.create(
                **(
                    self.projection._semantic_key()
                    | {"flows": (replace(flow, recv_task_ref=flow.wait_task_ref), *self.projection.flows[1:])}
                )
            )
        wrong_source = LiteMoeProjection.create(
            **(self.projection._semantic_key() | {"source_n4_id": "forged"})
        )
        with self.assertRaisesRegex(SchemaError, "exactly quotient"):
            validate_lite_moe_projection(wrong_source, self.n4)

    def test_schedule_global_buffer_and_lineage_tamper_fail_closed(self) -> None:
        first, second = self.schedule.buffers[:2]
        with self.assertRaisesRegex(SchemaError, "overlap"):
            LiteMoeScheduled.create(
                **(
                    self.schedule._semantic_key()
                    | {
                        "buffers": (
                            first,
                            replace(second, address=first.address),
                            *self.schedule.buffers[2:],
                        )
                    }
                )
            )
        forged_schedule = LiteMoeScheduled.create(
            **(self.schedule._semantic_key() | {"source_n4_id": "forged"})
        )
        with self.assertRaisesRegex(SchemaError, "exactly bind"):
            validate_lite_moe_schedule(forged_schedule, self.projection, self.n4)

        action = self.global_dag.actions[0]
        forged_global = LiteMoeGlobalDag.create(
            **(
                self.global_dag._semantic_key()
                | {
                    "actions": (
                        replace(action, buffer_uses=()),
                        *self.global_dag.actions[1:],
                    )
                }
            )
        )
        with self.assertRaisesRegex(SchemaError, "exactly quotient"):
            validate_lite_moe_global(
                forged_global, self.schedule, self.projection, self.n4
            )


if __name__ == "__main__":
    unittest.main()
