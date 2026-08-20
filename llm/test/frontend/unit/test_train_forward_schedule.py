from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from _fixtures import naive_intra_die_scheduling_context
from test_train_forward_projection import _project

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import schedule_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import schedule_bundle
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir2 import (
    IntraDieSchedule,
    IntraDieScheduleSet,
    OrdinaryNodeOrigin,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    TRAIN_SCHEDULED_IR2_SCHEMA_VERSION,
    TrainScheduledIR2,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)


def _schedule():
    source = _project()[-1]
    context = naive_intra_die_scheduling_context("n6_3b_test")
    result = schedule_train_forward(source, context)
    return source, context, result


def _rebuild_schedule(
    schedule: IntraDieSchedule,
    **updates: object,
) -> IntraDieSchedule:
    semantic_key = schedule._semantic_key()
    semantic_key.update(updates)
    return IntraDieSchedule.create(
        producer_pass=schedule.producer_pass,
        **semantic_key,
    )


def _with_schedule(replica, schedule_index: int, changed: IntraDieSchedule):
    schedules = list(replica.schedule_set.schedules)
    schedules[schedule_index] = changed
    return replace(
        replica,
        schedule_set=IntraDieScheduleSet.create(
            producer_pass=replica.schedule_set.producer_pass,
            source_projection_id=replica.schedule_set.source_projection_id,
            source_ir1_id=replica.schedule_set.source_ir1_id,
            schedules=tuple(schedules),
        ),
    )


class TrainForwardScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source, cls.context, cls.result = _schedule()

    def test_dp2_tp2_schedule_core_sram_flow_and_ce_goldens(self) -> None:
        result = self.result
        self.assertEqual(result.schema_version, TRAIN_SCHEDULED_IR2_SCHEMA_VERSION)
        expected_core_sets = ({0, 4}, {8, 12})
        binding_id_sets: list[set[str]] = []
        for replica_index, replica in enumerate(result.replicas):
            self.assertEqual(
                tuple(len(schedule.placements) for schedule in replica.schedule_set.schedules),
                ((77, 77, 0, 0) if replica_index == 0 else (0, 0, 77, 77)),
            )
            local_dies = {
                placement.die_id
                for placement in replica.projected.graph.groups[0].placements
            }
            local_routes = {
                route.id
                for route in replica.projected.graph.groups[0].embedding.routes
            }
            local_cores: set[int] = set()
            ce_node = next(
                node for node in replica.projected.graph.nodes
                if node.kind is OpKind.CE_FORWARD
            )
            predecessor = next(
                edge.source_node
                for edge in replica.projected.graph.edges
                if edge.destination_node == ce_node.id
            )
            replica_bindings: set[str] = set()
            for schedule, dag in zip(
                replica.schedule_set.schedules,
                replica.projected.projection.dags,
            ):
                if schedule.die_id not in local_dies:
                    self.assertEqual(
                        (
                            schedule.placements,
                            schedule.buffer_bindings,
                            schedule.task_buffer_uses,
                            schedule.task_state_uses,
                            schedule.flow_routes,
                            schedule.runtime_bindings,
                            schedule.core_orders,
                        ),
                        ((), (), (), (), (), (), ()),
                    )
                    continue
                self.assertEqual(
                    (
                        len(schedule.placements),
                        len(schedule.buffer_bindings),
                        len(schedule.task_buffer_uses),
                        len(schedule.task_state_uses),
                        len(schedule.flow_routes),
                        len(schedule.runtime_bindings),
                        len(schedule.core_orders),
                    ),
                    (77, 67, 135, 15, 16, 24, 1),
                )
                self.assertEqual(
                    Counter(binding.dtype for binding in schedule.buffer_bindings),
                    Counter({DType.FP16: 64, DType.INT32: 2, DType.FP32: 1}),
                )
                self.assertTrue(
                    all(route.pair_route_ref in local_routes for route in schedule.flow_routes)
                )
                local_cores.update(placement.core_id for placement in schedule.placements)
                replica_bindings.update(binding.id for binding in schedule.buffer_bindings)

                max_region_end = max(
                    binding.region_offset_bytes + binding.size_bytes
                    for binding in schedule.buffer_bindings
                )
                self.assertEqual(max_region_end, 17104)
                die = next(
                    item for item in replica.projected.graph.fabric.dies
                    if item.id == schedule.die_id
                )
                core = next(
                    item for item in die.cores
                    if item.runtime_core_id == schedule.placements[0].core_id
                )
                profile = next(
                    item for item in replica.projected.graph.fabric.sram_profiles
                    if item.id == core.sram_profile_ref
                )
                self.assertEqual(profile.capacity_bytes, 65536)
                self.assertLessEqual(max_region_end, profile.regions[0].size_bytes)

                ce_task = next(
                    task for task in dag.tasks
                    if isinstance(task.origin_ref, OrdinaryNodeOrigin)
                    and task.origin_ref.op_id == ce_node.id
                )
                placement = next(
                    item for item in schedule.placements
                    if item.task_id == ce_task.id
                )
                order = next(
                    item for item in schedule.core_orders
                    if item.core_id == placement.core_id
                )
                rank = ce_task.origin_ref.rank
                self.assertEqual(order.task_ids[-2:], (
                    f"task.{predecessor}.rank.{rank}.comp",
                    ce_task.id,
                ))
                uses = tuple(
                    use for use in schedule.task_buffer_uses
                    if use.task_id == ce_task.id
                )
                binding_index = {
                    binding.id: binding for binding in schedule.buffer_bindings
                }
                ce_bindings = tuple(binding_index[use.binding_id] for use in uses)
                self.assertEqual(
                    tuple(
                        (
                            binding.dtype,
                            binding.size_bytes,
                            binding.lifetime_start,
                            binding.lifetime_end_exclusive,
                        )
                        for binding in ce_bindings
                    ),
                    (
                        (DType.FP16, 256, 75, 77),
                        (DType.INT32, 16, 76, 77),
                        (DType.FP32, 16, 76, 77),
                    ),
                )
            self.assertEqual(local_cores, expected_core_sets[replica_index])
            binding_id_sets.append(replica_bindings)
        self.assertTrue(binding_id_sets[0].isdisjoint(binding_id_sets[1]))
        result.validate_against(self.source, self.context)

    def test_determinism_strict_round_trip_and_old_wrapper_boundary(self) -> None:
        self.assertEqual(
            schedule_train_forward(self.source, self.context),
            self.result,
        )
        decoded = loads_dataclass(
            TrainScheduledIR2,
            canonical_json(self.result),
            path="train_scheduled",
        )
        self.assertEqual(decoded, self.result)
        decoded.validate_against(self.source, self.context)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                self.result,
                schema_version="wafer_frontend.train_scheduled_ir2/v0",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "ProjectedIR2Bundle"):
            schedule_bundle(self.source, self.context)

    def test_order_buffer_route_and_replica_tamper_fail_closed(self) -> None:
        replica = self.result.replicas[0]
        schedule_index = 0
        schedule = replica.schedule_set.schedules[schedule_index]
        order = schedule.core_orders[0]
        bad_order = replace(
            order,
            task_ids=(*order.task_ids[:-2], order.task_ids[-1], order.task_ids[-2]),
        )
        with self.assertRaisesRegex(SchemaError, "dependency|order|precede"):
            _with_schedule(
                replica,
                schedule_index,
                _rebuild_schedule(schedule, core_orders=(bad_order,)),
            ).validate("replica")

        ce_task_id = order.task_ids[-1]
        ce_use = next(
            use for use in schedule.task_buffer_uses
            if use.task_id == ce_task_id
        )
        bad_bindings = tuple(
            replace(binding, lifetime_start=binding.lifetime_start + 1)
            if binding.id == ce_use.binding_id
            else binding
            for binding in schedule.buffer_bindings
        )
        with self.assertRaisesRegex(SchemaError, "lifetime"):
            _with_schedule(
                replica,
                schedule_index,
                _rebuild_schedule(schedule, buffer_bindings=bad_bindings),
            ).validate("replica")

        other_route = self.result.replicas[1].projected.graph.groups[0].embedding.routes[0]
        bad_routes = (
            replace(schedule.flow_routes[0], pair_route_ref=other_route.id),
            *schedule.flow_routes[1:],
        )
        with self.assertRaisesRegex(SchemaError, "route|PairRoute"):
            _with_schedule(
                replica,
                schedule_index,
                _rebuild_schedule(schedule, flow_routes=bad_routes),
            ).validate("replica")

        with self.assertRaisesRegex(SchemaError, "canonical DP order"):
            replace(
                self.result,
                replicas=tuple(reversed(self.result.replicas)),
            ).validate()


if __name__ == "__main__":
    unittest.main()
