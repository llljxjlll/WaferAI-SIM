from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import LoweringContext
from llm.frontend.wafer_frontend.schema.action import FusionPlan
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferBinding,
    BufferOwnership,
    BufferUseRole,
    CoreOrder,
    IntraDieSchedule,
    IntraDieScheduleSet,
    IR2ProjectionResult,
    TaskBufferUse,
    TaskPlacement,
    TensorSlice,
)

from test_action_schema import valid_plan
from test_global_action_schema import _create_global
from test_ir2_schema import ordinary_projection


def valid_lowering_context() -> LoweringContext:
    ir1, projection = ordinary_projection()
    schedules = []
    for dag in projection.dags:
        task = dag.tasks[0]
        die = next(item for item in ir1.fabric.dies if item.id == dag.die_id)
        runtime_core = die.cores[0].runtime_core_id
        input_value, output_value = dag.values
        input_bytes = math.prod(input_value.shape) * 2
        output_bytes = math.prod(output_value.shape) * 2
        output_offset = (input_bytes + 63) // 64 * 64
        input_binding = BufferBinding(
            f"binding_input_d{dag.die_id}", input_value.id,
            TensorSlice(input_value.id, (0,) * len(input_value.shape), input_value.shape),
            runtime_core, "sram_main", 0, input_bytes, 64, (0, 1, 2, 3),
            f"storage_input_d{dag.die_id}", None, BufferOwnership.BORROWED,
            0, 1, input_value.dtype, input_value.logical_layout,
        )
        output_binding = BufferBinding(
            f"binding_output_d{dag.die_id}", output_value.id,
            TensorSlice(output_value.id, (0,) * len(output_value.shape), output_value.shape),
            runtime_core, "sram_main", output_offset, output_bytes, 64,
            (0, 1, 2, 3), f"storage_output_d{dag.die_id}", None,
            BufferOwnership.OWNED, 0, 1, output_value.dtype,
            output_value.logical_layout,
        )
        schedules.append(IntraDieSchedule.create(
            producer_pass="lowering_context_schedule_fixture",
            dag_id=dag.id,
            die_id=dag.die_id,
            placements=(TaskPlacement(task.id, runtime_core),),
            buffer_bindings=(input_binding, output_binding),
            task_buffer_uses=(
                TaskBufferUse(
                    task.id,
                    input_binding.id,
                    BufferAccess.READ,
                    BufferUseRole.COMP_INPUT,
                    0,
                    None,
                    input_binding.tensor_slice,
                ),
                TaskBufferUse(
                    task.id,
                    output_binding.id,
                    BufferAccess.WRITE,
                    BufferUseRole.COMP_OUTPUT,
                    0,
                    None,
                    output_binding.tensor_slice,
                ),
            ),
            task_state_uses=(),
            flow_routes=(),
            runtime_bindings=(),
            core_orders=(CoreOrder(runtime_core, (task.id,)),),
        ))
    schedule_set = IntraDieScheduleSet.create(
        producer_pass="lowering_context_schedule_fixture",
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        schedules=tuple(schedules),
    )
    global_dag = _create_global(ir1, projection, schedule_set)
    return LoweringContext(
        ir1=ir1,
        fusion_plans=(),
        standalone_plans=(),
        projection=projection,
        schedule_set=schedule_set,
        global_dag=global_dag,
    )


class LoweringContextTest(unittest.TestCase):
    def test_complete_context_cross_validates_and_is_immutable(self) -> None:
        context = valid_lowering_context()
        context.validate()
        with self.assertRaises(FrozenInstanceError):
            context.global_dag = context.global_dag

    def test_schedule_tuple_must_preserve_projection_dag_order(self) -> None:
        context = valid_lowering_context()
        reversed_set = IntraDieScheduleSet.create(
            producer_pass=context.schedule_set.producer_pass,
            source_projection_id=context.projection.id,
            source_ir1_id=context.ir1.id,
            schedules=tuple(reversed(context.schedule_set.schedules)),
        )
        with self.assertRaisesRegex(SchemaError, "DAG order"):
            replace(context, schedule_set=reversed_set).validate()

    def test_schedule_must_match_its_dag_die_pair(self) -> None:
        context = valid_lowering_context()
        left, right = context.schedule_set.schedules
        left_fields = left._semantic_key()
        right_fields = right._semantic_key()
        left_fields["die_id"] = right.die_id
        right_fields["die_id"] = left.die_id
        swapped_pairing = IntraDieScheduleSet.create(
            producer_pass=context.schedule_set.producer_pass,
            source_projection_id=context.projection.id,
            source_ir1_id=context.ir1.id,
            schedules=(
                IntraDieSchedule.create(
                    producer_pass=left.producer_pass,
                    **left_fields,
                ),
                IntraDieSchedule.create(
                    producer_pass=right.producer_pass,
                    **right_fields,
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "exactly follow"):
            replace(context, schedule_set=swapped_pairing).validate()

    def test_plan_tuple_order_is_checked_before_set_based_cross_validation(self) -> None:
        context = valid_lowering_context()
        first = valid_plan()
        fields = first._semantic_key()
        fields["group_ref"] = "different_group"
        second = FusionPlan.create(producer_pass=first.producer_pass, **fields)
        forged_projection = replace(
            context.projection,
            fusion_plan_ids=(second.id, first.id),
        )
        with self.assertRaisesRegex(SchemaError, "exactly preserve projection id order"):
            replace(
                context,
                fusion_plans=(first, second),
                projection=forged_projection,
            ).validate()

    def test_global_dag_source_ids_are_cross_validated(self) -> None:
        context = valid_lowering_context()
        fields = context.global_dag._semantic_key()
        fields["source_schedule_set_id"] = "wrong_schedule_set"
        forged = GlobalActionDAG.create(
            producer_pass=context.global_dag.producer_pass,
            **fields,
        )
        with self.assertRaisesRegex(SchemaError, "different schedule set"):
            replace(context, global_dag=forged).validate()

    def test_projection_source_ir1_is_cross_validated(self) -> None:
        context = valid_lowering_context()
        fields = context.projection._semantic_key()
        fields["source_ir1_id"] = "wrong_ir1"
        forged = IR2ProjectionResult.create(
            producer_pass=context.projection.producer_pass,
            **fields,
        )
        with self.assertRaisesRegex(SchemaError, "different IR-1"):
            replace(context, projection=forged).validate()


if __name__ == "__main__":
    unittest.main()
