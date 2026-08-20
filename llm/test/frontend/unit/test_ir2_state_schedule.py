from __future__ import annotations

from dataclasses import replace
import math
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir1 import IR1, MemoryInitiator
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferBinding,
    BufferOwnership,
    BufferUseRole,
    CoreOrder,
    IntraDieDAG,
    IntraDieSchedule,
    SemanticTaskKind,
    StateUseAccess,
    TaskBufferUse,
    TaskPlacement,
    TaskStateUse,
    TensorSlice,
)

from test_naive_project_state import (
    _ordinary_kv_tp1,
    _ordinary_parameter_tp1,
)


_ROLE_ORDER = {
    role: index
    for index, role in enumerate(
        (
            BufferUseRole.COMP_INPUT,
            BufferUseRole.COMP_OUTPUT,
            BufferUseRole.SEND_SOURCE,
            BufferUseRole.RECV_DESTINATION,
            BufferUseRole.REDUCE_INPUT,
            BufferUseRole.REDUCE_OUTPUT,
            BufferUseRole.LOCAL_COPY_SOURCE,
            BufferUseRole.LOCAL_COPY_DESTINATION,
            BufferUseRole.DMA_SOURCE,
            BufferUseRole.DMA_DESTINATION,
        )
    )
}


def _recreate_schedule(
    schedule: IntraDieSchedule, **changes: object
) -> IntraDieSchedule:
    fields = schedule._semantic_key()
    fields.update(changes)
    return IntraDieSchedule.create(
        producer_pass=schedule.producer_pass,
        **fields,
    )


def _hand_schedule(graph: IR1, dag: IntraDieDAG) -> IntraDieSchedule:
    die = next(item for item in graph.fabric.dies if item.id == dag.die_id)
    core = die.cores[0]
    profile = next(
        item
        for item in graph.fabric.sram_profiles
        if item.id == core.sram_profile_ref
    )
    region = next(
        item
        for item in profile.regions
        if {
            MemoryInitiator.COMPUTE,
            MemoryInitiator.LSU,
        }.issubset(item.access)
    )
    values = {
        value.id: value
        for value in (*dag.values, *dag.state_staging_values)
    }
    order = tuple(task.id for task in dag.tasks)
    positions = {task_id: index for index, task_id in enumerate(order)}
    descriptors: list[
        tuple[str, BufferUseRole, int, BufferAccess, str]
    ] = []
    for task in dag.tasks:
        if task.kind is SemanticTaskKind.COMP:
            assert task.compute is not None
            descriptors.extend(
                (
                    task.id,
                    BufferUseRole.COMP_INPUT,
                    index,
                    BufferAccess.READ,
                    operand.value_id,
                )
                for index, operand in enumerate(task.compute.inputs)
            )
            descriptors.extend(
                (
                    task.id,
                    BufferUseRole.COMP_OUTPUT,
                    index,
                    BufferAccess.WRITE,
                    operand.value_id,
                )
                for index, operand in enumerate(task.compute.outputs)
            )
        elif task.kind is SemanticTaskKind.DMA_IN:
            assert task.dma is not None
            descriptors.append(
                (
                    task.id,
                    BufferUseRole.DMA_DESTINATION,
                    0,
                    BufferAccess.WRITE,
                    task.dma.local_value_ref,
                )
            )
        elif task.kind is SemanticTaskKind.DMA_OUT:
            assert task.dma is not None
            descriptors.append(
                (
                    task.id,
                    BufferUseRole.DMA_SOURCE,
                    0,
                    BufferAccess.READ,
                    task.dma.local_value_ref,
                )
            )
        else:
            raise AssertionError(f"unexpected task kind {task.kind.value}")

    bindings: list[BufferBinding] = []
    binding_by_value: dict[str, BufferBinding] = {}
    next_offset = 0
    for value_id in sorted({item[4] for item in descriptors}):
        value = values[value_id]
        size_bytes = math.prod(value.shape) * (
            2 if value.dtype is DType.FP16 else 4
        )
        next_offset = (next_offset + 63) // 64 * 64
        absolute_start = region.base_bytes + next_offset
        first_stripe = absolute_start // profile.bank_interleave_bytes
        last_stripe = (
            absolute_start + size_bytes - 1
        ) // profile.bank_interleave_bytes
        stripe_count = last_stripe - first_stripe + 1
        banks = tuple(
            range(profile.bank_count)
            if stripe_count >= profile.bank_count
            else sorted(
                (first_stripe + index) % profile.bank_count
                for index in range(stripe_count)
            )
        )
        value_descriptors = tuple(
            item for item in descriptors if item[4] == value_id
        )
        use_positions = tuple(
            positions[item[0]] for item in value_descriptors
        )
        binding = BufferBinding(
            id=f"sram_binding.{value_id}",
            value_id=value_id,
            tensor_slice=TensorSlice(
                value_id,
                (0,) * len(value.shape),
                value.shape,
            ),
            core_id=core.runtime_core_id,
            region_ref=region.id,
            region_offset_bytes=next_offset,
            size_bytes=size_bytes,
            alignment_bytes=64,
            banks=banks,
            storage_id=f"sram_storage.{value_id}",
            alias_of=None,
            ownership=(
                BufferOwnership.OWNED
                if any(
                    item[3] is BufferAccess.WRITE
                    for item in value_descriptors
                )
                else BufferOwnership.BORROWED
            ),
            lifetime_start=min(use_positions),
            lifetime_end_exclusive=max(use_positions) + 1,
            dtype=value.dtype,
            layout=value.logical_layout,
        )
        bindings.append(binding)
        binding_by_value[value_id] = binding
        next_offset += size_bytes

    uses = tuple(
        sorted(
            (
                TaskBufferUse(
                    task_id,
                    binding_by_value[value_id].id,
                    access,
                    role,
                    operand_index,
                    None,
                    binding_by_value[value_id].tensor_slice,
                )
                for task_id, role, operand_index, access, value_id
                in descriptors
            ),
            key=lambda use: (
                use.task_id,
                _ROLE_ORDER[use.role],
                use.operand_index,
                use.binding_id,
            ),
        )
    )
    manifest = graph.persistent_state_manifest
    assert manifest is not None
    hbm_by_state = {
        binding.state_ref: binding for binding in manifest.bindings
    }
    state_uses = tuple(
        sorted(
            (
                TaskStateUse(
                    task.id,
                    hbm_by_state[task.dma.state_ref].id,
                    (
                        StateUseAccess.READ
                        if task.kind is SemanticTaskKind.DMA_IN
                        else StateUseAccess.WRITE
                    ),
                )
                for task in dag.tasks
                if task.kind
                in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT)
                and task.dma is not None
            ),
            key=lambda use: (
                use.task_id,
                use.hbm_binding_ref,
                use.access.value,
            ),
        )
    )
    return IntraDieSchedule.create(
        producer_pass="handwritten_state_schedule",
        dag_id=dag.id,
        die_id=dag.die_id,
        placements=tuple(
            TaskPlacement(task.id, core.runtime_core_id)
            for task in dag.tasks
        ),
        buffer_bindings=tuple(bindings),
        task_buffer_uses=uses,
        task_state_uses=state_uses,
        flow_routes=(),
        runtime_bindings=(),
        core_orders=(CoreOrder(core.runtime_core_id, order),),
    )


def _state_case(maker):
    graph, *_rest = maker()
    projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
    dag = next(item for item in projection.dags if item.state_access_ids)
    return graph, dag, _hand_schedule(graph, dag)


class IR2StateScheduleTest(unittest.TestCase):
    def test_parameter_and_kv_handwritten_schedules_validate(self) -> None:
        for maker in (_ordinary_parameter_tp1, _ordinary_kv_tp1):
            graph, dag, schedule = _state_case(maker)
            with self.subTest(case=maker.__name__):
                schedule.validate_against(dag, graph)

    def test_state_schedule_witnesses_fail_closed(self) -> None:
        graph, dag, schedule = _state_case(_ordinary_kv_tp1)
        first_state_use = schedule.task_state_uses[0]
        first_buffer_use = schedule.task_buffer_uses[0]
        first_hbm_ref = first_state_use.hbm_binding_ref

        wrong_direction = _recreate_schedule(
            schedule,
            task_state_uses=(
                replace(
                    first_state_use,
                    access=(
                        StateUseAccess.WRITE
                        if first_state_use.access is StateUseAccess.READ
                        else StateUseAccess.READ
                    ),
                ),
                *schedule.task_state_uses[1:],
            ),
        )
        unknown_hbm = _recreate_schedule(
            schedule,
            task_state_uses=(
                replace(first_state_use, hbm_binding_ref="missing_hbm"),
                *schedule.task_state_uses[1:],
            ),
        )
        hbm_as_sram = _recreate_schedule(
            schedule,
            task_buffer_uses=(
                replace(first_buffer_use, binding_id=first_hbm_ref),
                *schedule.task_buffer_uses[1:],
            ),
        )
        collision_binding = replace(
            schedule.buffer_bindings[0],
            id=first_hbm_ref,
        )
        collision = _recreate_schedule(
            schedule,
            buffer_bindings=(
                collision_binding,
                *schedule.buffer_bindings[1:],
            ),
            task_buffer_uses=tuple(
                replace(use, binding_id=first_hbm_ref)
                if use.binding_id == schedule.buffer_bindings[0].id
                else use
                for use in schedule.task_buffer_uses
            ),
        )
        die = next(
            item for item in graph.fabric.dies if item.id == dag.die_id
        )
        moved = schedule.placements[0]
        other_core = die.cores[1].runtime_core_id
        heterogeneous = _recreate_schedule(
            schedule,
            placements=(
                replace(moved, core_id=other_core),
                *schedule.placements[1:],
            ),
            core_orders=(
                CoreOrder(other_core, (moved.task_id,)),
                CoreOrder(
                    schedule.core_orders[0].core_id,
                    schedule.core_orders[0].task_ids[1:],
                ),
            ),
        )
        cases = (
            ("direction", wrong_direction, dag),
            ("unknown_hbm", unknown_hbm, dag),
            ("hbm_as_sram", hbm_as_sram, dag),
            ("namespace_collision", collision, dag),
            ("heterogeneous_core", heterogeneous, dag),
        )
        for name, forged, forged_dag in cases:
            with self.subTest(case=name), self.assertRaises(SchemaError):
                forged.validate_against(forged_dag, graph)

        staging = dag.state_staging_values[0]
        comp = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.COMP
        )
        assert comp.compute is not None
        original_input_id = comp.compute.inputs[0].value_id
        forged_comp = replace(
            comp,
            read_values=(staging.id,),
            compute=replace(
                comp.compute,
                inputs=(
                    replace(
                        comp.compute.inputs[0],
                        value_id=staging.id,
                    ),
                ),
            ),
        )
        forged_staging = replace(
            staging,
            consumer_tasks=staging.consumer_tasks + (comp.id,),
        )
        dag_fields = dag._semantic_key()
        dag_fields.update(
            tasks=tuple(
                forged_comp if task.id == comp.id else task
                for task in dag.tasks
            ),
            values=tuple(
                replace(
                    value,
                    consumer_tasks=tuple(
                        task_id
                        for task_id in value.consumer_tasks
                        if task_id != comp.id
                    ),
                )
                if value.id == original_input_id
                else value
                for value in dag.values
            ),
            state_staging_values=tuple(
                forged_staging if value.id == staging.id else value
                for value in dag.state_staging_values
            ),
        )
        forged_dag = IntraDieDAG.create(
            producer_pass=dag.producer_pass,
            **dag_fields,
        )
        opaque_schedule = _recreate_schedule(
            schedule,
            dag_id=forged_dag.id,
        )
        with self.subTest(case="attention_staging"), self.assertRaisesRegex(
            SchemaError, "opaque state staging"
        ):
            opaque_schedule.validate_against(forged_dag, graph)


if __name__ == "__main__":
    unittest.main()
