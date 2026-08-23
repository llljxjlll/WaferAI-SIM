from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.moe_swizzle_abi import (
    allocate_moe_swizzle_core_address_abi,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleActionKind
from llm.frontend.wafer_frontend.schema.swizzle_moe import MoeHardwareFacts
from llm.frontend.wafer_frontend.schema.swizzle_moe_placement import (
    MoeSemanticWorkItem, MoeSemanticWorkKey, build_moe_work_owner_map,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_ir2 import (
    MoeSwizzleIr2Buffer,
    MoeSwizzleIr2BufferUse,
    MoeSwizzleIr2Flow,
    MoeSwizzleIr2PacketSlice,
    MoeSwizzleIr2Projection,
    MoeSwizzleIr2Task,
    MoeSwizzleIr2Value,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_operand_abi import (
    build_moe_swizzle_operand_abi,
)
from llm.test.frontend.integration.lite_moe_dp4_cases import (
    LiteMoeDp4Mode,
    build_lite_moe_dp4_case,
)


def _task(
    task_id: str,
    rank: int,
    kind: SwizzleActionKind,
    pipeline: int,
    *,
    deps: tuple[str, ...] = (),
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    uses: tuple[MoeSwizzleIr2BufferUse, ...] = (),
    peer: int | None = None,
    flow: str | None = None,
    packet: str | None = None,
    stage: int | None = None,
    tile: int | None = None,
    route: str | None = None,
    assignment: str | None = None,
    pivot: int | None = None,
) -> MoeSwizzleIr2Task:
    return MoeSwizzleIr2Task(
        id=task_id,
        rank=rank,
        die_id=rank,
        kind=kind,
        work_role="comp" if kind is SwizzleActionKind.COMP else "transport",
        deps=deps,
        read_value_refs=reads,
        write_value_refs=writes,
        buffer_uses=uses,
        assignment_refs=(assignment or f"assignment.r{rank}.p{pipeline}",),
        expert_index=rank,
        tile_index=pipeline if tile is None else tile,
        n_block=0,
        packet_ref=packet,
        stage=stage,
        pivot_rank=pivot,
        original_action_refs=(f"original.r{rank}.p{pipeline}.{kind.value}",),
        pipeline_index=pipeline,
        buffer_slot=None,
        buffer_family=None,
        peer_rank=peer,
        flow_ref=flow,
        route_ref=route,
        logical_bytes=64 if kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV) else 0,
        flops=64 if kind is SwizzleActionKind.COMP else 32 if kind is SwizzleActionKind.REDUCE else 0,
        matmul_m=1 if kind is SwizzleActionKind.COMP else None,
        matmul_n=8 if kind is SwizzleActionKind.COMP else None,
        matmul_k=4 if kind is SwizzleActionKind.COMP else None,
        dtype=DType.FP16 if kind is SwizzleActionKind.COMP else None,
        accumulation_dtype=DType.FP32 if kind is SwizzleActionKind.COMP else None,
    )


def _value(
    value_ref: str,
    rank: int,
    producer: str | None,
    consumers: tuple[str, ...],
    buffer_ref: str | None = None,
    terminal_ref: str | None = None,
) -> MoeSwizzleIr2Value:
    return MoeSwizzleIr2Value(
        value_ref, rank, value_ref, (32,), "flat_fp16", DType.FP16, 0, 64,
        producer, consumers, buffer_ref, terminal_ref, producer is None, False,
    )


def _route(ir1: IR1, source: int, destination: int) -> object:
    return next(
        route
        for group in ir1.groups
        for route in group.embedding.routes
        if (route.source_rank, route.destination_rank) == (source, destination)
    )


def _ring_projection(ir1: IR1) -> MoeSwizzleIr2Projection:
    tasks = []
    values = []
    buffers = []
    flows = []
    terminals = []
    for rank in range(4):
        previous = (rank - 1) % 4
        following = (rank + 1) % 4
        recv_refs = []
        for pipeline in range(2):
            packet_out = f"packet.r{rank}.p{pipeline}"
            packet_in = f"packet.r{previous}.p{pipeline}"
            flow_out = f"flow.r{rank}.to{following}.p{pipeline}"
            flow_in = f"flow.r{previous}.to{rank}.p{pipeline}"
            pack = f"task.r{rank}.p{pipeline}.pack"
            send = f"task.r{rank}.p{pipeline}.send"
            recv = f"task.r{rank}.p{pipeline}.recv"
            wait = f"task.r{rank}.p{pipeline}.wait"
            gemm = f"task.r{rank}.p{pipeline}.gemm"
            input_ref = f"value.r{rank}.p{pipeline}.input"
            pack_weight_ref = f"value.r{rank}.p{pipeline}.pack_weight"
            gemm_weight_ref = f"value.r{rank}.p{pipeline}.gemm_weight"
            send_ref = f"value.r{rank}.p{pipeline}.send"
            recv_ref = f"value.r{rank}.p{pipeline}.recv"
            output_ref = f"value.r{rank}.p{pipeline}.combined"
            terminal = f"terminal.r{rank}.p{pipeline}.combined"
            slot = pipeline % 2
            use = MoeSwizzleIr2BufferUse(f"buffer.r{rank}.recv", slot)
            route_out = _route(ir1, rank, following)
            route_in = _route(ir1, previous, rank)
            assignment_out = f"assignment.r{rank}.p{pipeline}"
            assignment_in = f"assignment.r{previous}.p{pipeline}"
            tasks.extend((
                _task(pack, rank, SwizzleActionKind.COMP, pipeline, reads=(input_ref, pack_weight_ref), writes=(send_ref,), assignment=assignment_out),
                _task(send, rank, SwizzleActionKind.SEND, pipeline, deps=(pack,), reads=(send_ref,), peer=following, flow=flow_out, packet=packet_out, stage=0, route=route_out.id, assignment=assignment_out),
                _task(recv, rank, SwizzleActionKind.RECV, pipeline, writes=(recv_ref,), uses=(use,), peer=previous, flow=flow_in, packet=packet_in, stage=0, route=route_in.id, assignment=assignment_in),
                _task(wait, rank, SwizzleActionKind.WAIT, pipeline, deps=(recv,), flow=flow_in, packet=packet_in, stage=0, assignment=assignment_in),
                _task(gemm, rank, SwizzleActionKind.COMP, pipeline, deps=(wait,), reads=(recv_ref, gemm_weight_ref), writes=(output_ref,), uses=(use,), assignment=assignment_in),
            ))
            values.extend((
                _value(input_ref, rank, None, (pack,)),
                _value(pack_weight_ref, rank, None, (pack,)),
                _value(gemm_weight_ref, rank, None, (gemm,)),
                _value(send_ref, rank, pack, (send,)),
                _value(recv_ref, rank, recv, (gemm,), f"buffer.r{rank}.recv"),
                _value(output_ref, rank, gemm, (), terminal_ref=terminal),
            ))
            recv_refs.append(recv_ref)
            terminals.append(terminal)
        buffers.append(MoeSwizzleIr2Buffer(rank, f"buffer.r{rank}.recv", tuple(recv_refs), 64, 2))
    # Flows are assembled after every endpoint task exists.
    for source in range(4):
        destination = (source + 1) % 4
        for pipeline in range(2):
            route = _route(ir1, source, destination)
            assignment = f"assignment.r{source}.p{pipeline}"
            flows.append(MoeSwizzleIr2Flow(
                f"flow.r{source}.to{destination}.p{pipeline}",
                f"packet.r{source}.p{pipeline}",
                0,
                None,
                source,
                destination,
                source,
                destination,
                route.id,
                tuple(route.die_path),
                64,
                (MoeSwizzleIr2PacketSlice(assignment, 0, 0, 64),),
                f"task.r{source}.p{pipeline}.send",
                f"task.r{destination}.p{pipeline}.recv",
                f"task.r{destination}.p{pipeline}.wait",
            ))
    return MoeSwizzleIr2Projection.create(
        source_execution_id="lite_moe_dp4.production.execution",
        source_overlay_id="moe_swizzle.overlay.test",
        tasks=tuple(tasks),
        values=tuple(values),
        buffers=tuple(buffers),
        flows=tuple(flows),
        terminal_refs=tuple(terminals),
        endpoint_session_capacity=2,
    )


def _binary_reduce_projection() -> MoeSwizzleIr2Projection:
    rank = 0
    first_task, accumulator_task, reduce_task = "task.first", "task.acc", "task.reduce"
    first, accumulator, output = "value.first", "value.acc", "value.output"
    use = MoeSwizzleIr2BufferUse("buffer.reduce", 0)
    tasks = (
        _task(first_task, rank, SwizzleActionKind.COMP, 0, reads=("value.first.lhs", "value.first.weight"), writes=(first,), uses=(use,), tile=0),
        _task(accumulator_task, rank, SwizzleActionKind.COMP, 0, reads=("value.acc.lhs", "value.acc.weight"), writes=(accumulator,), uses=(use,), tile=0),
        _task(reduce_task, rank, SwizzleActionKind.REDUCE, 0, deps=(first_task, accumulator_task), reads=(first, accumulator), writes=(output,), uses=(use,), tile=0),
    )
    values = (
        _value("value.first.lhs", rank, None, (first_task,)),
        _value("value.first.weight", rank, None, (first_task,)),
        _value("value.acc.lhs", rank, None, (accumulator_task,)),
        _value("value.acc.weight", rank, None, (accumulator_task,)),
        _value(first, rank, first_task, (reduce_task,), "buffer.reduce"),
        _value(accumulator, rank, accumulator_task, (reduce_task,), "buffer.reduce"),
        _value(output, rank, reduce_task, (), "buffer.reduce", "terminal.reduce"),
    )
    return MoeSwizzleIr2Projection.create(
        source_execution_id="execution.reduce",
        source_overlay_id="overlay.reduce",
        tasks=tasks,
        values=values,
        buffers=(MoeSwizzleIr2Buffer(rank, "buffer.reduce", (first, accumulator, output), 64, 1),),
        flows=(),
        terminal_refs=("terminal.reduce",),
        endpoint_session_capacity=2,
    )


class MoeSwizzleMulticoreAbiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ir1 = build_lite_moe_dp4_case(LiteMoeDp4Mode.INFER).forward.n4.graph

    def test_real_multicore_placement_and_transport_affinity(self) -> None:
        projection = _ring_projection(self.ir1)
        abi = allocate_moe_swizzle_core_address_abi(self.ir1, projection)
        abi.validate_against(self.ir1, projection)
        binding = {item.task_ref: item for item in abi.task_bindings}
        real = {
            (die.id, core.local_core_id)
            for die in self.ir1.fabric.dies for core in die.cores
        }
        self.assertTrue(all((item.logical_core.die_id, item.logical_core.local_core_id) in real for item in abi.task_bindings))
        for rank in range(4):
            comp_cores = {
                binding[task.id].logical_core.local_core_id
                for task in projection.tasks
                if task.rank == rank and task.kind is SwizzleActionKind.COMP
            }
            self.assertGreaterEqual(len(comp_cores), 2)
            for pipeline in range(2):
                self.assertEqual(binding[f"task.r{rank}.p{pipeline}.pack"].logical_core, binding[f"task.r{rank}.p{pipeline}.send"].logical_core)
                self.assertEqual(binding[f"task.r{rank}.p{pipeline}.recv"].logical_core, binding[f"task.r{rank}.p{pipeline}.wait"].logical_core)
                self.assertEqual(binding[f"task.r{rank}.p{pipeline}.wait"].logical_core, binding[f"task.r{rank}.p{pipeline}.gemm"].logical_core)
        recv_roots = [item for item in abi.storage_roots if ".recv" in item.buffer_ref]
        self.assertEqual({item.slot for item in recv_roots}, {0, 1})

    def test_one_core_die_is_legal_degradation(self) -> None:
        fabric = self.ir1.fabric
        dies = tuple(
            replace(
                die,
                noc_grid=(1, 1),
                cores=(replace(die.cores[0], runtime_core_id=die.id, noc_coord=(0, 0)),),
                ports=tuple(replace(port, noc_coord=(0, 0)) for port in die.ports),
            )
            for die in fabric.dies
        )
        source = self.ir1
        ir1 = IR1.create(
            producer_pass=source.producer_pass,
            source_ir0_id=source.source_ir0_id,
            profile=source.profile,
            fabric=replace(fabric, dies=dies),
            instances=source.instances,
            groups=source.groups,
            nodes=source.nodes,
            values=source.values,
            edges=source.edges,
            fusion_candidates=source.fusion_candidates,
            fused_op_skeletons=source.fused_op_skeletons,
            cross_routes=source.cross_routes,
            state_accesses=source.state_accesses,
            persistent_state_manifest=source.persistent_state_manifest,
            instance_profiles=source.instance_profiles,
            node_profiles=source.node_profiles,
            pd_plan_id=source.pd_plan_id,
        )
        ir1.validate()
        abi = allocate_moe_swizzle_core_address_abi(ir1, _ring_projection(ir1))
        for rank in range(4):
            self.assertEqual(
                {item.logical_core.local_core_id for item in abi.task_bindings if item.rank == rank},
                {dies[rank].cores[0].local_core_id},
            )

    def test_owner_affinity_is_stable_under_work_set_growth(self) -> None:
        facts = MoeHardwareFacts.from_fabric(self.ir1.fabric)
        original = (
            MoeSemanticWorkItem(0, MoeSemanticWorkKey(
                "comp", None, 0, 2, 0, "gate", 2,
            )),
            MoeSemanticWorkItem(0, MoeSemanticWorkKey(
                "comp", None, 0, 3, 0, "up", 3,
            )),
        )
        before = {
            item.work_key: item.runtime_core_id
            for item in build_moe_work_owner_map(facts, original)
        }
        extra = MoeSemanticWorkItem(0, MoeSemanticWorkKey(
            "comp", None, 0, 0, 0, "gate", 0,
        ))
        after = {
            item.work_key: item.runtime_core_id
            for item in build_moe_work_owner_map(facts, (extra, *original))
        }
        self.assertEqual(
            before, {key: after[key] for key in before},
        )
        self.assertEqual(len(set(before.values())), 2)

    def test_cross_core_value_without_local_copy_is_rejected(self) -> None:
        projection = _ring_projection(self.ir1)
        tasks = list(projection.tasks)
        target = next(index for index, item in enumerate(tasks) if item.id == "task.r0.p0.gemm")
        tasks[target] = replace(tasks[target], tile_index=1)
        tampered = MoeSwizzleIr2Projection.create(
            source_execution_id=projection.source_execution_id,
            source_overlay_id=projection.source_overlay_id,
            tasks=tuple(tasks),
            values=projection.values,
            buffers=projection.buffers,
            flows=projection.flows,
            terminal_refs=projection.terminal_refs,
            endpoint_session_capacity=projection.endpoint_session_capacity,
        )
        with self.assertRaisesRegex(SchemaError, "crosses cores without an explicit LOCAL_COPY"):
            allocate_moe_swizzle_core_address_abi(self.ir1, tampered)

    def test_binary_reduce_is_contiguous_and_in_place(self) -> None:
        projection = _binary_reduce_projection()
        abi = allocate_moe_swizzle_core_address_abi(self.ir1, projection)
        by_value = {item.value_ref: item for item in abi.value_bindings}
        self.assertEqual(by_value["value.first"].address + 64, by_value["value.acc"].address)
        self.assertEqual(by_value["value.output"].address, by_value["value.acc"].address)

    def test_double_buffer_slot_mismatch_fails_closed(self) -> None:
        projection = _ring_projection(self.ir1)
        tasks = list(projection.tasks)
        target = next(index for index, item in enumerate(tasks) if item.id == "task.r0.p1.recv")
        tasks[target] = replace(tasks[target], buffer_uses=(MoeSwizzleIr2BufferUse("buffer.r0.recv", 0),))
        with self.assertRaisesRegex(SchemaError, "fixed pipeline slot"):
            MoeSwizzleIr2Projection.create(
                source_execution_id=projection.source_execution_id,
                source_overlay_id=projection.source_overlay_id,
                tasks=tuple(tasks),
                values=projection.values,
                buffers=projection.buffers,
                flows=projection.flows,
                terminal_refs=projection.terminal_refs,
                endpoint_session_capacity=projection.endpoint_session_capacity,
            )

    def test_typed_value_extent_tamper_is_rejected(self) -> None:
        projection = _ring_projection(self.ir1)
        values = list(projection.values)
        values[0] = replace(values[0], shape=(16,))
        with self.assertRaisesRegex(SchemaError, "typed shape extent"):
            MoeSwizzleIr2Projection.create(
                source_execution_id=projection.source_execution_id,
                source_overlay_id=projection.source_overlay_id,
                tasks=projection.tasks,
                values=tuple(values),
                buffers=projection.buffers,
                flows=projection.flows,
                terminal_refs=projection.terminal_refs,
                endpoint_session_capacity=projection.endpoint_session_capacity,
            )

    def test_comp_flop_tamper_is_rejected(self) -> None:
        projection = _ring_projection(self.ir1)
        tasks = list(projection.tasks)
        target = next(index for index, item in enumerate(tasks) if item.kind is SwizzleActionKind.COMP)
        tasks[target] = replace(tasks[target], flops=tasks[target].flops + 1)
        with self.assertRaisesRegex(SchemaError, "exact typed MATMUL"):
            MoeSwizzleIr2Projection.create(
                source_execution_id=projection.source_execution_id,
                source_overlay_id=projection.source_overlay_id,
                tasks=tuple(tasks),
                values=projection.values,
                buffers=projection.buffers,
                flows=projection.flows,
                terminal_refs=projection.terminal_refs,
                endpoint_session_capacity=projection.endpoint_session_capacity,
            )

    def test_flow_assignment_slice_tamper_is_rejected(self) -> None:
        projection = _ring_projection(self.ir1)
        flows = list(projection.flows)
        flow = flows[0]
        flows[0] = replace(
            flow,
            assignment_slices=(
                replace(flow.assignment_slices[0], assignment_ref="assignment.tampered"),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "flow provenance mismatch"):
            MoeSwizzleIr2Projection.create(
                source_execution_id=projection.source_execution_id,
                source_overlay_id=projection.source_overlay_id,
                tasks=projection.tasks,
                values=projection.values,
                buffers=projection.buffers,
                flows=tuple(flows),
                terminal_refs=projection.terminal_refs,
                endpoint_session_capacity=projection.endpoint_session_capacity,
            )

    def test_flow_endpoint_and_route_tamper_are_rejected(self) -> None:
        projection = _ring_projection(self.ir1)
        flows = list(projection.flows)
        flows[0] = replace(flows[0], destination_rank=2)
        with self.assertRaisesRegex(SchemaError, "endpoint/wait closure"):
            MoeSwizzleIr2Projection.create(
                source_execution_id=projection.source_execution_id,
                source_overlay_id=projection.source_overlay_id,
                tasks=projection.tasks,
                values=projection.values,
                buffers=projection.buffers,
                flows=tuple(flows),
                terminal_refs=projection.terminal_refs,
                endpoint_session_capacity=projection.endpoint_session_capacity,
            )
        projection = _ring_projection(self.ir1)
        tasks = tuple(
            replace(task, route_ref="route.tampered")
            if task.flow_ref == projection.flows[0].id and task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)
            else task
            for task in projection.tasks
        )
        flows = (replace(projection.flows[0], route_ref="route.tampered"),) + projection.flows[1:]
        tampered = MoeSwizzleIr2Projection.create(
            source_execution_id=projection.source_execution_id,
            source_overlay_id=projection.source_overlay_id,
            tasks=tasks,
            values=projection.values,
            buffers=projection.buffers,
            flows=flows,
            terminal_refs=projection.terminal_refs,
            endpoint_session_capacity=projection.endpoint_session_capacity,
        )
        with self.assertRaisesRegex(SchemaError, "unknown PairRoute"):
            allocate_moe_swizzle_core_address_abi(self.ir1, tampered)

    def test_operand_abi_is_exact_typed_projection_quotient(self) -> None:
        projection = _ring_projection(self.ir1)
        operand_abi = build_moe_swizzle_operand_abi(projection)
        operand_abi.validate_against(projection)
        self.assertEqual(len(operand_abi.matmuls), 16)
        self.assertEqual(len(operand_abi.dtes), 16)
        self.assertTrue(all(item.payload_bits == item.logical_bytes * 8 for item in operand_abi.dtes))

    def test_flow_pivot_tamper_is_rejected(self) -> None:
        projection = _ring_projection(self.ir1)
        flows = (replace(projection.flows[0], pivot_rank=2),) + projection.flows[1:]
        with self.assertRaisesRegex(SchemaError, "flow provenance mismatch"):
            MoeSwizzleIr2Projection.create(
                source_execution_id=projection.source_execution_id,
                source_overlay_id=projection.source_overlay_id,
                tasks=projection.tasks,
                values=projection.values,
                buffers=projection.buffers,
                flows=flows,
                terminal_refs=projection.terminal_refs,
                endpoint_session_capacity=projection.endpoint_session_capacity,
            )

    def test_operand_extent_tamper_is_rejected(self) -> None:
        projection = _ring_projection(self.ir1)
        operand_abi = build_moe_swizzle_operand_abi(projection)
        operands = (replace(operand_abi.operands[0], byte_extent=32),) + operand_abi.operands[1:]
        with self.assertRaisesRegex(SchemaError, "typed shape"):
            type(operand_abi).create(
                source_projection_id=projection.id,
                operands=operands,
                matmuls=operand_abi.matmuls,
                dtes=operand_abi.dtes,
                reductions=operand_abi.reductions,
            )


if __name__ == "__main__":
    unittest.main()
