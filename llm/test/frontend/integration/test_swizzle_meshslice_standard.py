from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    decide_meshslice_2d_standard,
    link_meshslice_2d_standard_program,
)
from llm.frontend.wafer_frontend.passes import (
    place_meshslice_2d_ir1,
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.passes.discover_fusion import (
    with_discovered_fusion_candidates,
)
from llm.frontend.wafer_frontend.policies.swizzle.meshslice_2d import (
    generate_meshslice_2d_drafts,
    meshslice_candidate_scale,
    meshslice_execution_mode,
    MeshSliceExecutionMode,
)
from llm.frontend.wafer_frontend.policies.swizzle_topo import (
    SwizzlePlanner,
)
from llm.frontend.wafer_frontend.schema.common import (
    DType,
    MeshAxisName,
    ProfileKey,
    Sharding,
    TensorValue,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    PlacementSpec,
    PlacementStrategy,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    CollectiveRole,
    CollectiveWorkload,
    DeviceMesh,
    EdgeKind,
    EffectKind,
    FusionPattern,
    GemmPartition,
    GemmWorkload,
    GraphEdge,
    IR0,
    JobKind,
    LogicalInstance,
    LogicalNode,
    LogicalRole,
    MeshAxis,
    NodeEffects,
    NodeMath,
    NumericalPolicy,
    OpKind,
    OpPhase,
    ParallelAxes,
)
from llm.frontend.wafer_frontend.schema.ir1 import (
    IR1,
)
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleConstraints,
    SwizzleDecision,
    SwizzleEfficiencyPoint,
    SwizzleHardwareProfile,
)
from llm.frontend.wafer_frontend.schema.swizzle_ir2 import (
    SwizzleIr2BufferRole,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x2.json"
_ZERO_SHA = "0" * 64


def _source(
    *,
    exact_sharding: bool = True,
    rows: int = 2,
    columns: int = 2,
    m_factor: int = 32,
    n_factor: int = 32,
    k_factor: int = 16,
) -> IR0:
    ranks = rows * columns
    m = rows * m_factor
    n = columns * n_factor
    k = ranks * k_factor
    logical_tensor_bytes = m * k * 2
    rank_input_bytes = logical_tensor_bytes // ranks
    rank_payload_bytes = logical_tensor_bytes - rank_input_bytes
    mesh = DeviceMesh(
        "meshslice.dp_tp",
        (
            MeshAxis(MeshAxisName.DP, rows),
            MeshAxis(MeshAxisName.TP, columns),
        ),
    )
    instance = LogicalInstance(
        id="MS0",
        role=LogicalRole.PREFILL,
        replicas=1,
        parallel=ParallelAxes(
            tp=columns,
            sp=False,
            dp=rows,
            pp=1,
            ep=1,
        ),
        meshes=(mesh,),
    )
    gather = LogicalNode(
        id="meshslice.ag",
        instance_id=instance.id,
        kind=OpKind.COLLECTIVE,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref=mesh.id,
        inputs=("meshslice.ag.input",),
        outputs=("meshslice.lhs",),
        workload=CollectiveWorkload(
            collective=CollectiveKind.ALL_GATHER,
            reduce_op=None,
            mesh_axes=(MeshAxisName.DP, MeshAxisName.TP),
            participant_count=ranks,
            reduction_mesh_axes=(),
            scatter_tensor_axis=None,
            gather_tensor_axis=1,
            logical_tensor_bytes=logical_tensor_bytes,
            rank_input_bytes=rank_input_bytes,
            rank_output_bytes=logical_tensor_bytes,
            rank_logical_payload_bytes=rank_payload_bytes,
            group_logical_payload_bytes=rank_payload_bytes * ranks,
            dtype=DType.FP16,
            role=CollectiveRole.ACTIVATION,
            input_layout="MK_dp_tp_shard",
            output_layout="MK_dp_tp",
        ),
        math=NodeMath(DType.FP32, NumericalPolicy.BITWISE),
        effects=NodeEffects(EffectKind.PURE, None, None),
        impl_ref="collective_derived",
    )
    gemm = LogicalNode(
        id="meshslice.gemm",
        instance_id=instance.id,
        kind=OpKind.GEMM,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref=mesh.id,
        inputs=("meshslice.lhs", "meshslice.rhs"),
        outputs=("meshslice.output",),
        workload=GemmWorkload(
            logical_shape=(m, n, k),
            rank_shape=(m, n // columns, k),
            partition=GemmPartition.COLUMN_PARALLEL,
            dtype=DType.FP16,
        ),
        math=NodeMath(DType.FP32, NumericalPolicy.BITWISE),
        effects=NodeEffects(EffectKind.PURE, None, None),
        impl_ref="matmul_forward",
    )
    two_d = (MeshAxisName.DP, MeshAxisName.TP)
    one_d = (None, MeshAxisName.TP)
    dim_map = two_d if exact_sharding else one_d
    values = (
        TensorValue(
            "meshslice.ag.input",
            (m, k // ranks),
            DType.FP16,
            "MK_dp_tp_shard",
            Sharding(mesh.id, dim_map, ()),
            None,
            (gather.id,),
            None,
        ),
        TensorValue(
            "meshslice.lhs",
            (m, k),
            DType.FP16,
            "MK_dp_tp",
            Sharding(mesh.id, dim_map, ()),
            gather.id,
            (gemm.id,),
            None,
        ),
        TensorValue(
            "meshslice.rhs",
            (k, n),
            DType.FP16,
            "KN_dp_tp",
            Sharding(mesh.id, dim_map, ()),
            None,
            (gemm.id,),
            None,
        ),
        TensorValue(
            "meshslice.output",
            (m, n),
            DType.FP16,
            "MN_dp_tp",
            Sharding(mesh.id, dim_map, ()),
            gemm.id,
            (),
            None,
        ),
    )
    graph = IR0.create(
        producer_pass="meshslice_2d_source",
        job=JobKind.INFER,
        instances=(instance,),
        nodes=(gather, gemm),
        values=values,
        edges=(
            GraphEdge(
                "meshslice.edge",
                EdgeKind.DATA,
                gather.id,
                gemm.id,
                "meshslice.lhs",
            ),
        ),
        fusion_candidates=(),
        profile=ProfileKey(m, 0, 1, m, m, 0, None),
    )
    graph.validate("meshslice_source")
    return with_discovered_fusion_candidates(graph)


def _context(
    *,
    rows: int = 2,
    columns: int = 2,
) -> PlacementContext:
    raw = (
        json.loads(_HARDWARE.read_text(encoding="utf-8"))
        if (rows, columns) == (2, 2)
        else minimal_hardware(columns, rows, sram_bytes=16 << 20)
    )
    return PlacementContext.create(
        producer_pass="meshslice_2d_test",
        fabric=physical_fabric_from_data(raw, path="meshslice.hardware"),
        placement=PlacementSpec(PlacementStrategy.COMPACT, ()),
        hbm_address_spaces=hbm_address_spaces_from_data(
            raw, path="meshslice.hardware"
        ),
    )


def _planner(
    *,
    max_actions: int = 4096,
    max_buffers: int = 256,
    efficient_tile_floor: tuple[int, int, int] = (8, 8, 8),
    max_chunk_count: int = 32,
    sram_budget_bytes: int = 1 << 20,
) -> SwizzlePlanner:
    return SwizzlePlanner(
        hardware_profile=SwizzleHardwareProfile.create(
            peak_flops_per_cycle=1024.0,
            confidence_fraction=0.05,
            efficiency_points=(
                SwizzleEfficiencyPoint(32, 32, 32, 0.9),
            ),
            dte_launch_cycles=2,
            dte_sync_cycles=1,
            hop_latency_cycles=1,
            lane_bytes_per_cycle=32.0,
            max_inflight_dte=2,
            min_transfer_bytes=16,
            efficient_tile_floor=efficient_tile_floor,
            sram_budget_bytes=sram_budget_bytes,
            double_buffer_supported=True,
        ),
        constraints=SwizzleConstraints(
            allowed_algorithms=(
                SwizzleAlgorithm.MESHSLICE_2D_OS,
                SwizzleAlgorithm.UNFUSED,
            ),
            max_candidates=32,
            max_actions=max_actions,
            max_buffers=max_buffers,
            max_chunk_count=max_chunk_count,
            allow_unroll_two=True,
        ),
        generators=(generate_meshslice_2d_drafts,),
    )


def _case(
    *,
    exact_sharding: bool = True,
    rows: int = 2,
    columns: int = 2,
    planner: SwizzlePlanner | None = None,
) -> tuple[IR1, SwizzleDecision]:
    ir1 = place_meshslice_2d_ir1(
        _source(
            exact_sharding=exact_sharding,
            rows=rows,
            columns=columns,
        ),
        _context(rows=rows, columns=columns),
    )
    planner = planner or _planner()
    decision = decide_meshslice_2d_standard(
        ir1,
        ir1.fused_op_skeletons[0],
        planner.hardware_profile,
        planner.constraints,
    )
    return ir1, decision


class SwizzleMeshSliceStandardTest(unittest.TestCase):
    def test_all_100_shapes_have_one_exact_execution_mode(self) -> None:
        counts = Counter(
            meshslice_execution_mode(rows, columns)
            for rows in range(1, 11)
            for columns in range(1, 11)
        )
        self.assertEqual(
            counts,
            Counter({
                MeshSliceExecutionMode.FULL_2D: 81,
                MeshSliceExecutionMode.ROW_ONLY: 9,
                MeshSliceExecutionMode.COLUMN_ONLY: 9,
                MeshSliceExecutionMode.LOCAL: 1,
            }),
        )
        for rows in range(1, 11):
            for columns in range(1, 11):
                ranks = rows * columns
                peers = rows + columns - 2
                # The fixed chunk=1 estimator is the production sweep budget;
                # it covers LOCAL through 10x10 without constructing 100 huge
                # stable-artifact payloads in every unit-test run.
                self.assertEqual(
                    meshslice_candidate_scale(
                        FusionPattern.AG_GEMM,
                        rows=rows,
                        columns=columns,
                        slice_count=1,
                    ),
                    (ranks * (3 * peers + 1), 3 * ranks),
                )

    def test_all_four_modes_reach_standard_manifest(self) -> None:
        expected_modes = {
            (1, 1): MeshSliceExecutionMode.LOCAL,
            (1, 3): MeshSliceExecutionMode.ROW_ONLY,
            (3, 1): MeshSliceExecutionMode.COLUMN_ONLY,
            (2, 3): MeshSliceExecutionMode.FULL_2D,
        }
        for (rows, columns), mode in expected_modes.items():
            with self.subTest(rows=rows, columns=columns):
                ranks = rows * columns
                expected_actions = ranks * (
                    3 * (rows + columns - 2) + 1
                )
                ir1, decision = _case(
                    rows=rows,
                    columns=columns,
                    planner=_planner(
                        max_actions=expected_actions,
                        max_buffers=max(3 * ranks, 4),
                        max_chunk_count=1,
                        sram_budget_bytes=16 << 20,
                    ),
                )
                self.assertIs(
                    decision.baseline.algorithm,
                    SwizzleAlgorithm.UNFUSED,
                )
                candidate = next(
                    item
                    for item in decision.ranked_candidates
                    if item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
                )
                source, audit = link_meshslice_2d_standard_program(
                    ir1, decision, candidate_ref=candidate.id
                )
                source.validate_against()
                self.assertEqual(
                    (
                        audit.mode,
                        audit.ranks,
                        audit.rows,
                        audit.columns,
                        audit.row_flows,
                        audit.column_flows,
                        len(source.projection.flows),
                    ),
                    (
                        mode,
                        ranks,
                        rows,
                        columns,
                        ranks * (columns - 1),
                        ranks * (rows - 1),
                        ranks * (rows + columns - 2),
                    ),
                )
                program_io = build_timing_program_io(source, _ZERO_SHA)
                program_io.validate_against(source.manifest)
                self.assertEqual(len(program_io.output_probes), ranks)

    def test_slice_one_candidate_scales_to_acceptance_rectangles(self) -> None:
        for rows, columns in (
            (2, 10), (10, 2), (5, 6), (6, 5), (10, 10),
        ):
            with self.subTest(rows=rows, columns=columns):
                ranks = rows * columns
                expected_actions = ranks * (3 * (rows + columns - 2) + 1)
                _ir1, decision = _case(
                    rows=rows,
                    columns=columns,
                    planner=_planner(
                        max_actions=expected_actions,
                        max_buffers=512,
                        max_chunk_count=1,
                        sram_budget_bytes=16 << 20,
                    ),
                )
                candidate = next(
                    item
                    for item in decision.ranked_candidates
                    if item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
                )
                self.assertEqual(candidate.chunk_count, 1)
                self.assertEqual(
                    sum(
                        len(program.actions)
                        for program in candidate.rank_programs
                    ),
                    expected_actions,
                )
                self.assertEqual(
                    (
                        len(candidate.topology_witness.row_orders),
                        len(candidate.topology_witness.column_orders),
                    ),
                    (rows, columns),
                )

    def test_scale_budget_rejects_before_action_dag_construction(self) -> None:
        rows = columns = 10
        ranks = rows * columns
        expected_actions = ranks * (3 * (rows + columns - 2) + 1)
        with patch(
            "llm.frontend.wafer_frontend.policies.swizzle.meshslice_2d."
            "_programs_for_slice_count"
        ) as builder:
            _ir1, decision = _case(
                rows=rows,
                columns=columns,
                planner=_planner(
                    max_actions=expected_actions - 1,
                    max_buffers=512,
                    max_chunk_count=1,
                    sram_budget_bytes=16 << 20,
                ),
            )
        builder.assert_not_called()
        self.assertFalse(any(
            item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
            for item in decision.ranked_candidates
        ))

    def test_rectangular_mesh_reaches_standard_manifest(self) -> None:
        for rows, columns in ((2, 3), (3, 2)):
            with self.subTest(rows=rows, columns=columns):
                ir1, decision = _case(rows=rows, columns=columns)
                candidate = next(
                    item
                    for item in decision.ranked_candidates
                    if item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
                    and item.chunk_count == 1
                )
                source, audit = link_meshslice_2d_standard_program(
                    ir1,
                    decision,
                    candidate_ref=candidate.id,
                )
                ranks = rows * columns
                self.assertEqual(ir1.groups[0].logical_shape, (rows, columns))
                self.assertEqual(len(source.projection.rank_dags), ranks)
                self.assertEqual(len(source.fragment.core_streams), ranks)
                self.assertEqual(
                    (audit.ranks, audit.rows, audit.columns, audit.chunks),
                    (ranks, rows, columns, 1),
                )
                self.assertEqual(
                    (audit.row_flows, audit.column_flows),
                    (
                        ranks * (columns - 1),
                        ranks * (rows - 1),
                    ),
                )
                program_io = build_timing_program_io(source, _ZERO_SHA)
                program_io.validate_against(source.manifest)
                self.assertEqual(len(program_io.output_probes), ranks)
                for stream in source.fragment.core_streams:
                    recv_offsets = defaultdict(list)
                    for relocation in stream.address_relocations:
                        record = stream.records[relocation.record_index]
                        if (
                            record.opcode is RecordOpcode.DTE_RECV
                            and relocation.operand_id
                            is SemanticOperandId.DESTINATION_ADDRESS
                        ):
                            recv_offsets[relocation.symbol_ref].append(
                                relocation.addend
                            )
                    self.assertEqual(
                        sum(map(len, recv_offsets.values())),
                        rows + columns - 2,
                    )
                    self.assertTrue(all(
                        offset == 0
                        for offsets in recv_offsets.values()
                        for offset in offsets
                    ))
                    self.assertEqual(
                        len(recv_offsets), rows + columns - 2
                    )
                    self.assertTrue(all(len(offsets) == 1 for offsets in recv_offsets.values()))

    def test_two_by_two_reaches_exact_standard_manifest(self) -> None:
        ir1, decision = _case()
        candidate = next(
            item
            for item in decision.ranked_candidates
            if item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
            and item.chunk_count == 2
        )
        source, audit = link_meshslice_2d_standard_program(
            ir1, decision, candidate_ref=candidate.id
        )
        rebuilt, rebuilt_audit = link_meshslice_2d_standard_program(
            ir1, decision, candidate_ref=candidate.id
        )


        source.validate_against()
        self.assertEqual((source, audit), (rebuilt, rebuilt_audit))
        self.assertEqual(len(source.projection.rank_dags), 4)
        self.assertEqual(len(source.fragment.core_streams), 4)
        self.assertEqual(len(source.manifest.core_bindings), 4)
        self.assertEqual(len(source.manifest.envelope.active_cores), 4)
        self.assertEqual(len(source.manifest.envelope.terminal_cores), 4)
        self.assertEqual((audit.row_flows, audit.column_flows), (8, 8))
        self.assertEqual(audit.barrier_actions, 0)
        self.assertEqual(audit.event_records, 0)
        records = tuple(
            record
            for stream in source.fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(
            (
                len(records),
                len(source.fragment.claimed_action_ids),
                len(source.fragment.buffer_abi),
                len(source.fragment.runtime_symbols),
                len(source.fragment.program_symbols),
                len(source.manifest.input_digests),
                len(source.manifest.runtime_symbol_definitions),
                len(source.manifest.program_symbol_definitions),
                len(source.manifest.address_operand_bindings),
            ),
            (104, 56, 20, 64, 41, 9, 68, 41, 132),
        )
        self.assertEqual(
            Counter(record.opcode.name for record in records),
            Counter({
                "DTE_RECV": 16, "DTE_SEND": 16, "DTE_WAIT": 16,
                "MATMUL": 8, "SRAM_ALLOC_AT": 20, "SRAM_BIND": 8,
                "SRAM_FREE": 20,
            }),
        )
        self.assertEqual(
            len(source.manifest.envelope.start_events),
            4,
        )
        roots = tuple(
            item for item in source.fragment.buffer_abi
            if item.alias_of is None
        )
        self.assertEqual(
            Counter(item.ownership for item in roots),
            Counter({BufferOwnership.BORROWED: 16,
                     BufferOwnership.OWNED: 4}),
        )
        self.assertFalse(any(
            item.layout in {
                "swizzle_standard_terminal_root/v1",
                "swizzle_standard_terminal_subview/v1",
                "swizzle_standard_storage_root/v1",
                "swizzle_standard_storage_subview/v1",
            }
            for item in source.fragment.buffer_abi
        ))
        owned = tuple(
            item for item in roots
            if item.ownership is BufferOwnership.OWNED
        )
        projected_symbols = {
            value.id: value.symbolic_ref
            for dag in source.projection.rank_dags
            for value in dag.values
        }
        self.assertEqual(
            {projected_symbols[item.value_id] for item in owned},
            {f"buffer.meshslice.rank.{rank}.output" for rank in range(4)},
        )
        program_io = build_timing_program_io(source, _ZERO_SHA)
        program_io.validate_against(source.manifest)
        self.assertEqual(
            (len(program_io.initializations),
             len(program_io.output_probes), len(program_io.blobs)),
            (20, 4, 2),
        )
        self.assertEqual(
            sum(item.length_bytes for item in program_io.initializations),
            sum(item.size_bytes for item in roots),
        )
        self.assertEqual(
            sum(item.length_bytes for item in program_io.output_probes),
            sum(item.size_bytes for item in owned),
        )
        self.assertEqual(
            {item.target.buffer_abi_id for item in program_io.output_probes},
            {item.id for item in owned},
        )
        self.assertEqual(
            {item.target.runtime_core_id for item in program_io.output_probes},
            {binding.runtime_core_id
             for binding in source.manifest.core_bindings},
        )
        roles = {
            buffer.role
            for dag in source.projection.rank_dags
            for buffer in dag.buffers
        }
        self.assertIn(SwizzleIr2BufferRole.DOUBLE_BUFFER, roles)
        self.assertIn(SwizzleIr2BufferRole.LOOP_ACCUMULATOR, roles)
        for dag in source.projection.rank_dags:
            views_by_task = {
                task.id: tuple(
                    view for view in source.operand_abi.operands
                    if view.task_ref == task.id
                )
                for task in dag.tasks
            }
            for chunk in range(2):
                tasks = tuple(
                    item for item in dag.tasks if item.chunk_index == chunk
                )
                sends = tuple(
                    item for item in tasks
                    if item.kind is SwizzleActionKind.SEND
                )
                self.assertEqual(len(sends), 2)
                self.assertEqual(sends[0].deps, sends[1].deps)
                compute = next(
                    item for item in tasks
                    if item.kind is SwizzleActionKind.COMP
                )
                output_views = tuple(
                    view for view in views_by_task[compute.id]
                    if projected_symbols[view.value_ref].endswith(".output")
                )
                self.assertEqual(
                    tuple(view.use.value for view in output_views),
                    ("write",) if chunk == 0 else ("read", "write"),
                )

    def test_missing_2d_sharding_never_produces_candidate(self) -> None:
        _ir1, decision = _case(exact_sharding=False)
        self.assertFalse(
            any(
                item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
                for item in decision.ranked_candidates
            )
        )

    def test_chunk_four_keeps_lifecycle_constant_and_reuses_both_slots(self) -> None:
        ir1, decision = _case()
        candidate = next(
            item
            for item in decision.ranked_candidates
            if item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
            and item.chunk_count == 4
        )
        source, audit = link_meshslice_2d_standard_program(
            ir1, decision, candidate_ref=candidate.id
        )
        records = tuple(
            record
            for stream in source.fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(
            (audit.chunks, audit.root_buffers,
             audit.alloc_records, audit.free_records),
            (4, 20, 20, 20),
        )
        self.assertEqual(
            Counter(item.ownership for item in source.fragment.buffer_abi),
            Counter({BufferOwnership.BORROWED: 16,
                     BufferOwnership.OWNED: 4}),
        )
        self.assertEqual(
            (Counter(record.opcode.name for record in records)
                    ["SRAM_ALLOC_AT"],
             Counter(record.opcode.name for record in records)["SRAM_FREE"]),
            (20, 20),
        )
        self.assertGreater(len(records), 104)
        self.assertGreater(len(source.fragment.claimed_action_ids), 56)
        for dag in source.projection.rank_dags:
            double_buffers = {
                buffer.buffer_ref for buffer in dag.buffers
                if buffer.role is SwizzleIr2BufferRole.DOUBLE_BUFFER
            }
            self.assertEqual(
                {
                    use.slot
                    for task in dag.tasks
                    for use in task.buffer_uses
                    if use.buffer_ref in double_buffers
                },
                {0, 1},
            )

    def test_one_by_four_logical_group_fails_closed(self) -> None:
        ir1, decision = _case()
        candidate = next(
            item
            for item in decision.ranked_candidates
            if item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
            and item.chunk_count == 2
        )
        group = replace(
            ir1.groups[0],
            logical_shape=(1, 4),
            placements=tuple(
                replace(item, logical_coord=(0, item.rank))
                for item in ir1.groups[0].placements
            ),
        )
        forged = IR1.create(
            producer_pass=ir1.producer_pass,
            source_ir0_id=ir1.source_ir0_id,
            profile=ir1.profile,
            fabric=ir1.fabric,
            instances=ir1.instances,
            groups=(group,),
            nodes=ir1.nodes,
            values=ir1.values,
            edges=ir1.edges,
            fusion_candidates=ir1.fusion_candidates,
            fused_op_skeletons=ir1.fused_op_skeletons,
        )
        with self.assertRaisesRegex(SchemaError, "typed 2x2 physical group"):
            link_meshslice_2d_standard_program(
                forged, decision, candidate_ref=candidate.id
            )


if __name__ == "__main__":
    unittest.main()
