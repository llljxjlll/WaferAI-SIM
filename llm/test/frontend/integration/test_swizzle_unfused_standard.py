from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.swizzle_unfused import (
    _allocate_storage_intervals,
    allocate_unfused_comparison_core_abi,
    build_unfused_comparison_operand_abi,
    lower_unfused_comparison_opcodes,
)
from llm.frontend.wafer_frontend.lowering.swizzle_unfused_standard import (
    _validate_compute_wire_addresses,
    link_unfused_comparison_program,
)
from llm.frontend.wafer_frontend.passes.project_unfused_comparison import (
    _multi_rank_peer_waves,
    build_unfused_comparison_plan,
    project_unfused_comparison,
)
from llm.frontend.wafer_frontend.passes.unfused_comparison_program_io import (
    unfused_comparison_terminal_abi_ids,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.global_action import LogicalCoreRef
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleCandidate,
)
from llm.frontend.wafer_frontend.schema.swizzle_unfused_abi import (
    UnfusedComparisonCoreABI,
)

from swizzle_cases import build_swizzle_integration_cases
from swizzle_scale_cases import build_swizzle_scale_case, build_swizzle_scale_points
from test_swizzle_unfused_comparison import _flexible_rank_problem


def _build(case):
    plan = build_unfused_comparison_plan(
        case.partitioned_graph, case.decision.problem, case.decision.baseline,
    )
    projection = project_unfused_comparison(case.partitioned_graph, plan)
    core_abi = allocate_unfused_comparison_core_abi(
        case.partitioned_graph, plan, projection,
    )
    operand_abi = build_unfused_comparison_operand_abi(
        case.partitioned_graph, plan, projection,
    )
    lowered = lower_unfused_comparison_opcodes(plan, projection)
    return link_unfused_comparison_program(
        case.partitioned_graph, plan, projection, lowered, core_abi, operand_abi,
    )


class SwizzleUnfusedStandardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_swizzle_integration_cases()

    def test_three_closed_sources_are_exact_and_deterministic(self) -> None:
        expected = {
            FusionPattern.AG_GEMM: (24, (2, 4, 10), 8, 17, 32, 10),
            FusionPattern.GEMM_RS: (38, (6, 4, 16), 10, 27, 52, 12),
            FusionPattern.GEMM_AR: (46, (4, 4, 20), 24, 29, 50, 26),
        }
        for case in self.cases:
            with self.subTest(pattern=case.pattern.value):
                decision_before = case.decision
                source = _build(case)
                source.validate_against()
                self.assertEqual(source, _build(case))
                ownership = Counter(item.ownership for item in source.fragment.buffer_abi)
                counts = (
                    sum(len(stream.records) for stream in source.fragment.core_streams),
                    (
                        ownership[BufferOwnership.OWNED],
                        ownership[BufferOwnership.BORROWED],
                        ownership[BufferOwnership.ALIASED],
                    ),
                    len(source.fragment.runtime_symbols),
                    len(source.fragment.program_symbols),
                    len(source.manifest.address_operand_bindings),
                    len(source.manifest.runtime_symbol_definitions),
                )
                self.assertEqual(counts, expected[case.pattern])
                self.assertEqual(len(source.manifest.input_digests), 8)
                self.assertEqual(len(source.fragment.core_streams), 2)
                self.assertEqual(case.decision, decision_before)
                terminal_ids = unfused_comparison_terminal_abi_ids(source)
                terminal_bytes = sum(
                    item.size_bytes
                    for item in source.fragment.buffer_abi
                    if item.id in terminal_ids
                )
                dtype_bytes = {
                    DType.FP16: 2,
                    DType.FP32: 4,
                    DType.INT32: 4,
                }[source.plan.problem.gemm.dtype]
                expected_output_bytes = dtype_bytes
                for extent in source.plan.problem.gemm.output.shape:
                    expected_output_bytes *= extent
                if case.pattern in (
                    FusionPattern.AG_GEMM,
                    FusionPattern.GEMM_RS,
                ):
                    self.assertEqual(terminal_bytes, expected_output_bytes)
                roots = {
                    item.binding_id: item
                    for item in source.fragment.buffer_abi
                    if item.alias_of is None
                }
                aliases = [
                    item for item in source.fragment.buffer_abi
                    if item.alias_of is not None
                ]
                self.assertTrue(aliases)
                self.assertTrue(all(item.alias_of in roots for item in aliases))

    def test_local_copy_and_reduce_are_explicit_and_contiguous(self) -> None:
        for case in self.cases[1:]:
            with self.subTest(pattern=case.pattern.value):
                source = _build(case)
                quotient = Counter(
                    opcode
                    for task in source.lowered.tasks
                    for opcode in task.opcodes
                )
                self.assertEqual(quotient[RecordOpcode.DTE_ISSUE], 2)
                self.assertEqual(quotient[RecordOpcode.LOCAL_REDUCE], 2)
                issue_records = tuple(
                    record
                    for stream in source.fragment.core_streams
                    for record in stream.records
                    if record.opcode is RecordOpcode.DTE_ISSUE
                )
                self.assertEqual(len(issue_records), 2)
                for record in issue_records:
                    operands = {item.name: item for item in record.operands}
                    self.assertEqual(
                        operands["payload_bits"].literal_value,
                        operands["size_bytes"].literal_value * 8,
                    )
                for contract in source.operand_abi.reduce_contracts:
                    views = sorted(
                        (item for item in source.operand_abi.operands if item.task_ref == contract.task_ref),
                        key=lambda item: item.ordinal,
                    )
                    if case.pattern is FusionPattern.GEMM_AR:
                        self.assertEqual(
                            views[0].storage_ref, views[1].storage_ref
                        )
                        self.assertEqual(
                            views[0].byte_offset + views[0].byte_extent,
                            views[1].byte_offset,
                        )
                    else:
                        self.assertNotEqual(
                            views[0].storage_ref, views[1].storage_ref
                        )
                        self.assertEqual(
                            (views[0].byte_offset, views[1].byte_offset),
                            (0, 0),
                        )
                    self.assertEqual((views[2].storage_ref, views[2].byte_offset), (views[1].storage_ref, views[1].byte_offset))

    def test_reduce_uses_problem_fp32_accumulation_and_rejects_tamper(self) -> None:
        for case in self.cases[1:]:
            with self.subTest(pattern=case.pattern.value):
                source = _build(case)
                self.assertIs(source.plan.problem.gemm.accumulation_dtype, DType.FP32)
                self.assertTrue(source.operand_abi.reduce_contracts)
                self.assertTrue(all(
                    item.accumulation_dtype is source.plan.problem.gemm.accumulation_dtype
                    for item in source.operand_abi.reduce_contracts
                ))
                reduce_records = tuple(
                    record
                    for stream in source.fragment.core_streams
                    for record in stream.records
                    if record.opcode is RecordOpcode.LOCAL_REDUCE
                )
                self.assertEqual(len(reduce_records), 2)
                for record in reduce_records:
                    operands = {item.name: item for item in record.operands}
                    self.assertEqual(
                        tuple(
                            operands[name].literal_value
                            for name in (
                                "input_dtype",
                                "accumulator_dtype",
                                "output_dtype",
                            )
                        ),
                        (0, 1, 0),
                    )

                wrong = replace(
                    source.operand_abi.reduce_contracts[0],
                    accumulation_dtype=DType.FP16,
                )
    def test_restable_address_and_old_digest_tamper_fail_closed(self) -> None:
        source = _build(self.cases[2])
        first, second, *tail = source.core_abi.storage_bindings
        forged_core = UnfusedComparisonCoreABI.create(
            source_ir1_id=source.core_abi.source_ir1_id,
            source_plan_ref=source.core_abi.source_plan_ref,
            source_projection_ref=source.core_abi.source_projection_ref,
            task_bindings=source.core_abi.task_bindings,
            storage_bindings=(replace(first, base_address=second.base_address), second, *tail),
            runtime_bindings=source.core_abi.runtime_bindings,
            barrier_events=source.core_abi.barrier_events,
        )
        with self.assertRaisesRegex(SchemaError, "overlap|exact deterministic"):
            forged_core.validate_against(source.ir1, source.plan, source.projection)

        digests = list(source.manifest.input_digests)
        digests[0] = replace(digests[0], schema_version="wafer_frontend.old/v0")
        with self.assertRaisesRegex(SchemaError, "schema versions are not exact"):
            replace(source.manifest, input_digests=tuple(digests)).validate()

    def test_four_rank_ag_rs_standard_sources_have_four_exact_streams(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[1])
        expected = {
            FusionPattern.AG_GEMM: (
                (20, 20, 20, 20),
                (4, 8, 28),
                48,
                41,
                80,
                52,
            ),
            FusionPattern.GEMM_RS: (
                (29, 29, 29, 29),
                (12, 8, 40),
                52,
                73,
                136,
                56,
            ),
        }
        for decision in case.decisions:
            with self.subTest(pattern=decision.problem.pattern.value):
                adapter = SimpleNamespace(
                    partitioned_graph=case.partitioned_graph,
                    decision=decision,
                )
                source = _build(adapter)
                source.validate_against()
                self.assertEqual(source, _build(adapter))
                ownership = Counter(
                    item.ownership for item in source.fragment.buffer_abi
                )
                counts = (
                    tuple(
                        len(stream.records)
                        for stream in source.fragment.core_streams
                    ),
                    (
                        ownership[BufferOwnership.OWNED],
                        ownership[BufferOwnership.BORROWED],
                        ownership[BufferOwnership.ALIASED],
                    ),
                    len(source.fragment.runtime_symbols),
                    len(source.fragment.program_symbols),
                    len(source.manifest.address_operand_bindings),
                    len(source.manifest.runtime_symbol_definitions),
                )
                self.assertEqual(counts, expected[decision.problem.pattern])
                self.assertEqual(len(source.fragment.core_streams), 4)
                self.assertEqual(len(source.manifest.input_digests), 8)
                for stream in source.fragment.core_streams:
                    quotient = Counter(record.opcode for record in stream.records)
                    self.assertEqual(quotient[RecordOpcode.DTE_SEND], 3)
                    self.assertEqual(quotient[RecordOpcode.DTE_RECV], 3)

    def test_four_rank_standard_rejects_a_dropped_core_stream(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[1])
        source = _build(SimpleNamespace(
            partitioned_graph=case.partitioned_graph,
            decision=case.decisions[0],
        ))
        forged = replace(
            source,
            fragment=replace(
                source.fragment,
                core_streams=source.fragment.core_streams[:-1],
            ),
        )
        with self.assertRaisesRegex(
            SchemaError, "every claimed action must emit at least one record"
        ):
            forged.validate_against()

    def test_three_rank_ar_standard_lower_link_is_closed(self) -> None:
        ir1 = build_swizzle_scale_case(
            build_swizzle_scale_points()[1]
        ).partitioned_graph
        problem, baseline = _flexible_rank_problem(
            self.cases[2].decision, 3
        )
        problem_semantic = problem._semantic_key()
        problem_semantic["source_ir1_id"] = ir1.id
        problem = type(problem).create(**problem_semantic)
        baseline_semantic = baseline._semantic_key()
        baseline_semantic["problem_ref"] = problem.id
        baseline = SwizzleCandidate.create(**baseline_semantic)
        source = _build(SimpleNamespace(
            partitioned_graph=ir1,
            decision=SimpleNamespace(problem=problem, baseline=baseline),
        ))
        source.validate_against()
        self.assertEqual(source, _build(SimpleNamespace(
            partitioned_graph=ir1,
            decision=SimpleNamespace(problem=problem, baseline=baseline),
        )))
        self.assertEqual(
            (
                len(source.projection.flows),
                len(source.core_abi.barrier_events),
                tuple(len(stream.records) for stream in source.fragment.core_streams),
            ),
            (12, 8, (32, 32, 34)),
        )
        for stream in source.fragment.core_streams:
            quotient = Counter(record.opcode for record in stream.records)
            self.assertEqual(quotient[RecordOpcode.DTE_SEND], 4)
            self.assertEqual(quotient[RecordOpcode.DTE_RECV], 4)
            self.assertEqual(quotient[RecordOpcode.LOCAL_REDUCE], 2)
            event_opcodes = tuple(
                record.opcode
                for record in stream.records
                if record.opcode in (
                    RecordOpcode.EVENT_SET,
                    RecordOpcode.EVENT_WAIT,
                )
            )
            self.assertEqual(
                event_opcodes,
                (RecordOpcode.EVENT_WAIT,) * 2 + (RecordOpcode.EVENT_SET,) * 2
                if stream.logical_core.die_id == 2
                else (RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT),
            )
        barrier_rank_by_task = {
            action.id: program.rank
            for program in source.plan.rank_programs
            for action in program.actions
            if action.kind is SwizzleActionKind.BARRIER
        }
        self.assertEqual(
            Counter(
                barrier_rank_by_task[event.owner_task_ref]
                for event in source.core_abi.barrier_events
            ),
            Counter({0: 2, 1: 2, 2: 4}),
        )
        final_wave_ranks = {
            rank for pair in _multi_rank_peer_waves((0, 1, 2))[-1]
            for rank in pair
        }
        self.assertNotIn(2, final_wave_ranks)

    def test_circle_rank_streams_put_one_fence_at_each_peer_wave_tail(self) -> None:
        from test_swizzle_meshslice_standard import _case, _planner

        decision = self.cases[2].decision
        ignored = {RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.SRAM_FREE}
        for rank_count in (3, 6, 10):
            with self.subTest(rank_count=rank_count):
                ir1, _ = _case(
                    rows=1,
                    columns=rank_count,
                    planner=_planner(
                        max_actions=2_000,
                        max_buffers=256,
                        max_chunk_count=1,
                        sram_budget_bytes=16 << 20,
                    ),
                )
                problem, baseline = _flexible_rank_problem(decision, rank_count)
                problem_semantic = problem._semantic_key()
                problem_semantic["source_ir1_id"] = ir1.id
                problem = type(problem).create(**problem_semantic)
                baseline_semantic = baseline._semantic_key()
                baseline_semantic["problem_ref"] = problem.id
                baseline = SwizzleCandidate.create(**baseline_semantic)
                source = _build(SimpleNamespace(
                    partitioned_graph=ir1,
                    decision=SimpleNamespace(
                        problem=problem,
                        baseline=baseline,
                    ),
                ))
                actions = {
                    action.id: action
                    for program in source.plan.rank_programs
                    for action in program.actions
                }
                records_by_action = {}
                for stream in source.fragment.core_streams:
                    for record in stream.records:
                        if record.opcode not in ignored:
                            records_by_action.setdefault(
                                record.source_global_action_id, []
                            ).append(record.opcode)
                remote_waits = 0
                for program in source.plan.rank_programs:
                    for index, action in enumerate(program.actions):
                        if action.kind is not SwizzleActionKind.WAIT:
                            continue
                        recv = actions[action.deps[0]]
                        if recv.kind is not SwizzleActionKind.RECV:
                            continue
                        remote_waits += 1
                        if action.rank < recv.peer_rank:
                            self.assertEqual(
                                records_by_action[action.id],
                                [RecordOpcode.DTE_WAIT, RecordOpcode.DTE_FENCE],
                            )
                        else:
                            self.assertEqual(
                                records_by_action[action.id],
                                [RecordOpcode.DTE_WAIT],
                            )
                            send = program.actions[index + 1]
                            self.assertIs(send.kind, SwizzleActionKind.SEND)
                            self.assertEqual(send.peer_rank, recv.peer_rank)
                            self.assertEqual(
                                records_by_action[send.id],
                                [RecordOpcode.DTE_SEND, RecordOpcode.DTE_FENCE],
                            )
                self.assertEqual(remote_waits, 2 * rank_count * (rank_count - 1))
                self.assertEqual(
                    sum(
                        opcode is RecordOpcode.DTE_FENCE
                        for opcodes in records_by_action.values()
                        for opcode in opcodes
                    ),
                    remote_waits,
                )

    def test_split_reduce_inputs_reject_reversed_physical_pair(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[2])
        source = _build(SimpleNamespace(
            partitioned_graph=case.partitioned_graph,
            decision=case.decisions[1],
        ))
        task_ref = source.operand_abi.reduce_contracts[0].task_ref
        action = next(
            action
            for program in source.plan.rank_programs
            for action in program.actions
            if action.id == task_ref
        )
        views = sorted(
            (
                item for item in source.projection.operands
                if item.task_ref == task_ref
            ),
            key=lambda item: item.ordinal,
        )
        bindings = source.core_abi.storage_bindings
        first = next(
            item for item in bindings
            if (item.rank, item.storage_ref)
            == (action.rank, views[0].storage_ref)
        )
        second = next(
            item for item in bindings
            if (item.rank, item.storage_ref)
            == (action.rank, views[1].storage_ref)
        )
        forged = UnfusedComparisonCoreABI.create(
            source_ir1_id=source.core_abi.source_ir1_id,
            source_plan_ref=source.core_abi.source_plan_ref,
            source_projection_ref=source.core_abi.source_projection_ref,
            task_bindings=source.core_abi.task_bindings,
            storage_bindings=tuple(
                replace(item, base_address=second.base_address)
                if item is first
                else replace(item, base_address=first.base_address)
                if item is second
                else item
                for item in bindings
            ),
            runtime_bindings=source.core_abi.runtime_bindings,
            barrier_events=source.core_abi.barrier_events,
        )
        with self.assertRaisesRegex(
            SchemaError, "split REDUCE inputs must form one exact contiguous"
        ):
            forged.validate_against(
                source.ir1, source.plan, source.projection,
            )

    def test_lifetime_coloring_rejects_live_overlap_tamper(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[2])
        source = _build(SimpleNamespace(
            partitioned_graph=case.partitioned_graph,
            decision=case.decisions[1],
        ))
        bindings = source.core_abi.storage_bindings
        left, right = next(
            (left, right)
            for index, left in enumerate(bindings)
            for right in bindings[index + 1 :]
            if left.logical_core == right.logical_core
            and left.region_ref == right.region_ref
            and left.lifetime_start < right.lifetime_end_exclusive
            and right.lifetime_start < left.lifetime_end_exclusive
            and left.base_address != right.base_address
        )
        forged_bindings = tuple(
            replace(item, base_address=left.base_address)
            if item is right else item
            for item in bindings
        )
        forged = UnfusedComparisonCoreABI.create(
            source_ir1_id=source.core_abi.source_ir1_id,
            source_plan_ref=source.core_abi.source_plan_ref,
            source_projection_ref=source.core_abi.source_projection_ref,
            task_bindings=source.core_abi.task_bindings,
            storage_bindings=forged_bindings,
            runtime_bindings=source.core_abi.runtime_bindings,
            barrier_events=source.core_abi.barrier_events,
        )
        with self.assertRaisesRegex(
            SchemaError, "overlap during live intervals"
        ):
            forged.validate_against(
                source.ir1, source.plan, source.projection,
            )

    def test_two_rank_storage_is_sequential_while_scale_can_reuse(self) -> None:
        for case in self.cases:
            source = _build(case)
            roots = tuple(
                item
                for item in source.fragment.buffer_abi
                if item.alias_of is None
            )
            for index, left in enumerate(roots):
                for right in roots[index + 1 :]:
                    if (
                        left.logical_core == right.logical_core
                        and left.region_ref == right.region_ref
                    ):
                        self.assertTrue(
                            left.region_offset_bytes + left.size_bytes
                            <= right.region_offset_bytes
                            or right.region_offset_bytes + right.size_bytes
                            <= left.region_offset_bytes
                        )

        specs = {
            "first": (64, 0, 1),
            "second": (64, 1, 2),
        }
        sequential = _allocate_storage_intervals(
            rank=0,
            logical_core=LogicalCoreRef(0, 0),
            region_ref="comm",
            region_base=0,
            region_size=128,
            storage_specs=specs,
            reuse_lifetimes=False,
        )
        colored = _allocate_storage_intervals(
            rank=0,
            logical_core=LogicalCoreRef(0, 0),
            region_ref="comm",
            region_base=0,
            region_size=128,
            storage_specs=specs,
            reuse_lifetimes=True,
        )
        self.assertEqual(
            tuple(item.base_address for item in sequential),
            (0, 64),
        )
        self.assertEqual(
            tuple(item.base_address for item in colored),
            (0, 0),
        )

    def test_compute_storage_units_are_small_first_and_pair_stays_whole(
        self,
    ) -> None:
        bindings = _allocate_storage_intervals(
            rank=0,
            logical_core=LogicalCoreRef(0, 0),
            region_ref="dense_release",
            region_base=0,
            region_size=1 << 20,
            storage_specs={
                "aaa_unrelated": (102400, 0, 2),
                "data": (2048, 0, 2),
                "input": (25600, 0, 2),
                "output": (102400, 0, 2),
                "reduce_accumulator": (102400, 0, 2),
            },
            contiguous_pairs=(("output", "reduce_accumulator"),),
            priority_storage_refs=frozenset(("data", "input", "output")),
            reuse_lifetimes=False,
        )
        by_ref = {item.storage_ref: item for item in bindings}
        self.assertEqual(
            tuple(
                by_ref[ref].base_address
                for ref in ("data", "input", "output")
            ),
            (0, 2048, 27648),
        )
        self.assertEqual(
            by_ref["reduce_accumulator"].base_address,
            by_ref["output"].base_address + by_ref["output"].size_bytes,
        )
        self.assertGreaterEqual(by_ref["aaa_unrelated"].base_address, 232448)

    def test_compute_wire_preflight_accepts_max_and_rejects_plus_one(
        self,
    ) -> None:
        source = _build(self.cases[0])
        stream = next(
            item for item in source.fragment.core_streams
            if any(record.opcode is RecordOpcode.MATMUL for record in item.records)
        )
        record_index = next(
            index for index, record in enumerate(stream.records)
            if record.opcode is RecordOpcode.MATMUL
        )
        relocation = next(
            item for item in stream.address_relocations
            if item.record_index == record_index
            and item.operand_id is SemanticOperandId.COMPUTE_INPUT_ADDRESS
        )
        accepted = tuple(
            replace(item, value=0xFFFF - relocation.addend)
            if item.symbol.id == relocation.symbol_ref else item
            for item in source.manifest.program_symbol_definitions
        )
        _validate_compute_wire_addresses(source.fragment, accepted)
        rejected = tuple(
            replace(item, value=0x10000 - relocation.addend)
            if item.symbol.id == relocation.symbol_ref else item
            for item in source.manifest.program_symbol_definitions
        )
        with self.assertRaisesRegex(
            SchemaError,
            "compute relocated address cannot fit the uint16 wire",
        ):
            _validate_compute_wire_addresses(source.fragment, rejected)

    def test_individual_storage_larger_than_region_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            SchemaError, "individual UNFUSED storage exceeds exact SRAM region"
        ):
            _allocate_storage_intervals(
                rank=0,
                logical_core=LogicalCoreRef(0, 0),
                region_ref="comm",
                region_base=0,
                region_size=64 * 1024,
                storage_specs={"too_large": (64 * 1024 + 1, 0, 1)},
            )

    def test_live_storage_set_larger_than_region_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            SchemaError, "live UNFUSED storage set exceeds exact SRAM region"
        ):
            _allocate_storage_intervals(
                rank=0,
                logical_core=LogicalCoreRef(0, 0),
                region_ref="comm",
                region_base=0,
                region_size=64 * 1024,
                storage_specs={
                    "live_a": (40 * 1024, 0, 2),
                    "live_b": (40 * 1024, 1, 3),
                },
            )


if __name__ == "__main__":
    unittest.main()
