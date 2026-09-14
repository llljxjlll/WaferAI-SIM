from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.swizzle_meshslice_fallback import (
    link_meshslice_unfused_fallback_program,
    MeshSliceFallbackReason,
    MeshSliceSelectedPath,
)
from llm.frontend.wafer_frontend.lowering.swizzle_unfused import (
    _build_unfused_comparison_abis_prevalidated,
    allocate_unfused_comparison_core_abi,
    build_unfused_comparison_operand_abi,
    lower_unfused_comparison_opcodes,
)
from llm.frontend.wafer_frontend.lowering.swizzle_unfused_standard import (
    _link_unfused_comparison_program_prevalidated,
    link_unfused_comparison_program,
)
from llm.frontend.wafer_frontend.passes.project_unfused_comparison import (
    _project_unfused_comparison_prevalidated,
    build_unfused_comparison_plan,
    project_unfused_comparison,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleGroupView,
    SwizzleProblem,
    SwizzleRankPlacement,
    SwizzleRouteView,
)

from swizzle_cases import build_swizzle_integration_cases
from test_swizzle_meshslice_standard import _case, _planner
from test_swizzle_unfused_comparison import _flexible_rank_problem


_ZERO_SHA = "0" * 64
_SHAPES = ((1, 1), (1, 3), (3, 1), (2, 3))


def _exact_group(ir1) -> SwizzleGroupView:
    physical = ir1.groups[0]
    die_by_id = {item.id: item for item in ir1.fabric.dies}
    result = SwizzleGroupView(
        group_ref=physical.id,
        logical_shape=physical.logical_shape,
        placements=tuple(
            SwizzleRankPlacement(
                rank=item.rank,
                x=die_by_id[item.die_id].coord[0],
                y=die_by_id[item.die_id].coord[1],
            )
            for item in physical.placements
        ),
        routes=tuple(
            SwizzleRouteView(
                id=item.id,
                source_rank=item.source_rank,
                destination_rank=item.destination_rank,
                die_path=item.die_path,
                resource_ids=item.resource_ids,
            )
            for item in physical.embedding.routes
        ),
    )
    result.validate("fallback_test.group")
    return result


def _fallback_problem(ir1, decision):
    rank_count = len(ir1.groups[0].placements)
    flexible, baseline = _flexible_rank_problem(decision, rank_count)
    allowed = tuple(sorted(
        set(flexible.constraints.allowed_algorithms)
        | {SwizzleAlgorithm.MESHSLICE_2D_OS},
        key=lambda item: item.value,
    ))
    semantic = flexible._semantic_key()
    semantic.update(
        source_ir1_id=ir1.id,
        fused_op_id=ir1.fused_op_skeletons[0].id,
        group=_exact_group(ir1),
        constraints=replace(
            flexible.constraints,
            allowed_algorithms=allowed,
        ),
    )
    problem = SwizzleProblem.create(**semantic)
    baseline_semantic = baseline._semantic_key()
    baseline_semantic["problem_ref"] = problem.id
    return problem, SwizzleCandidate.create(**baseline_semantic)


class SwizzleMeshSliceFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_decisions = {
            case.pattern: case.decision
            for case in build_swizzle_integration_cases()
            if case.pattern in (FusionPattern.GEMM_RS, FusionPattern.GEMM_AR)
        }
        cls.meshes = {}
        for rows, columns in _SHAPES:
            ranks = rows * columns
            expected_actions = ranks * (
                3 * (rows + columns - 2) + 1
            )
            ir1, _decision = _case(
                rows=rows,
                columns=columns,
                planner=_planner(
                    max_actions=max(expected_actions, 1),
                    max_buffers=max(3 * ranks, 4),
                    max_chunk_count=1,
                    sram_budget_bytes=16 << 20,
                ),
            )
            cls.meshes[(rows, columns)] = ir1

    def test_representative_shapes_lower_link_and_program_io(self) -> None:
        for pattern, decision in self.source_decisions.items():
            for rows, columns in _SHAPES:
                with self.subTest(
                    pattern=pattern.value,
                    rows=rows,
                    columns=columns,
                ):
                    ir1 = self.meshes[(rows, columns)]
                    problem, baseline = _fallback_problem(ir1, decision)
                    result = link_meshslice_unfused_fallback_program(
                        ir1,
                        problem,
                        baseline,
                    )
                    result.validate()
                    self.assertIs(
                        result.selected_path,
                        MeshSliceSelectedPath.UNFUSED_FALLBACK,
                    )
                    self.assertIs(
                        result.reason,
                        MeshSliceFallbackReason.STRICT_TWO_INPUT_REDUCE_ABI,
                    )
                    self.assertIs(result.pattern, pattern)
                    self.assertEqual(
                        len(result.linked_source.fragment.core_streams),
                        rows * columns,
                    )
                    program_io = build_timing_program_io(
                        result.linked_source,
                        _ZERO_SHA,
                    )
                    program_io.validate_against(
                        result.linked_source.manifest
                    )
                    self.assertEqual(
                        len(program_io.output_probes),
                        rows * columns,
                    )

    def test_single_rank_is_exact_gemm_then_typed_local_copy(self) -> None:
        ir1 = self.meshes[(1, 1)]
        for pattern, decision in self.source_decisions.items():
            with self.subTest(pattern=pattern.value):
                problem, baseline = _fallback_problem(ir1, decision)
                result = link_meshslice_unfused_fallback_program(
                    ir1,
                    problem,
                    baseline,
                )
                actions = result.linked_source.plan.rank_programs[0].actions
                self.assertEqual(
                    tuple(item.kind.value for item in actions),
                    ("comp", "local_copy"),
                )
                opcodes = Counter(
                    record.opcode
                    for stream in result.linked_source.fragment.core_streams
                    for record in stream.records
                )
                self.assertEqual(opcodes[RecordOpcode.MATMUL], 1)
                self.assertEqual(opcodes[RecordOpcode.DTE_ISSUE], 1)
                self.assertEqual(opcodes[RecordOpcode.DTE_WAIT], 1)
                self.assertEqual(opcodes[RecordOpcode.DTE_SEND], 0)
                self.assertEqual(opcodes[RecordOpcode.DTE_RECV], 0)
                self.assertEqual(opcodes[RecordOpcode.LOCAL_REDUCE], 0)

    def test_frozen_2x2_rs_terminal_probes_do_not_reuse_storage(self) -> None:
        rows, columns = 2, 2
        ir1, _decision = _case(
            rows=rows,
            columns=columns,
            planner=_planner(
                max_actions=rows * columns * (
                    3 * (rows + columns - 2) + 1
                ),
                max_buffers=3 * rows * columns,
                max_chunk_count=1,
                sram_budget_bytes=16 << 20,
            ),
        )
        decision = self.source_decisions[FusionPattern.GEMM_RS]
        problem, baseline = _fallback_problem(ir1, decision)
        result = link_meshslice_unfused_fallback_program(
            ir1,
            problem,
            baseline,
        )
        program_io = build_timing_program_io(result.linked_source, _ZERO_SHA)
        abis = {
            abi.id: abi
            for abi in result.linked_source.fragment.buffer_abi
        }

        for probe in program_io.output_probes:
            terminal = abis[probe.target.buffer_abi_id]
            terminal_end = terminal.region_offset_bytes + terminal.size_bytes
            for other in abis.values():
                if (
                    other.id == terminal.id
                    or other.alias_of is not None
                    or other.storage_id == terminal.storage_id
                    or other.logical_core != terminal.logical_core
                    or other.region_ref != terminal.region_ref
                ):
                    continue
                other_end = other.region_offset_bytes + other.size_bytes
                self.assertTrue(
                    terminal_end <= other.region_offset_bytes
                    or other_end <= terminal.region_offset_bytes,
                    msg=(
                        f"terminal probe {terminal.id} physically overlaps "
                        f"distinct root {other.id}"
                    ),
                )

    def test_private_2x3_link_is_canonical_exact_and_public_rejects_forgery(self) -> None:
        ir1 = self.meshes[(2, 3)]
        decision = self.source_decisions[FusionPattern.GEMM_RS]
        problem, baseline = _fallback_problem(ir1, decision)

        strict_plan = build_unfused_comparison_plan(ir1, problem, baseline)
        strict_projection = project_unfused_comparison(ir1, strict_plan)
        strict_core = allocate_unfused_comparison_core_abi(
            ir1, strict_plan, strict_projection
        )
        strict_operand = build_unfused_comparison_operand_abi(
            ir1, strict_plan, strict_projection
        )
        strict_lowered = lower_unfused_comparison_opcodes(
            strict_plan, strict_projection
        )
        strict = link_unfused_comparison_program(
            ir1,
            strict_plan,
            strict_projection,
            strict_lowered,
            strict_core,
            strict_operand,
        )

        with builder_validation_session():
            fast_plan = build_unfused_comparison_plan(ir1, problem, baseline)
            fast_projection = _project_unfused_comparison_prevalidated(
                ir1, fast_plan
            )
            fast_core, fast_operand, fast_lowered = (
                _build_unfused_comparison_abis_prevalidated(
                    ir1, fast_plan, fast_projection
                )
            )
            fast = _link_unfused_comparison_program_prevalidated(
                ir1,
                fast_plan,
                fast_projection,
                fast_lowered,
                fast_core,
                fast_operand,
            )

        self.assertEqual(fast.id, strict.id)
        self.assertEqual(canonical_json(fast), canonical_json(strict))
        self.assertEqual(canonical_digest(fast), canonical_digest(strict))

        forged_projection = replace(
            strict_projection,
            source_plan_ref="unfused_comparison_plan_forged",
        )
        with self.assertRaises(SchemaError):
            link_unfused_comparison_program(
                ir1,
                strict_plan,
                forged_projection,
                strict_lowered,
                strict_core,
                strict_operand,
            )

    def test_selection_metadata_is_fail_closed(self) -> None:
        ir1 = self.meshes[(1, 1)]
        decision = self.source_decisions[FusionPattern.GEMM_RS]
        problem, baseline = _fallback_problem(ir1, decision)
        result = link_meshslice_unfused_fallback_program(
            ir1,
            problem,
            baseline,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "strict RS/AR UNFUSED fallback selection",
        ):
            replace(
                result,
                selected_path=MeshSliceSelectedPath.MESHSLICE_STANDARD,
            ).validate()


if __name__ == "__main__":
    unittest.main()
