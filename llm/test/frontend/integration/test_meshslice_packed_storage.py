from __future__ import annotations

from dataclasses import replace
from math import prod
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.lowering.swizzle_meshslice_standard import (
    link_meshslice_2d_standard_program,
)
from llm.frontend.wafer_frontend.passes.meshslice_packed_storage import (
    build_meshslice_packed_storage,
)
from llm.frontend.wafer_frontend.passes.load_fabric import SIMULATOR_CYCLE_NS
from llm.frontend.wafer_frontend.passes.program_io import (
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    ProgramSymbolKind,
    RecordOpcode,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_runtime import (
    FlexibleMeshRuntimeBaseline,
    FlexibleMeshRuntimeCase,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_workload import (
    FlexibleMeshSliceOperation,
    FlexibleMeshWorkloadSpec,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
)

from flexible_mesh_runtime_meshslice import (
    _runtime_case,
    ProductionMeshSliceRuntimeMaterializer,
)


class MeshSlicePackedStorageTest(unittest.TestCase):
    @staticmethod
    def _source(rows: int, columns: int):
        _hardware, ir1, decision = _runtime_case(rows, columns)
        candidate = next(
            item for item in decision.ranked_candidates
            if item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
            and item.chunk_count == 1
        )
        source, audit = link_meshslice_2d_standard_program(
            ir1, decision, candidate_ref=candidate.id
        )
        return source, audit

    @staticmethod
    def _matmul_parameters(source) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            record.operands[-1].literal_value
            for stream in source.fragment.core_streams
            for record in stream.records
            if record.opcode is RecordOpcode.MATMUL
        )

    def test_runtime_tiny_gemm_has_positive_p5_cost_without_drifting_other_shapes(self) -> None:
        expected = {
            (1, 1): (1, 4, 4, 4),
            (1, 2): (1, 4, 4, 4),
            (2, 1): (1, 4, 4, 4),
            (2, 2): (1, 4, 8, 4),
        }
        for (rows, columns), parameters in expected.items():
            with self.subTest(rows=rows, columns=columns):
                hardware, _ir1, _decision = _runtime_case(rows, columns)
                source, _audit = self._source(rows, columns)
                actual = self._matmul_parameters(source)
                self.assertEqual(actual, (parameters,) * (rows * columns))
                if (rows, columns) == (1, 1):
                    batch, tokens, channels, outputs = parameters
                    vec_ops = batch * tokens * channels * outputs * 2
                    core = hardware["cores"][0]
                    vec_cycle_ns = (
                        vec_ops // (core["vec_x"] * core["vec_cnt"])
                    ) * SIMULATOR_CYCLE_NS
                    self.assertGreater(vec_cycle_ns, 0)

    def test_representative_rectangles_have_exact_root_full_chunk_graph(self) -> None:
        for rows, columns in ((1, 3), (3, 1), (2, 3)):
            with self.subTest(rows=rows, columns=columns):
                source, audit = self._source(rows, columns)
                packed = build_meshslice_packed_storage(
                    source.plan,
                    source.projection,
                    source.core_abi,
                    source.operand_abi,
                )
                self.assertIsNotNone(packed)
                ranks = rows * columns
                packed_operands = ranks * (
                    int(rows > 1) + int(columns > 1)
                )
                chunks = ranks * (
                    (rows if rows > 1 else 0)
                    + (columns if columns > 1 else 0)
                )
                self.assertEqual(len(packed.roots), packed_operands)
                self.assertEqual(sum(len(root.chunks) for root in packed.roots), chunks)
                self.assertEqual(audit.ranks, ranks)
                for root in packed.roots:
                    self.assertEqual(root.full_view.byte_offset, 0)
                    self.assertEqual(
                        root.full_view.byte_extent,
                        root.address_binding.size_bytes,
                    )
                    cursor = 0
                    for chunk in root.chunks:
                        self.assertEqual(chunk.byte_offset, cursor)
                        self.assertEqual(
                            prod(chunk.shape) * (2 if chunk.dtype.value == "fp16" else 4),
                            chunk.byte_extent,
                        )
                        cursor += chunk.byte_extent
                    self.assertEqual(cursor, root.address_binding.size_bytes)

    def test_dte_relocations_use_chunk_local_zero_addends(self) -> None:
        for rows, columns in ((1, 3), (3, 1), (2, 3)):
            with self.subTest(rows=rows, columns=columns):
                source, _audit = self._source(rows, columns)
                fragment = source.fragment
                symbols = {item.id: item for item in fragment.program_symbols}
                buffers = {item.binding_id: item for item in fragment.buffer_abi}
                bindings = {
                    (
                        item.fragment_id,
                        item.logical_core,
                        item.fragment_record_index,
                        item.operand_id,
                    ): item
                    for item in source.manifest.address_operand_bindings
                }
                for stream in fragment.core_streams:
                    for relocation in stream.address_relocations:
                        record = stream.records[relocation.record_index]
                        if record.opcode not in (
                            RecordOpcode.DTE_SEND,
                            RecordOpcode.DTE_RECV,
                        ):
                            continue
                        if relocation.operand_id not in (
                            SemanticOperandId.SOURCE_ADDRESS,
                            SemanticOperandId.DESTINATION_ADDRESS,
                        ):
                            continue
                        self.assertEqual(relocation.addend, 0)
                        symbol = symbols[relocation.symbol_ref]
                        self.assertIs(symbol.kind, ProgramSymbolKind.ABSOLUTE_ADDRESS)
                        abi = buffers[symbol.source_ref]
                        self.assertEqual(
                            abi.layout, "swizzle_meshslice_packed_chunk/v1"
                        )
                        witness = bindings[(
                            fragment.id,
                            stream.logical_core,
                            relocation.record_index,
                            relocation.operand_id,
                        )]
                        self.assertEqual(witness.buffer_abi_ids, (abi.id,))
                        self.assertEqual(witness.tensor_slices, (abi.tensor_slice,))

                for abi in fragment.buffer_abi:
                    self.assertEqual(
                        abi.region_offset_bytes % abi.alignment_bytes,
                        0,
                    )

    def test_nonfrozen_rectangles_are_session_safe_matching_waves(self) -> None:
        for rows, columns in ((1, 3), (3, 1), (2, 3)):
            source, _audit = self._source(rows, columns)
            order = {
                item.task_ref: item.core_order
                for item in source.core_abi.task_bindings
            }
            for dag in source.projection.rank_dags:
                kinds = tuple(
                    task.kind
                    for task in sorted(dag.tasks, key=lambda item: order[item.id])
                    if task.kind in (
                        SwizzleActionKind.SEND,
                        SwizzleActionKind.RECV,
                        SwizzleActionKind.WAIT,
                    )
                )
                self.assertEqual(len(kinds), 3 * (rows + columns - 2))
                for offset in range(0, len(kinds), 3):
                    self.assertIn(
                        kinds[offset:offset + 3],
                        (
                            (SwizzleActionKind.SEND,
                             SwizzleActionKind.RECV,
                             SwizzleActionKind.WAIT),
                            (SwizzleActionKind.RECV,
                             SwizzleActionKind.WAIT,
                             SwizzleActionKind.SEND),
                        ),
                    )

    def test_representative_rs_ar_use_strict_unfused_fallback_program_io(self) -> None:
        materializer = ProductionMeshSliceRuntimeMaterializer(
            mapping_text="0:0\n"
        )
        for rows, columns in ((1, 3), (3, 1), (2, 3)):
            for operation in (
                FlexibleMeshSliceOperation.GEMM_RS,
                FlexibleMeshSliceOperation.GEMM_AR,
            ):
                with self.subTest(rows=rows, columns=columns, operation=operation):
                    case = FlexibleMeshRuntimeCase.create(
                        FlexibleMeshWorkloadSpec.dense_infer(
                            RectMeshSpec(rows, columns)
                        ),
                        operation,
                    )
                    self.assertIs(
                        case.selected_baseline,
                        FlexibleMeshRuntimeBaseline.UNFUSED_FALLBACK,
                    )
                    self.assertEqual(
                        case.fallback_reason, "STRICT_TWO_INPUT_REDUCE_ABI"
                    )
                    executable = materializer.materialize(case)
                    self.assertEqual(
                        len(executable.manifest.core_streams), rows * columns
                    )
                    contract = build_timing_program_io(
                        executable.linked_source, "0" * 64
                    )
                    contract.validate_against(executable.manifest)

    def test_private_session_is_byte_exact_and_public_validation_stays_strict(self) -> None:
        mesh = RectMeshSpec(2, 3)
        strict = (canonical_json(mesh), canonical_digest(mesh))
        with builder_validation_session():
            fast = (canonical_json(mesh), canonical_digest(mesh))
            forged_mesh = replace(mesh, rows=3)
            forged_digest = canonical_digest(forged_mesh)
        self.assertEqual(fast, strict)
        self.assertNotEqual(forged_digest, strict[1])

        source, _audit = self._source(1, 3)
        source.fragment.validate("fragment")
        chunk = next(
            abi for abi in source.fragment.buffer_abi
            if abi.layout == "swizzle_meshslice_packed_chunk/v1"
        )
        forged_fragment = type(source.fragment).create(
            producer_pass=source.fragment.producer_pass,
            source_global_dag_id=source.fragment.source_global_dag_id,
            kind=source.fragment.kind,
            claimed_action_ids=source.fragment.claimed_action_ids,
            core_streams=source.fragment.core_streams,
            runtime_symbols=source.fragment.runtime_symbols,
            program_symbols=source.fragment.program_symbols,
            buffer_abi=tuple(
                replace(abi, alignment_bytes=64)
                if abi.id == chunk.id else abi
                for abi in source.fragment.buffer_abi
            ),
            state_abi=source.fragment.state_abi,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "packed alias alignment witness is not exact",
        ):
            forged_fragment.validate("fragment")

        forged_source = replace(source, id="swizzle_standard_linked_program_forged")
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            forged_source.validate_against()



if __name__ == "__main__":
    unittest.main()
