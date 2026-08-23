from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
from llm.frontend.wafer_frontend.passes.unfused_comparison_program_io import (
    unfused_comparison_terminal_abi_ids,
)
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.program_io import _manifest_allocations

from swizzle_cases import build_swizzle_integration_cases
from swizzle_scale_cases import build_swizzle_scale_case, build_swizzle_scale_points
from test_swizzle_unfused_standard import _build


_ZERO_SHA = "0" * 64


def _replace_manifest_buffer(source, old, new, **manifest_changes):
    fragment = replace(
        source.fragment,
        buffer_abi=tuple(new if item == old else item for item in source.fragment.buffer_abi),
    )
    return replace(source.manifest, fragments=(fragment,), **manifest_changes)


class SwizzleUnfusedProgramIoTest(unittest.TestCase):
    def test_three_patterns_build_exact_zero_sha_timing_contracts(self) -> None:
        expected = {
            "ag_gemm": (6, 2, 3),
            "gemm_rs": (10, 2, 2),
            "gemm_ar": (8, 2, 3),
        }
        for case in build_swizzle_integration_cases():
            with self.subTest(pattern=case.pattern.value):
                source = _build(case)
                first = build_timing_program_io(source, _ZERO_SHA)
                second = build_timing_program_io(source, _ZERO_SHA)
                self.assertEqual(first, second)
                first.validate_against(source.manifest)
                self.assertEqual(first.program_artifact_sha256, _ZERO_SHA)
                self.assertEqual(
                    (
                        len(first.initializations),
                        len(first.output_probes),
                        len(first.blobs),
                    ),
                    expected[case.pattern.value],
                )

                roots = {
                    item.id
                    for item in source.fragment.buffer_abi
                    if item.alias_of is None
                }
                initialized_roots = {
                    item.target.buffer_abi_id for item in first.initializations
                }
                self.assertTrue(initialized_roots <= roots)
                omitted_roots = roots - initialized_roots
                self.assertEqual(
                    len(omitted_roots),
                    0,
                )
                buffers_by_id = {
                    item.id: item for item in source.fragment.buffer_abi
                }
                self.assertTrue(all(
                    buffers_by_id[item].ownership is BufferOwnership.OWNED
                    and buffers_by_id[item].lifetime_start > 0
                    for item in omitted_roots
                ))
                self.assertEqual(
                    {item.target.buffer_abi_id for item in first.output_probes},
                    unfused_comparison_terminal_abi_ids(source),
                )
                self.assertTrue(
                    all(
                        item.ownership is BufferOwnership.OWNED
                        for item in source.fragment.buffer_abi
                        if item.id in unfused_comparison_terminal_abi_ids(source)
                    )
                )

    def test_bare_manifest_is_not_a_supported_source(self) -> None:
        source = _build(build_swizzle_integration_cases()[0])
        with self.assertRaisesRegex(SchemaError, "supported linked program carrier"):
            build_timing_program_io(source.manifest, _ZERO_SHA)  # type: ignore[arg-type]

    def test_subview_gate_rejects_wrong_manifest_producer(self) -> None:
        source = _build(build_swizzle_integration_cases()[0])
        forged = replace(source.manifest, producer_pass="not_the_unfused_linker")
        with self.assertRaisesRegex(
            SchemaError, "directly reuse its canonical root allocation"
        ):
            _manifest_allocations(forged, "linked_program_manifest")

    def test_subview_gate_rejects_wrong_root_layout(self) -> None:
        source = _build(build_swizzle_integration_cases()[0])
        alias = next(item for item in source.fragment.buffer_abi if item.alias_of)
        root = next(
            item
            for item in source.fragment.buffer_abi
            if item.binding_id == alias.alias_of
        )
        forged = _replace_manifest_buffer(
            source, root, replace(root, layout="not_unfused_storage/v1")
        )
        with self.assertRaisesRegex(
            SchemaError, "directly reuse its canonical root allocation"
        ):
            _manifest_allocations(forged, "linked_program_manifest")

    def test_subview_gate_rejects_out_of_bounds_alias(self) -> None:
        source = _build(build_swizzle_integration_cases()[0])
        alias = next(item for item in source.fragment.buffer_abi if item.alias_of)
        root = next(
            item
            for item in source.fragment.buffer_abi
            if item.binding_id == alias.alias_of
        )
        forged_alias = replace(
            alias,
            region_offset_bytes=root.region_offset_bytes + root.size_bytes,
        )
        forged = _replace_manifest_buffer(source, alias, forged_alias)
        with self.assertRaisesRegex(
            SchemaError, "directly reuse its canonical root allocation"
        ):
            _manifest_allocations(forged, "linked_program_manifest")

    def test_four_rank_ag_rs_program_io_is_exact_and_deterministic(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[1])
        expected = {
            "ag_gemm": (
                (12, 4, 3),
                tuple(((32, 48), (0, rank * 48)) for rank in range(4)),
                12_288,
            ),
            "gemm_rs": (
                (12, 4, 3),
                tuple(((8, 64), (rank * 8, 0)) for rank in range(4)),
                4_096,
            ),
        }
        for decision in case.decisions:
            with self.subTest(pattern=decision.problem.pattern.value):
                source = _build(SimpleNamespace(
                    partitioned_graph=case.partitioned_graph,
                    decision=decision,
                ))
                first = build_timing_program_io(source, _ZERO_SHA)
                second = build_timing_program_io(source, _ZERO_SHA)
                self.assertEqual(first, second)
                first.validate_against(source.manifest)
                self.assertEqual(
                    (
                        len(first.initializations),
                        len(first.output_probes),
                        len(first.blobs),
                    ),
                    expected[decision.problem.pattern.value][0],
                )
                self.assertEqual(
                    {item.target.buffer_abi_id for item in first.output_probes},
                    unfused_comparison_terminal_abi_ids(source),
                )
                self.assertEqual(
                    tuple(
                        (
                            item.target.tensor_slice.shape,
                            item.target.tensor_slice.offset,
                        )
                        for item in first.output_probes
                    ),
                    expected[decision.problem.pattern.value][1],
                )
                buffers = {
                    item.id: item for item in source.fragment.buffer_abi
                }
                self.assertEqual(
                    sum(
                        buffers[item.target.buffer_abi_id].size_bytes
                        for item in first.output_probes
                    ),
                    expected[decision.problem.pattern.value][2],
                )

    def test_four_rank_program_io_rejects_out_of_bounds_alias(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[1])
        source = _build(SimpleNamespace(
            partitioned_graph=case.partitioned_graph,
            decision=case.decisions[1],
        ))
        alias = next(item for item in source.fragment.buffer_abi if item.alias_of)
        root = next(
            item
            for item in source.fragment.buffer_abi
            if item.binding_id == alias.alias_of
        )
        forged = _replace_manifest_buffer(
            source,
            alias,
            replace(
                alias,
                region_offset_bytes=root.region_offset_bytes + root.size_bytes,
            ),
        )
        with self.assertRaisesRegex(
            SchemaError, "directly reuse its canonical root allocation"
        ):
            _manifest_allocations(forged, "linked_program_manifest")

    def test_s2_rs_lifetime_reuse_is_exact_and_program_io_safe(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[2])
        source = _build(SimpleNamespace(
            partitioned_graph=case.partitioned_graph,
            decision=case.decisions[1],
        ))
        contract = build_timing_program_io(source, _ZERO_SHA)
        contract.validate_against(source.manifest)
        self.assertEqual(
            (
                len(contract.initializations),
                len(contract.output_probes),
                len(contract.blobs),
            ),
            (12, 4, 2),
        )

        bindings = source.core_abi.storage_bindings
        reused_pairs = tuple(
            (left, right)
            for index, left in enumerate(bindings)
            for right in bindings[index + 1 :]
            if left.logical_core == right.logical_core
            and left.region_ref == right.region_ref
            and left.base_address < right.base_address + right.size_bytes
            and right.base_address < left.base_address + left.size_bytes
        )
        self.assertEqual(len(reused_pairs), 8)
        self.assertTrue(all(
            left.lifetime_end_exclusive <= right.lifetime_start
            or right.lifetime_end_exclusive <= left.lifetime_start
            for left, right in reused_pairs
        ))

        roots = tuple(
            item for item in source.fragment.buffer_abi
            if item.alias_of is None
        )
        roots_by_storage = {}
        for binding in bindings:
            root = next(
                item for item in roots
                if item.logical_core == binding.logical_core
                and item.region_ref == binding.region_ref
                and item.region_offset_bytes == binding.base_address
                and item.size_bytes == binding.size_bytes
                and item.lifetime_start == binding.lifetime_start
                and item.lifetime_end_exclusive
                == binding.lifetime_end_exclusive
            )
            roots_by_storage[(binding.rank, binding.storage_ref)] = root
            self.assertEqual(
                (root.lifetime_start, root.lifetime_end_exclusive),
                (binding.lifetime_start, binding.lifetime_end_exclusive),
            )

        initialization_ids = {
            item.target.buffer_abi_id for item in contract.initializations
        }
        later_reused_roots = {
            roots_by_storage[
                (right.rank, right.storage_ref)
                if left.lifetime_end_exclusive <= right.lifetime_start
                else (left.rank, left.storage_ref)
            ].id
            for left, right in reused_pairs
        }
        self.assertTrue(later_reused_roots.isdisjoint(initialization_ids))
        buffers = {item.id: item for item in source.fragment.buffer_abi}
        self.assertEqual(
            tuple(
                (
                    item.target.tensor_slice.shape,
                    item.target.tensor_slice.offset,
                    buffers[item.target.buffer_abi_id].size_bytes,
                )
                for item in contract.output_probes
            ),
            tuple(
                ((16, 64), (rank * 16, 0), 2_048)
                for rank in range(4)
            ),
        )


if __name__ == "__main__":
    unittest.main()
