from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.swizzle_abi import (
    allocate_swizzle_core_address_abi,
)
from llm.frontend.wafer_frontend.lowering.swizzle import lower_swizzle_projection
from llm.frontend.wafer_frontend.lowering.swizzle_standard import (
    link_swizzle_standard_program,
)
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
from llm.frontend.wafer_frontend.passes.project_swizzle_plan import (
    project_swizzle_plan,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    materialize_swizzle_decision,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize_ir1 import (
    materialize_swizzle_plan,
)
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    _canonical_lifecycle_roots,
)
from llm.frontend.wafer_frontend.schema.ir2 import TensorSlice
from llm.frontend.wafer_frontend.schema.swizzle_operand_abi import (
    build_swizzle_operand_abi,
)
from llm.frontend.wafer_frontend.schema.swizzle_standard import (
    SwizzleStandardLinkedProgram,
)

from swizzle_cases import build_swizzle_integration_cases
from swizzle_scale_cases import build_swizzle_scale_case, build_swizzle_scale_points
from test_swizzle_w9_lowering import _products


_ZERO_SHA = "0" * 64


def _source(case):
    plan, projection, lowered, _legacy = _products(case)
    core_abi = allocate_swizzle_core_address_abi(
        case.partitioned_graph, plan, projection
    )
    operand_abi = build_swizzle_operand_abi(
        case.partitioned_graph, plan, projection
    )
    return link_swizzle_standard_program(
        case.partitioned_graph,
        plan,
        projection,
        lowered,
        core_abi,
        operand_abi,
    )


def _packed_source():
    case = build_swizzle_scale_case(build_swizzle_scale_points()[1])
    decision = case.decisions[0]
    adapter = materialize_swizzle_decision(decision)
    plan = materialize_swizzle_plan(
        case.partitioned_graph,
        decision,
        case.partitioned_graph.profile,
        deployment_selection=adapter.deployment_selection,
    )
    projection = project_swizzle_plan(
        case.partitioned_graph, plan,
    ).projection
    return link_swizzle_standard_program(
        case.partitioned_graph,
        plan,
        projection,
        lower_swizzle_projection(plan, projection),
        allocate_swizzle_core_address_abi(
            case.partitioned_graph, plan, projection,
        ),
        build_swizzle_operand_abi(
            case.partitioned_graph, plan, projection,
        ),
    )
class SwizzleProgramIoTest(unittest.TestCase):
    def test_s0_wrapper_rejects_packed_wang_layout(self) -> None:
        source = _source(build_swizzle_integration_cases()[0])
        aliases = {
            item.alias_of
            for item in source.fragment.buffer_abi
            if item.alias_of is not None
        }
        root = next(
            item
            for item in source.fragment.buffer_abi
            if item.alias_of is None and item.binding_id not in aliases
        )
        semantic = source.fragment._semantic_key()
        semantic["buffer_abi"] = tuple(
            replace(item, layout="swizzle_standard_storage_root/v1")
            if item == root else item
            for item in source.fragment.buffer_abi
        )
        forged_fragment = CommandFragment.create(
            producer_pass=source.fragment.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "packed BufferABI layouts require exact four-rank Wang",
        ):
            SwizzleStandardLinkedProgram.create(
                ir1=source.ir1,
                plan=source.plan,
                projection=source.projection,
                lowered=source.lowered,
                core_abi=source.core_abi,
                operand_abi=source.operand_abi,
                fragment=forged_fragment,
                manifest=source.manifest,
            )

    def test_fused_terminal_subview_allowlist_fails_closed(self) -> None:
        source = _packed_source()
        root = next(
            item
            for item in source.fragment.buffer_abi
            if item.layout == "swizzle_standard_terminal_root/v1"
        )
        alias = next(
            item
            for item in source.fragment.buffer_abi
            if item.alias_of == root.binding_id
        )

        wrong_layout = replace(alias, layout="row_major")
        out_of_bounds = replace(
            alias,
            region_offset_bytes=root.region_offset_bytes + root.size_bytes,
        )
        wrong_value = replace(
            alias,
            value_id="forged.output",
            tensor_slice=TensorSlice(
                "forged.output",
                alias.tensor_slice.offset,
                alias.tensor_slice.shape,
            ),
        )
        for forged in (wrong_layout, out_of_bounds, wrong_value):
            with self.subTest(forged=forged):
                buffers = tuple(
                    forged if item == alias else item
                    for item in source.fragment.buffer_abi
                )
                with self.assertRaisesRegex(
                    SchemaError,
                    "lifecycle alias must exactly preserve",
                ):
                    _canonical_lifecycle_roots(buffers, path="fragment.buffer_abi")

    def test_three_patterns_build_exact_zero_sha_timing_contracts(self) -> None:
        expected = {
            "ag_gemm": (10, 4, 3),
            "gemm_rs": (12, 2, 3),
            "gemm_ar": (12, 4, 2),
        }
        for case in build_swizzle_integration_cases():
            with self.subTest(pattern=case.pattern.value):
                source = _source(case)
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

                blobs = {blob.id: blob for blob in first.blobs}
                for entry in (*first.initializations, *first.output_probes):
                    blob = blobs[entry.blob_ref]
                    self.assertEqual(entry.length_bytes, blob.length_bytes)
                    self.assertEqual(len(blob.payload()), blob.length_bytes)
                self.assertEqual(
                    {probe.target.runtime_core_id for probe in first.output_probes},
                    {
                        binding.runtime_core_id
                        for binding in source.manifest.core_bindings
                        if binding.logical_core
                        in source.manifest.envelope.terminal_cores
                    },
                )

                initialized = {
                    entry.target.buffer_abi_id for entry in first.initializations
                }
                self.assertEqual(
                    initialized,
                    {
                        abi.id
                        for abi in source.fragment.buffer_abi
                        if abi.ownership is not BufferOwnership.ALIASED
                    },
                )

    def test_bare_manifest_is_not_a_supported_source(self) -> None:
        source = _source(build_swizzle_integration_cases()[0])
        with self.assertRaisesRegex(SchemaError, "supported linked program carrier"):
            build_timing_program_io(source.manifest, _ZERO_SHA)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
