from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.schema.swizzle_evidence import (
    SwizzleComparisonBranch,
)

from swizzle_comparison import build_swizzle_comparison_suite
from swizzle_runtime_provider import ProductionSwizzleLowerLinkProvider


_ACTUAL_SHA = "1" * 64


class SwizzleRuntimeProviderTest(unittest.TestCase):
    def test_swizzle_branch_builds_production_source_and_actual_sha_io(self) -> None:
        provider = ProductionSwizzleLowerLinkProvider(
            hardware_json="{}", mapping_text="explicit-mapping"
        )
        expected = {
            "ag_gemm": (10, 4),
            "gemm_rs": (12, 2),
            "gemm_ar": (12, 4),
        }
        for case in build_swizzle_comparison_suite().cases:
            with self.subTest(pattern=case.pattern.value):
                executable = provider.lower_link(
                    case, SwizzleComparisonBranch.SWIZZLE
                )
                executable.validate()
                contract = provider.build_actual_sha_program_io(
                    executable, _ACTUAL_SHA
                )
                contract.validate_against(executable.manifest)
                self.assertEqual(contract.program_artifact_sha256, _ACTUAL_SHA)
                self.assertEqual(
                    (len(contract.initializations), len(contract.output_probes)),
                    expected[case.pattern.value],
                )

    def test_naive_branch_builds_typed_source_and_actual_sha_io(self) -> None:
        provider = ProductionSwizzleLowerLinkProvider(
            hardware_json="{}", mapping_text="explicit-mapping"
        )
        expected = {
            "ag_gemm": (22, 16),
            "gemm_rs": (32, 24),
            "gemm_ar": (42, 28),
        }
        expected_io = {
            "ag_gemm": (6, 2, 3),
            "gemm_rs": (8, 2, 2),
            "gemm_ar": (8, 2, 2),
        }
        for case in build_swizzle_comparison_suite().cases:
            with self.subTest(pattern=case.pattern.value):
                executable = provider.lower_link(
                    case, SwizzleComparisonBranch.NAIVE
                )
                executable.validate()
                fragment = executable.manifest.fragments[0]
                self.assertEqual(
                    executable.manifest.producer_pass,
                    "unfused_comparison_standard_linker",
                )
                self.assertEqual(len(executable.manifest.input_digests), 8)
                self.assertEqual(
                    (
                        sum(len(item.records) for item in fragment.core_streams),
                        len(fragment.buffer_abi),
                    ),
                    expected[case.pattern.value],
                )
                contract = provider.build_actual_sha_program_io(
                    executable, _ACTUAL_SHA
                )
                contract.validate_against(executable.manifest)
                self.assertEqual(contract.program_artifact_sha256, _ACTUAL_SHA)
                self.assertEqual(
                    (
                        len(contract.initializations),
                        len(contract.output_probes),
                        len(contract.blobs),
                    ),
                    expected_io[case.pattern.value],
                )


if __name__ == "__main__":
    unittest.main()
