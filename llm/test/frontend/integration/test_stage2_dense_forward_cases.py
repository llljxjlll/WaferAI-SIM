from __future__ import annotations

from collections import Counter
import json
import unittest

from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from stage2_dense_forward_cases import build_stage2_dense_forward_case


def _manifest_counts(case):
    leaves = tuple(
        item.fragment if hasattr(item, "fragment") else item
        for item in case.manifest.fragments
    )
    opcodes = Counter(
        record.opcode.name
        for fragment in leaves
        for stream in fragment.core_streams
        for record in stream.records
    )
    return (
        len(case.global_dag.actions),
        len(leaves),
        sum(opcodes.values()),
        len(case.manifest.address_operand_bindings),
        dict(opcodes),
    )


class Stage2DenseForwardCaseTest(unittest.TestCase):
    def test_tp1_is_self_contained_deterministic_and_state_exact(self) -> None:
        first = build_stage2_dense_forward_case(1)
        second = build_stage2_dense_forward_case(1)

        self.assertEqual(
            _manifest_counts(first)[:4],
            (44, 44, 159, 278),
        )
        self.assertEqual(
            _manifest_counts(first)[4],
            {
                "ATTENTION_EXACT": 2,
                "EMBEDDING_LOOKUP": 1,
                "LSU_LOAD": 15,
                "LSU_STORE": 4,
                "MATMUL": 9,
                "RESIDUAL": 4,
                "RMSNORM": 5,
                "ROPE_QK_EXACT": 2,
                "SRAM_ALLOC_AT": 45,
                "SRAM_BIND": 25,
                "SRAM_FREE": 45,
                "SWIGLU": 2,
            },
        )
        first.oracle.validate_against_template(first.template)
        self.assertEqual(first.oracle.parameters.placed_bytes, 12448)
        self.assertEqual(first.oracle.graph.node_count, 25)

        state_manifest = first.graph.persistent_state_manifest
        self.assertIsNotNone(state_manifest)
        assert state_manifest is not None
        parameter_refs = {
            state.id
            for state in state_manifest.declarations
            if state.identity.kind is StateKind.PARAMETER
        }
        kv_refs = {
            state.id
            for state in state_manifest.declarations
            if state.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        }
        self.assertEqual(len(parameter_refs), 15)
        self.assertEqual(len(kv_refs), 4)
        self.assertEqual(set(first.state_seed_refs), parameter_refs)
        self.assertTrue(kv_refs.isdisjoint(first.state_seed_refs))
        self.assertEqual(first.state_expected_refs, ())
        self.assertEqual(len(first.program_io.initializations), 60)
        self.assertEqual(len(first.program_io.output_probes), 1)

        hardware = json.loads(first.runtime_hardware_inputs.hardware_json)
        regions = hardware["memory"]["sram"]["regions"]
        self.assertEqual(hardware["memory"]["sram_size"], 64 * 1024)
        self.assertEqual(hardware["memory"]["sram"]["capacity_bytes"], 64 * 1024)
        self.assertEqual(
            tuple(
                (
                    region["base_bytes"],
                    region["size_bytes"],
                    region["allocator"],
                )
                for region in regions
            ),
            (
                (0, 32 * 1024, "block"),
                (32 * 1024, 8 * 1024, "block"),
                (40 * 1024, 8 * 1024, "block"),
                (48 * 1024, 8 * 1024, "block"),
                (56 * 1024, 8 * 1024, "block"),
            ),
        )
        self.assertEqual(
            tuple(region["base_bytes"] + region["size_bytes"] for region in regions),
            (32 * 1024, 40 * 1024, 48 * 1024, 56 * 1024, 64 * 1024),
        )

        for name in (
            "template",
            "oracle",
            "global_dag",
            "lowered",
            "manifest",
            "program_io",
        ):
            self.assertEqual(
                canonical_digest(getattr(first, name)),
                canonical_digest(getattr(second, name)),
                name,
            )
        self.assertEqual(
            first.runtime_hardware_inputs,
            second.runtime_hardware_inputs,
        )

    def test_tp2_tp4_manifest_oracle_and_determinism_are_exact(self) -> None:
        expected = {
            2: (
                (160, 92, 510, 804),
                14656,
                {
                    "ATTENTION_EXACT": 4,
                    "DTE_ISSUE": 8,
                    "DTE_RECV": 16,
                    "DTE_SEND": 16,
                    "DTE_WAIT": 16,
                    "EMBEDDING_LOOKUP": 2,
                    "EVENT_SET": 8,
                    "EVENT_WAIT": 8,
                    "LOCAL_REDUCE": 8,
                    "LSU_LOAD": 30,
                    "LSU_STORE": 8,
                    "MATMUL": 26,
                    "RESIDUAL": 8,
                    "RMSNORM": 10,
                    "ROPE_QK_EXACT": 4,
                    "SRAM_ALLOC_AT": 138,
                    "SRAM_BIND": 58,
                    "SRAM_FREE": 138,
                    "SWIGLU": 4,
                },
            ),
            4: (
                (544, 180, 1452, 2184),
                19072,
                {
                    "ATTENTION_EXACT": 8,
                    "DTE_ISSUE": 16,
                    "DTE_RECV": 96,
                    "DTE_SEND": 96,
                    "DTE_WAIT": 64,
                    "EMBEDDING_LOOKUP": 4,
                    "EVENT_SET": 24,
                    "EVENT_WAIT": 24,
                    "LOCAL_REDUCE": 16,
                    "LSU_LOAD": 60,
                    "LSU_STORE": 16,
                    "MATMUL": 84,
                    "RESIDUAL": 16,
                    "RMSNORM": 20,
                    "ROPE_QK_EXACT": 8,
                    "SRAM_ALLOC_AT": 372,
                    "SRAM_BIND": 148,
                    "SRAM_FREE": 372,
                    "SWIGLU": 8,
                },
            ),
        }
        for tp_degree, (counts, placed_bytes, opcode_counts) in expected.items():
            with self.subTest(tp_degree=tp_degree):
                first = build_stage2_dense_forward_case(tp_degree)
                second = build_stage2_dense_forward_case(tp_degree)
                actual = _manifest_counts(first)
                self.assertEqual(actual[:4], counts)
                self.assertEqual(actual[4], opcode_counts)
                first.oracle.validate_against_template(first.template)
                self.assertEqual(first.oracle.parameters.placed_bytes, placed_bytes)
                self.assertEqual(first.oracle.graph.node_count, 33)

                state_manifest = first.graph.persistent_state_manifest
                self.assertIsNotNone(state_manifest)
                assert state_manifest is not None
                parameter_refs = {
                    state.id
                    for state in state_manifest.declarations
                    if state.identity.kind is StateKind.PARAMETER
                }
                kv_refs = {
                    state.id
                    for state in state_manifest.declarations
                    if state.identity.kind
                    in (StateKind.KV_KEY, StateKind.KV_VALUE)
                }
                self.assertEqual(len(parameter_refs), 15 * tp_degree)
                self.assertEqual(len(kv_refs), 4 * tp_degree)
                self.assertEqual(set(first.state_seed_refs), parameter_refs)
                self.assertTrue(kv_refs.isdisjoint(first.state_seed_refs))
                self.assertEqual(first.state_expected_refs, ())

                for name in (
                    "template",
                    "oracle",
                    "global_dag",
                    "lowered",
                    "manifest",
                    "program_io",
                ):
                    self.assertEqual(
                        canonical_digest(getattr(first, name)),
                        canonical_digest(getattr(second, name)),
                        f"tp={tp_degree} {name}",
                    )
                self.assertEqual(
                    first.runtime_hardware_inputs,
                    second.runtime_hardware_inputs,
                )


if __name__ == "__main__":
    unittest.main()
