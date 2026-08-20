from __future__ import annotations

from collections import Counter
import json
import unittest

from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.ir0 import StateAccessMode
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from stage3_decode_cases import (
    Stage3StaticCaseKind,
    build_stage3_decode_case,
    build_stage3_static_case,
)


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


class Stage3DecodeCaseTest(unittest.TestCase):
    def _assert_policy_identity(self, case) -> None:
        case.policy.validate_against(
            case.planning_context,
            case.scheduling_context,
        )
        self.assertEqual(
            tuple(
                (
                    selection.kind.value,
                    selection.name,
                    selection.implementation_id,
                    selection.implementation_schema_version,
                    selection.capability_ids,
                )
                for selection in case.policy.selections
            ),
            (
                (
                    "inter_die",
                    "naive",
                    "wafer_frontend.policy.inter_die.naive",
                    "wafer_frontend.naive_inter_die_policy/v1",
                    ("s1.gemm_collective.naive",),
                ),
                (
                    "standalone_collective",
                    "direct_all_gather",
                    "wafer_frontend.policy.standalone.direct_all_gather",
                    "wafer_frontend.direct_all_gather_policy/v1",
                    ("s1.gemm_collective.naive",),
                ),
                (
                    "intra_die",
                    "naive",
                    "wafer_frontend.policy.intra_die.naive",
                    "wafer_frontend.naive_intra_die_policy/v8",
                    ("s1.gemm_collective.naive",),
                ),
            ),
        )

    def test_tp1_prefill_and_mixed_are_deterministic_and_exact(self) -> None:
        goldens = {
            Stage3StaticCaseKind.PREFILL: (
                (44, 44, 159, 278),
                4,
                15,
                0,
                60,
                1,
                72,
                0,
                1024,
                2048,
            ),
            Stage3StaticCaseKind.MIXED: (
                (80, 80, 235, 374),
                24,
                31,
                16,
                96,
                17,
                94,
                5120,
                1024,
                12288,
            ),
        }
        for kind, golden in goldens.items():
            with self.subTest(kind=kind.value):
                first = build_stage3_static_case(kind, 1)
                second = build_stage3_static_case(kind, 1)
                self._assert_policy_identity(first)
                self.assertEqual(_manifest_counts(first)[:4], golden[0])
                state_manifest = first.graph.persistent_state_manifest
                self.assertIsNotNone(state_manifest)
                assert state_manifest is not None
                parameter_refs = {
                    item.id
                    for item in state_manifest.declarations
                    if item.identity.kind is StateKind.PARAMETER
                }
                kv_refs = {
                    item.id
                    for item in state_manifest.declarations
                    if item.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
                }
                read_write_refs = {
                    access.state_ref
                    for access in first.graph.state_accesses
                    if access.state_ref in kv_refs
                    and access.mode is StateAccessMode.READ_WRITE
                }
                self.assertEqual(len(kv_refs), golden[1])
                self.assertEqual(set(first.state_seed_refs), parameter_refs | read_write_refs)
                self.assertEqual(set(first.state_expected_refs), read_write_refs)
                self.assertEqual(len(first.state_seed_refs), golden[2])
                self.assertEqual(len(first.state_expected_refs), golden[3])
                self.assertEqual(len(first.program_io.initializations), golden[4])
                self.assertEqual(len(first.program_io.output_probes), golden[5])
                self.assertEqual(
                    (
                        first.oracle.logical_work.attention.query_key_pairs,
                        first.oracle.kv.logical_read_bytes,
                        first.oracle.kv.logical_write_bytes,
                        first.oracle.kv.logical_reserved_bytes,
                    ),
                    golden[6:],
                )
                for name in (
                    "template",
                    "static_profile",
                    "oracle",
                    "policy",
                    "planning_context",
                    "scheduling_context",
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

    def test_tp1_is_deterministic_and_closes_kv_program_io(self) -> None:
        first = build_stage3_decode_case(1)
        second = build_stage3_decode_case(1)
        self._assert_policy_identity(first)
        counts = _manifest_counts(first)
        self.assertEqual(counts[:4], (104, 104, 275, 422))
        self.assertEqual(
            counts[4],
            {
                "ATTENTION_EXACT": 2,
                "EMBEDDING_LOOKUP": 1,
                "LSU_LOAD": 47,
                "LSU_STORE": 32,
                "MATMUL": 9,
                "RESIDUAL": 4,
                "RMSNORM": 5,
                "ROPE_QK_EXACT": 2,
                "SRAM_ALLOC_AT": 73,
                "SRAM_BIND": 25,
                "SRAM_FREE": 73,
                "SWIGLU": 2,
            },
        )
        first.oracle.validate_against_template(first.template)
        self.assertEqual(first.oracle.kv.logical_read_bytes, 18432)
        self.assertEqual(first.oracle.kv.logical_write_bytes, 1024)
        self.assertEqual(first.oracle.kv.logical_reserved_bytes, 24576)

        state_manifest = first.graph.persistent_state_manifest
        self.assertIsNotNone(state_manifest)
        assert state_manifest is not None
        parameter_refs = {
            item.id
            for item in state_manifest.declarations
            if item.identity.kind is StateKind.PARAMETER
        }
        kv_refs = {
            item.id
            for item in state_manifest.declarations
            if item.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        }
        self.assertEqual(len(parameter_refs), 15)
        self.assertEqual(len(kv_refs), 32)
        self.assertEqual(set(first.state_seed_refs), parameter_refs | kv_refs)
        self.assertEqual(set(first.state_expected_refs), kv_refs)
        self.assertEqual(len(first.program_io.initializations), 120)
        self.assertEqual(len(first.program_io.output_probes), 33)

        hardware = json.loads(first.runtime_hardware_inputs.hardware_json)
        regions = hardware["memory"]["sram"]["regions"]
        self.assertEqual(hardware["memory"]["sram_size"], 64 * 1024)
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
                (0, 48 * 1024, "block"),
                (48 * 1024, 4 * 1024, "block"),
                (52 * 1024, 4 * 1024, "block"),
                (56 * 1024, 4 * 1024, "block"),
                (60 * 1024, 4 * 1024, "block"),
            ),
        )

        for name in (
            "template",
            "static_profile",
            "oracle",
            "policy",
            "planning_context",
            "scheduling_context",
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


if __name__ == "__main__":
    unittest.main()
