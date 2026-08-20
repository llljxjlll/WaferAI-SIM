from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
    ManifestInputKind,
    RecordOpcode,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.lite_moe_backward import (
    S3_LITE_MOE_BACKWARD_CASE_ID,
)
from llm.frontend.wafer_frontend.schema.lite_moe_backward_n6 import (
    LiteMoeBackwardLinkedProgram,
)

from lite_moe_backward_cases import build_lite_moe_backward_case


class LiteMoeBackwardCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_lite_moe_backward_case()

    def test_exact_production_overlay_and_oracle(self) -> None:
        overlay = self.case.overlay
        self.assertEqual(self.case.case_id, S3_LITE_MOE_BACKWARD_CASE_ID)
        self.assertEqual(
            (
                len(overlay.remote_grad_dtes),
                len(overlay.token_wgrads),
                len(overlay.expert_reduces),
                len(overlay.sgd_stores),
                len(overlay.trainable_down_states),
                overlay.oracle.remote_grad_bytes_total,
                overlay.oracle.token_wgrad_bytes_total,
                overlay.oracle.sgd_store_count
                * overlay.oracle.state_store_bytes_each,
            ),
            (4, 8, 4, 4, 4, 128, 16384, 4096),
        )
        self.assertEqual(
            tuple(item.declaration.id for item in overlay.trainable_down_states),
            tuple(item.down_weight_state_ref for item in overlay.sgd_stores),
        )
        self.assertEqual(
            tuple(item.binding.id for item in overlay.trainable_down_states),
            tuple(item.down_weight_hbm_binding_ref for item in overlay.sgd_stores),
        )
        self.assertEqual(
            tuple(item.token_index for item in overlay.remote_grad_dtes),
            (1, 3, 4, 6),
        )
        self.assertEqual(
            tuple(item.deps for item in overlay.expert_reduces),
            tuple(
                tuple(
                    item.id
                    for item in overlay.token_wgrads
                    if item.expert_index == expert
                )
                for expert in range(4)
            ),
        )
        self.assertEqual(
            tuple(item.deps for item in overlay.sgd_stores),
            tuple((item.id,) for item in overlay.expert_reduces),
        )

    def test_exact_linked_manifest_and_typed_stage_order(self) -> None:
        manifest = self.case.linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                sum(
                    len(stream.records)
                    for fragment in manifest.fragments
                    for stream in fragment.core_streams
                ),
                len(manifest.input_digests),
                len(manifest.core_streams),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
            ),
            (28, 104, 33, 2, 18, 69, 180, 8),
        )
        self.assertEqual(
            Counter(
                record.opcode
                for fragment in manifest.fragments
                for stream in fragment.core_streams
                for record in stream.records
            ),
            Counter(
                {
                    RecordOpcode.SRAM_ALLOC_AT: 28,
                    RecordOpcode.SRAM_FREE: 28,
                    RecordOpcode.SRAM_BIND: 12,
                    RecordOpcode.MATMUL: 8,
                    RecordOpcode.DTE_SEND: 4,
                    RecordOpcode.DTE_RECV: 4,
                    RecordOpcode.DTE_WAIT: 4,
                    RecordOpcode.LOCAL_REDUCE: 4,
                    RecordOpcode.LSU_LOAD: 4,
                    RecordOpcode.SGD_UPDATE: 4,
                    RecordOpcode.LSU_STORE: 4,
                }
            ),
        )
        top = tuple(
            item for item in manifest.input_digests
            if item.kind is ManifestInputKind.S3_LITE_MOE
        )
        global_inputs = tuple(
            item for item in manifest.input_digests
            if item.kind is ManifestInputKind.GLOBAL_ACTION_DAG
        )
        self.assertEqual(tuple(item.artifact_id for item in top), (self.case.lowered.id,))
        self.assertEqual(
            tuple(item.artifact_id for item in global_inputs),
            (self.case.overlay.id,),
        )

        overlay = self.case.overlay
        loads = {
            f"{overlay.id}.expert{item.expert_index}.weight_load"
            for item in overlay.trainable_down_states
        }
        dte = {
            action
            for unit in overlay.remote_grad_dtes
            for action in (f"{unit.id}.send", f"{unit.id}.recv", f"{unit.id}.wait")
        }
        wgrad = {item.id for item in overlay.token_wgrads}
        reduce = {item.id for item in overlay.expert_reduces}
        sgd = {item.id for item in overlay.sgd_stores}
        stages = (loads, dte, wgrad, reduce, sgd)
        for stream in manifest.core_streams:
            action_sequence = []
            for record in stream.records:
                if (
                    not action_sequence
                    or action_sequence[-1] != record.source_global_action_id
                ):
                    action_sequence.append(record.source_global_action_id)
            stage_sequence = tuple(
                next(index for index, members in enumerate(stages) if action in members)
                for action in action_sequence
            )
            self.assertEqual(stage_sequence, tuple(sorted(stage_sequence)))
            self.assertEqual(set(stage_sequence), set(range(5)))

    def test_restable_core_stream_tamper_rejected_by_linked_carrier(self) -> None:
        manifest = self.case.linked.manifest
        first = manifest.core_streams[0]
        forged_stream = replace(
            first,
            records=(*first.records[1:], first.records[0]),
        )
        semantic = {
            field.name: getattr(manifest, field.name)
            for field in fields(manifest)
            if field.name not in ("schema_version", "producer_pass", "id")
        }
        semantic["core_streams"] = (forged_stream, *manifest.core_streams[1:])
        forged = LinkedProgramManifest.create(
            producer_pass=manifest.producer_pass,
            **semantic,
        )
        forged.validate("forged")
        with self.assertRaisesRegex(
            SchemaError,
            "manifest is not the exact production backward quotient",
        ):
            LiteMoeBackwardLinkedProgram.create(
                source=self.case.lowered,
                manifest=forged,
            )

    def test_reduce_source_bindings_cover_two_contiguous_contributions(self) -> None:
        manifest = self.case.linked.manifest
        fragments = {fragment.id: fragment for fragment in manifest.fragments}
        definitions = {
            item.symbol.id: item
            for item in manifest.program_symbol_definitions
        }
        reduce_bindings = []
        for binding in manifest.address_operand_bindings:
            fragment = fragments[binding.fragment_id]
            record = fragment.core_streams[0].records[binding.fragment_record_index]
            if (
                record.opcode is RecordOpcode.LOCAL_REDUCE
                and binding.operand_id is SemanticOperandId.SOURCE_ADDRESS
            ):
                reduce_bindings.append((fragment, binding))
        self.assertEqual(len(reduce_bindings), 4)
        for fragment, binding in reduce_bindings:
            self.assertEqual(len(binding.buffer_abi_ids), 2)
            self.assertEqual(len(binding.tensor_slices), 2)
            abis = tuple(
                next(item for item in fragment.buffer_abi if item.id == abi_id)
                for abi_id in binding.buffer_abi_ids
            )
            self.assertEqual(tuple(item.size_bytes for item in abis), (2048, 2048))
            self.assertEqual(
                abis[1].region_offset_bytes,
                abis[0].region_offset_bytes + 2048,
            )
            unit = next(
                item
                for item in self.case.overlay.expert_reduces
                if item.id
                == fragment.core_streams[0].records[
                    binding.fragment_record_index
                ].source_global_action_id
            )
            self.assertEqual(
                tuple(item.value_id for item in abis),
                unit.contribution_refs,
            )
            record = fragment.core_streams[0].records[
                binding.fragment_record_index
            ]
            source_symbol = next(
                operand.symbol_ref
                for operand in record.operands
                if operand.name == "source_address"
            )
            destination_symbol = next(
                operand.symbol_ref
                for operand in record.operands
                if operand.name == "destination_address"
            )
            self.assertNotEqual(source_symbol, destination_symbol)
            self.assertEqual(definitions[source_symbol].size_bytes, 4096)
            self.assertEqual(definitions[destination_symbol].size_bytes, 2048)
        self.assertEqual(
            len({
                next(
                    operand.symbol_ref
                    for operand in fragment.core_streams[0].records[
                        binding.fragment_record_index
                    ].operands
                    if operand.name == "source_address"
                )
                for fragment, binding in reduce_bindings
            }),
            4,
        )

    def test_overlay_and_lineage_tamper_fail_closed(self) -> None:
        overlay = self.case.overlay
        remote = overlay.remote_grad_dtes[0]
        reduce = overlay.expert_reduces[0]
        mutations = (
            replace(
                overlay,
                remote_grad_dtes=(
                    replace(remote, bytes=31),
                    *overlay.remote_grad_dtes[1:],
                ),
            ),
            replace(
                overlay,
                expert_reduces=(
                    replace(reduce, deps=tuple(reversed(reduce.deps))),
                    *overlay.expert_reduces[1:],
                ),
            ),
        )
        for forged in mutations:
            with self.subTest(), self.assertRaises(SchemaError):
                replace(self.case, overlay=forged).validate()


if __name__ == "__main__":
    unittest.main()
