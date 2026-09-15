"""Real leaf source span proof, including native WGRAD fault injection."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode, SemanticOperandId,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_gradient_source_bridge import (
    build_moe_full_train_gradient_source_bridge,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_named_wgrad_tiles import (
    build_moe_full_train_named_wgrad_tiles,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_named_wgrad_operands import (
    build_moe_full_train_physical_wgrad_operand_sources,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeFullTrainNamedWgradOperandsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        tiles = build_moe_full_train_named_wgrad_tiles(
            Fixture.phase, Fixture.sequence, Fixture.placement,
            original_dense=Fixture.dense, dense_manifest=Fixture.manifest,
            context=Fixture.context,
        )
        cls.bridge = build_moe_full_train_gradient_source_bridge(
            Fixture.phase, tiles, Fixture.sequence, Fixture.placement,
            original_dense=Fixture.dense, dense_manifest=Fixture.manifest,
            context=Fixture.context,
        )
        cls.evidence = build_moe_full_train_physical_wgrad_operand_sources(
            cls.bridge, Fixture.sequence,
        )

    def _check(self, evidence):
        evidence.validate_against(self.bridge, Fixture.sequence)

    def test_producer_real_forward_and_dgrad_three_nonoverlap_fp32_slices(self):
        self._check(self.evidence)
        self.assertEqual(len(self.evidence.entries), 12)
        for layer in (0, 1):
            for expert in (0, 1):
                entries = [entry for entry in self.evidence.entries
                           if (entry.layer, entry.expert) == (layer, expert)]
                with self.subTest(layer=layer, expert=expert):
                    self.assertEqual([entry.projection for entry in entries],
                                     ["gate", "up", "down"])
                    self.assertEqual([entry.derivative_upstream.offset_bytes
                                      for entry in entries], [0, 32, 0])
                    self.assertEqual([entry.derivative_upstream.size_bytes
                                      for entry in entries], [32, 32, 16])
                    self.assertEqual([entry.forward_activation.size_bytes
                                      for entry in entries], [16, 16, 32])
                    self.assertEqual([entry.fp32_gradient_output.offset_bytes
                                      for entry in entries], [0, 128, 256])
                    self.assertEqual([entry.fp32_gradient_output.size_bytes
                                      for entry in entries], [128, 128, 128])

    def _fault_sequence(self, entry, *, opcode=None, upstream_offset=None):
        units = list(Fixture.sequence.units)
        unit_index = next(index for index, unit in enumerate(units)
                          if (unit.step, unit.layer) == (0, entry.layer))
        unit = units[unit_index]
        manifest = unit.linked_manifest
        fragments = list(manifest.fragments)
        for fragment_index, fragment in enumerate(fragments):
            streams = list(fragment.core_streams)
            for stream_index, stream in enumerate(streams):
                if stream.logical_core.die_id != entry.expert:
                    continue
                for record_index, record in enumerate(stream.records):
                    if record.source_global_action_id != entry.native_record_action_ref:
                        continue
                    if record.opcode is not RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING:
                        continue
                    records = list(stream.records)
                    relocs = list(stream.address_relocations)
                    if opcode is not None:
                        records[record_index] = replace(record, opcode=opcode)
                    if upstream_offset is not None:
                        reloc_index = next(index for index, reloc in enumerate(relocs)
                                           if reloc.record_index == record_index
                                           and reloc.operand_id is SemanticOperandId.COMPUTE_DATA_ADDRESS)
                        relocs[reloc_index] = replace(relocs[reloc_index],
                                                      addend=upstream_offset)
                    streams[stream_index] = replace(stream,
                        records=tuple(records), address_relocations=tuple(relocs))
                    fragments[fragment_index] = replace(fragment,
                        core_streams=tuple(streams))
                    units[unit_index] = replace(unit,
                        linked_manifest=replace(manifest, fragments=tuple(fragments)))
                    return replace(Fixture.sequence, units=tuple(units))
        self.fail("actual source-backed native expert WGRAD record missing")

    def test_gate_up_native_wgrad_consumes_swiglu_backward_and_rejects_wrong_half(self):
        self.evidence.require_gate_up_derivative_consumption(Fixture.sequence)
        up = next(entry for entry in self.evidence.entries
                  if entry.projection == "up")
        wrong = self._fault_sequence(up, upstream_offset=0)
        with self.assertRaisesRegex(SchemaError, "operand tensor slice does not cover exact WGRAD span"):
            self.evidence.require_gate_up_derivative_consumption(wrong)

    def test_three_native_fp32_projections_and_reject_matmul_substitution(self):
        self.evidence.require_native_fp32_producers(Fixture.sequence)
        down = next(entry for entry in self.evidence.entries
                    if entry.layer == 1 and entry.projection == "down")
        wrong = self._fault_sequence(down, opcode=RecordOpcode.MATMUL)
        with self.assertRaisesRegex(SchemaError, "MATMUL.*LOCAL_REDUCE.*native 0x25"):
            self.evidence.require_native_fp32_producers(wrong)

    def test_last_down_or_gate_upstream_source_cannot_be_swapped(self):
        for position in (0, 11):
            entries = list(self.evidence.entries)
            entries[position] = replace(
                entries[position], derivative_upstream=
                entries[position].forward_activation)
            with self.subTest(position=position), self.assertRaisesRegex(
                    SchemaError, "forward, backward SwiGLU"):
                self._check(replace(self.evidence, entries=tuple(entries)))

    def test_gate_up_second_half_or_fp32_storage_cannot_overlap(self):
        entry = self.evidence.entries[1]
        wrong = replace(entry.derivative_upstream, offset_bytes=0)
        with self.assertRaisesRegex(SchemaError, "all named FP32"):
            self._check(replace(self.evidence,
                entries=(*self.evidence.entries[:1],
                         replace(entry, derivative_upstream=wrong),
                         *self.evidence.entries[2:])))
        entry = self.evidence.entries[-1]
        wrong = replace(entry.fp32_gradient_output, offset_bytes=128)
        with self.assertRaisesRegex(SchemaError, "all named FP32"):
            self._check(replace(self.evidence,
                entries=(*self.evidence.entries[:-1],
                         replace(entry, fp32_gradient_output=wrong))))


if __name__ == "__main__":
    unittest.main()
