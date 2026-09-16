"""Single-rank full-model MoE gate MATMUL extent regression."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest

from llm.frontend.wafer_frontend.passes.load_fabric import (
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import (
    compile_moe_full_model_inference_sequence,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
    RecordOpcode,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_moe_compile_sequence import _manifest
from llm.test.frontend.unit.test_moe_full_model_compile_sequence import (
    _legacy_template,
)


_ROOT = Path(__file__).resolve().parents[4]
_FINALIZER = _ROOT / "build/npusim_program_finalizer"
_GATE_PARAMETERS = (1, 4, 4, 1)


def _full_model_manifest():
    materialization = _manifest(
        WorkloadFamily.MOE_INFERENCE,
        rows=1,
        columns=1,
    )
    fabric = physical_fabric_from_data(
        minimal_hardware(1, 1, sram_bytes=65536)
    )
    sequence = compile_moe_full_model_inference_sequence(
        materialization,
        _legacy_template(),
        fabric,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    sequence.validate()
    return sequence.segments[0].executable_manifest


def _gate_records_and_bindings(manifest):
    result = []
    for fragment in manifest.fragments:
        if fragment.producer_pass != "flexible_moe_production_lowering":
            continue
        for stream in fragment.core_streams:
            for record_index, record in enumerate(stream.records):
                if (
                    record.opcode is not RecordOpcode.MATMUL
                    or record.operands[-1].literal_value != _GATE_PARAMETERS
                ):
                    continue
                binding = next(
                    item
                    for item in manifest.address_operand_bindings
                    if item.fragment_id == fragment.id
                    and item.fragment_record_index == record_index
                    and item.operand_id
                    is SemanticOperandId.COMPUTE_DATA_ADDRESS
                )
                result.append((record, binding))
    return tuple(result)


def _run_finalizer(manifest, root: Path, stem: str):
    linked = root / f"{stem}.linked.json"
    artifact = root / f"{stem}.npup"
    report = root / f"{stem}.finalizer.json"
    linked.write_text(canonical_json(manifest), encoding="utf-8")
    return subprocess.run(
        [
            str(_FINALIZER),
            "--input",
            str(linked),
            "--output",
            str(artifact),
            "--report",
            str(report),
        ],
        cwd=_ROOT / "llm",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=120,
    )


class FlexibleMoeSingleRankGateFinalizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _full_model_manifest()

    def test_full_model_gate_parameters_match_m_h_e_and_weight_abi(self) -> None:
        pairs = _gate_records_and_bindings(self.manifest)
        self.assertEqual(len(pairs), 2)
        all_abis = {
            abi.id: abi
            for fragment in self.manifest.fragments
            for abi in fragment.buffer_abi
        }
        for record, binding in pairs:
            with self.subTest(action=record.source_global_action_id):
                self.assertEqual(
                    record.operands[-1].literal_value,
                    _GATE_PARAMETERS,
                )
                self.assertEqual(binding.tensor_slices[0].shape, (4,))
                self.assertEqual(
                    all_abis[binding.buffer_abi_ids[0]].size_bytes,
                    8,
                )

    @unittest.skipUnless(
        _FINALIZER.is_file(),
        "requires the production npusim_program_finalizer binary",
    )
    def test_finalizer_accepts_exact_full_model_gate_extent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="moe-gate-finalizer-") as raw:
            finalized = _run_finalizer(
                self.manifest, Path(raw), "gate-exact"
            )
        self.assertEqual(finalized.returncode, 0, finalized.stdout)

    @unittest.skipUnless(
        _FINALIZER.is_file(),
        "requires the production npusim_program_finalizer binary",
    )
    def test_finalizer_rejects_short_full_model_gate_weight_extent(self) -> None:
        manifest = self.manifest
        _, binding = _gate_records_and_bindings(manifest)[0]
        tensor_slice = binding.tensor_slices[0]
        short_binding = replace(
            binding,
            tensor_slices=(
                replace(tensor_slice, shape=(tensor_slice.shape[0] - 1,)),
            ),
        )
        semantic = manifest._semantic_key()
        semantic["address_operand_bindings"] = tuple(
            short_binding if item == binding else item
            for item in manifest.address_operand_bindings
        )
        broken = LinkedProgramManifest.create(
            producer_pass=manifest.producer_pass,
            **semantic,
        )
        with tempfile.TemporaryDirectory(prefix="moe-gate-finalizer-") as raw:
            finalized = _run_finalizer(broken, Path(raw), "gate-short")
        self.assertNotEqual(finalized.returncode, 0)
        self.assertIn(
            "payload byte extent does not fit its dense view/root",
            finalized.stdout,
        )


if __name__ == "__main__":
    unittest.main()
