"""Real two-step SGD source tests for external-only physical StateABI seeds."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.external_state_authority import (
    defer_external_state_initializations,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramHbmTarget,
    ProgramIoContract,
)
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryObjectKind

from .run_dense_external_offload_runtime_canary import _build_offload_case
from .run_dense_training_sequence_runtime_canary import _offload_sequence


class ExternalStateAuthorityProgramIoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sequence = _offload_sequence()
        linked = sequence.segments[0].linked_program
        manifest = linked.manifest
        seeds, expected = build_deterministic_timing_state_overrides(linked)
        assert not expected
        first_fragment = manifest.fragments[0]
        state_abis = (
            first_fragment.fragment.state_abi
            if hasattr(first_fragment, "fragment") else first_fragment.state_abi
        )
        ordered = sorted(state_abis, key=lambda abi: abi.address)
        payload = b"".join(seeds[abi.state_ref] for abi in ordered)
        assert all(
            abi.address == sum(item.size_bytes for item in ordered[:index])
            for index, abi in enumerate(ordered)
        )
        resident_peak = max(
            peak.peak_bytes for peak in sequence.materialization.memory_plan.peaks
        )
        parameter_bytes = sum(
            item.size_bytes for item in sequence.materialization.state_inventory
            if item.object_kind is MemoryObjectKind.PARAMETER
        )
        _, _, program, *_ = _build_offload_case(
            sequence.materialization,
            hbm_bytes_override=resident_peak - parameter_bytes,
            dirty_writeback=True,
            payload_override=payload,
        )
        cls.manifest = manifest
        cls.program = program
        cls.seeds = seeds
        cls.contract = build_timing_program_io(
            linked, "0" * 64, state_seed_overrides=seeds,
        )

    def test_real_15_state_linked_offload_defers_every_hbm_host_seed(self) -> None:
        deferred = defer_external_state_initializations(
            self.contract, self.manifest, self.program, self.seeds,
        )
        deferred.validate_against(self.manifest)
        self.assertEqual(
            sum(isinstance(item.target, ProgramHbmTarget)
                for item in self.contract.initializations), 15,
        )
        self.assertEqual(
            sum(isinstance(item.target, ProgramHbmTarget)
                for item in deferred.initializations), 0,
        )
        self.assertEqual(len(deferred.initializations),
                         len(self.contract.initializations) - 15)
        self.assertEqual(deferred.source_linked_manifest_digest,
                         self.contract.source_linked_manifest_digest)
        self.assertEqual(deferred.program_artifact_sha256,
                         self.contract.program_artifact_sha256)

    def test_external_payload_disagrees_with_real_state_seed(self) -> None:
        seeds = dict(self.seeds)
        state_ref = next(iter(seeds))
        seeds[state_ref] = bytes([seeds[state_ref][0] ^ 1]) + seeds[state_ref][1:]
        with self.assertRaisesRegex(SchemaError, "authentic payload"):
            defer_external_state_initializations(
                self.contract, self.manifest, self.program, seeds,
            )

    def test_missing_hbm_seed_before_deferral_rejected(self) -> None:
        missing = tuple(
            item for item in self.contract.initializations
            if not (
                isinstance(item.target, ProgramHbmTarget) and
                item.target.state_ref == next(iter(self.seeds))
            )
        )
        contract = ProgramIoContract.create(
            producer_pass="external_authority_negative_fixture",
            mode=self.contract.mode,
            source_manifest=self.manifest,
            program_artifact_sha256=self.contract.program_artifact_sha256,
            blobs=self.contract.blobs,
            initializations=missing,
            output_probes=self.contract.output_probes,
        )
        with self.assertRaisesRegex(SchemaError, "all deferred StateABIs"):
            defer_external_state_initializations(
                contract, self.manifest, self.program, self.seeds,
            )

    def test_final_writeback_phase_missing_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            defer_external_state_initializations(
                self.contract, self.manifest,
                replace(self.program, descriptors=self.program.descriptors[:1]),
                self.seeds,
            )


if __name__ == "__main__":
    unittest.main()
