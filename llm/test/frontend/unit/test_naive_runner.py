from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.pass_manager import (
    PASS_SPECS,
    PassReceipt,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.runner import (
    NaiveRunCase,
    NaiveRunReport,
    NaiveRunRequest,
    NaiveRunValidation,
    run_naive,
)
from llm.frontend.wafer_frontend.schema.policy import PolicySelection
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
    to_primitive,
)


def _compile_provenance() -> dict[str, object]:
    registry = production_registry()
    selections = (
        registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
        registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE,
            "direct_all_gather",
        ).selection,
        registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection,
    )
    stage_digests = tuple(f"{index:064x}" for index in range(11))
    context_passes = {
        "placement",
        "fusion_partition",
        "inter_die_plan",
        "project_to_ir2",
        "intra_die_schedule",
    }
    receipts = []
    for index, spec in enumerate(PASS_SPECS[:10]):
        if spec.name == "inter_die_plan":
            receipt_selections = selections[:2]
        elif spec.name == "intra_die_schedule":
            receipt_selections = selections[2:]
        else:
            receipt_selections = ()
        receipt = PassReceipt(
            pass_name=spec.name,
            input_phase=spec.input_phase,
            output_phase=spec.output_phase,
            input_digest=stage_digests[index],
            context_digest=(
                "f" * 64 if spec.name in context_passes else None
            ),
            output_digest=stage_digests[index + 1],
            policy_selections=receipt_selections,
        )
        receipt.validate()
        receipts.append(to_primitive(receipt))
    return {
        "profile_id": "profile",
        "profile_weight": 1.0,
        "pass_receipts": tuple(receipts),
        "stage_digests": stage_digests,
        "context_ids": tuple(f"context_{index}" for index in range(5)),
        "policy_selections": tuple(
            to_primitive(selection) for selection in selections
        ),
        "linked_bundle_id": "bundle",
        "linked_profile_id": "entry",
        "linked_manifest_id": "manifest",
        "linked_manifest_digest": "1" * 64,
    }


def _report() -> NaiveRunReport:
    return NaiveRunReport.create(
        case=NaiveRunCase.E1,
        validation_mode=NaiveRunValidation.TIMING,
        inputs={
            "spec_digest": "a" * 64,
            "fabric_digest": "b" * 64,
            "hardware_sha256": "c" * 64,
            "simulation_sha256": "d" * 64,
            "mapping_sha256": "e" * 64,
        },
        tools={
            "finalizer_sha256": "f" * 64,
            "npusim_sha256": "0" * 64,
            "linked_manifest_schema": "linked/v1",
            "program_io_schema": "io/v1",
        },
        provenance=_compile_provenance(),
        artifact={
            "artifact_sha256": "2" * 64,
            "artifact_bytes": 1,
            "core_count": 1,
            "record_count": 1,
            "relocation_count": 1,
            "finalizer_report_digest": "3" * 64,
            "program_io_id": "io",
            "program_io_digest": "4" * 64,
        },
        static_metrics={
            "op_counts": {},
            "task_counts": {},
            "action_counts": {},
            "unique_flow_count": 0,
            "opcode_counts": {},
            "fragment_count": 1,
            "record_count": 1,
            "rank_gemm_flops": 0,
            "rank_attention_matmul_flops": 0,
            "analytic_transfer_bytes": 0,
            "scheduled_binding_count": 1,
            "per_core_sram_max_end": {"0": 64},
        },
        runtime={
            "repeat": 2,
            "makespan_cycles": 1,
            "ack_total": 2,
            "done_total": 1,
            "ack_by_core": ((0, 2),),
            "done_by_core": ((0, 1),),
            "drain_residuals": 0,
            "credit_balanced": True,
            "program_io_phases": ("resolved", "applied", "verify"),
            "program_io_initializations": 1,
            "program_io_probes": 1,
            "repeat_signature_stable": True,
            "observed_transfer_bytes": 0,
            "d2d_link_packets": (),
        },
        validation={
            "timing": "pass",
            "address_lifecycle": "pass",
            "transport_control": "pass",
            "compute_functional": "unsupported",
            "reduction_u3f": "unsupported",
            "end_to_end_functional": "unsupported",
            "capability_notes": ("timing only",),
        },
    )


class NaiveRunnerSchemaTest(unittest.TestCase):
    def test_report_is_stable_strict_and_round_trips(self) -> None:
        report = _report()
        report.validate()
        decoded = loads_dataclass(
            NaiveRunReport, canonical_json(report), path="report"
        )
        self.assertEqual(decoded.id, report.id)
        self.assertEqual(canonical_json(decoded), canonical_json(report))
        self.assertEqual(canonical_digest(decoded), canonical_digest(report))

        raw = json.loads(canonical_json(report))
        raw["runtime"]["unexpected"] = 0
        with self.assertRaisesRegex(SchemaError, "exact report field set"):
            loads_dataclass(NaiveRunReport, json.dumps(raw), path="report")

        with self.assertRaisesRegex(SchemaError, "unstable report id"):
            replace(report, id="forged").validate()
        with self.assertRaisesRegex(SchemaError, "positive stable makespan"):
            replace(
                report,
                runtime={**report.runtime, "makespan_cycles": 0},
            ).validate()

        assert isinstance(report.provenance, dict)
        selections = list(report.provenance["policy_selections"])
        self.assertEqual(len(selections), 3)
        with self.assertRaisesRegex(SchemaError, "exact inter/standalone/intra"):
            replace(
                report,
                provenance={
                    **report.provenance,
                    "policy_selections": tuple(reversed(selections)),
                },
            ).validate()
        original_intra = PolicySelection.create(
            kind=RegistryKind.INTRA_DIE,
            name="naive",
            implementation_id="tampered.intra",
            implementation_schema_version="tampered.intra/v1",
            configuration_digest="9" * 64,
            capability_ids=("s1.gemm_collective.naive",),
        )
        with self.assertRaisesRegex(SchemaError, "summary disagrees"):
            replace(
                report,
                provenance={
                    **report.provenance,
                    "policy_selections": (
                        *selections[:2],
                        to_primitive(original_intra),
                    ),
                },
            ).validate()

    def test_request_rejects_wrong_mode_repeat_output_and_missing_tools(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources = tuple(root / name for name in ("spec", "hw", "sim", "map"))
            for source in sources:
                source.write_text("{}", encoding="utf-8")
            npusim = root / "npusim"
            finalizer = root / "finalizer"
            for executable in (npusim, finalizer):
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o700)
            request = NaiveRunRequest(
                case=NaiveRunCase.E1,
                validation=NaiveRunValidation.TIMING,
                spec_path=sources[0],
                hardware_config_path=sources[1],
                simulation_config_path=sources[2],
                mapping_config_path=sources[3],
                output_dir=root / "out",
                npusim_path=npusim,
                finalizer_path=finalizer,
            )
            request.validate()
            with self.assertRaisesRegex(UnsupportedFeatureError, "independent"):
                replace(
                    request,
                    case=NaiveRunCase.E0,
                    validation=NaiveRunValidation.REDUCTION_U3F,
                ).validate()
            with self.assertRaisesRegex(UnsupportedFeatureError, "timing"):
                replace(
                    request,
                    validation=NaiveRunValidation.REDUCTION_U3F,
                ).validate()
            with self.assertRaisesRegex(SchemaError, "exactly two"):
                replace(request, repeat=1).validate()
            request.output_dir.mkdir()
            with self.assertRaisesRegex(SchemaError, "must not already exist"):
                request.validate()
            request.output_dir.rmdir()
            with self.assertRaisesRegex(SchemaError, "PolicyRegistry"):
                run_naive(
                    request,
                    registry=object(),  # type: ignore[arg-type]
                )
            self.assertFalse(request.output_dir.exists())
            self.assertFalse((root / ".out.lock").exists())
            npusim.chmod(0o600)
            with self.assertRaisesRegex(SchemaError, "executable"):
                request.validate()


if __name__ == "__main__":
    unittest.main()
