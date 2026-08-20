from __future__ import annotations

import json
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.stage2_dense_forward_oracle import (
    build_stage2_dense_forward_oracle,
)
from llm.frontend.wafer_frontend.schema.capability import CapabilityStatus
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec, InferOutput
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramIoMode,
    ProgramIoTargetKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.stage2_dense_forward_evidence import (
    STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
    STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION,
    STAGE2_DENSE_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION,
    Stage2DenseForwardArtifactEvidence,
    Stage2DenseForwardCompileEvidence,
    Stage2DenseForwardControlEvidence,
    Stage2DenseForwardCoreCount,
    Stage2DenseForwardNamedCount,
    Stage2DenseForwardProbeEvidence,
    Stage2DenseForwardRepeatEvidence,
    Stage2DenseForwardRuntimeReport,
    Stage2DenseForwardSidecarEvidence,
    Stage2DenseForwardToolEvidence,
    _CASE_GOLDENS,
)

from _fixtures import valid_spec


_DIGEST = "1" * 64


def _oracle(tp: int):
    raw = valid_spec()
    raw["model"].update(  # type: ignore[index]
        V=32,
        H=16,
        I=32,
        NH=4,
        KVH=4,
        DH=4,
        rotary_dim=4,
        L=2,
        max_position_embeddings=128,
    )
    raw["parallel"]["instances"][0].update(  # type: ignore[index]
        tp=tp,
        sp=tp > 1,
    )
    raw["workload"]["infer"].update(  # type: ignore[index]
        output="logits",
        profile={
            "prefill_tokens": 8,
            "decode_tokens": 0,
            "num_seqs": 1,
            "context_sum": 8,
            "context_max": 8,
            "kv_pages": 1,
            "expert_load": None,
        },
    )
    template = build_ir0(from_data(ExperimentSpec, raw, path="spec"))
    return build_stage2_dense_forward_oracle(
        template,
        template.profiles[0].key,
        tp_degree=tp,
    )


def _report(tp: int) -> tuple[Stage2DenseForwardRuntimeReport, object]:
    oracle = _oracle(tp)
    golden = _CASE_GOLDENS[tp]
    artifact = golden["artifact"]
    memory = golden["memory"]
    probes = tuple(
        Stage2DenseForwardProbeEvidence(
            f"probe.{index}",
            ProgramIoTargetKind.SRAM,
            64,
            _DIGEST,
            _DIGEST,
            True,
            True,
            True,
        )
        for index in range(tp)
    )
    control = Stage2DenseForwardControlEvidence(
        tuple(
            Stage2DenseForwardCoreCount(item.runtime_core_id, 2)
            for item in memory
        ),
        tuple(
            Stage2DenseForwardCoreCount(item.runtime_core_id, 1)
            for item in memory
        ),
        tuple(Stage2DenseForwardNamedCount(name, 0) for name in (
            "collective",
            "global",
            "p2p",
            "timing",
        )),
        True,
    )
    repeat_fields = {
        "makespan_cycles": golden["makespan"],
        "marker_digest": _DIGEST,
        "memory_digest": canonical_digest(memory),
        "probe_digest": canonical_digest(probes),
        "control_digest": canonical_digest(control),
        "d2d_digest": canonical_digest(golden["d2d"]),
    }
    report = Stage2DenseForwardRuntimeReport.create(
        baseline_epoch=STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
        tp_degree=tp,
        infer_output=InferOutput.LOGITS,
        capability_status=CapabilityStatus.E2E_TIMING,
        oracle_id=oracle.id,
        oracle_digest=canonical_digest(oracle),
        compile=Stage2DenseForwardCompileEvidence(
            oracle.source_template_id,
            _DIGEST,
            "ir1",
            _DIGEST,
            "global",
            _DIGEST,
            "lowered",
            _DIGEST,
        ),
        tools=Stage2DenseForwardToolEvidence(_DIGEST, _DIGEST, _DIGEST),
        hardware_digest=_DIGEST,
        simulation_digest=_DIGEST,
        mapping_digest=_DIGEST,
        artifact=Stage2DenseForwardArtifactEvidence(
            "linked",
            _DIGEST,
            artifact[6],
            artifact[0],
            artifact[1],
            artifact[2],
            artifact[3],
            artifact[4],
            artifact[5],
            golden["opcodes"],
        ),
        sidecar=Stage2DenseForwardSidecarEvidence(
            "program_io",
            _DIGEST,
            ProgramIoMode.TIMING,
            *golden["sidecar"],
        ),
        memory=memory,
        probes=probes,
        control=control,
        d2d=golden["d2d"],
        marker_schema_version=STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION,
        repeat_count=2,
        makespan_cycles=golden["makespan"],
        repeats=tuple(
            Stage2DenseForwardRepeatEvidence(index, **repeat_fields)
            for index in range(2)
        ),
        timing_execution=True,
        dense_forward_structure_exact=True,
        analytic_work_exact=True,
        program_io_boundary_exact=True,
        traffic_accounting_exact=True,
        compute_functional=False,
        model_functional=False,
    )
    return report, oracle


class Stage2DenseForwardEvidenceTest(unittest.TestCase):
    def test_tp1_tp2_tp4_strict_roundtrip_and_oracle_closure(self) -> None:
        self.assertEqual(
            STAGE2_DENSE_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION,
            "wafer_frontend.stage2_dense_forward_runtime_report/v1alpha1",
        )
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                report, oracle = _report(tp)
                report.validate_against(oracle)
                self.assertEqual(
                    loads_dataclass(
                        Stage2DenseForwardRuntimeReport,
                        canonical_json(report),
                        path="report",
                    ),
                    report,
                )

    def test_strict_serde_and_stable_id_fail_closed(self) -> None:
        report, _ = _report(1)
        raw = json.loads(canonical_json(report))
        raw["unexpected"] = 1
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(
                Stage2DenseForwardRuntimeReport,
                json.dumps(raw),
                path="report",
            )
        del raw["unexpected"]
        del raw["tools"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(
                Stage2DenseForwardRuntimeReport,
                json.dumps(raw),
                path="report",
            )
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(report, id="forged").validate()

    def test_runtime_and_claim_tamper_matrix(self) -> None:
        report, oracle = _report(1)
        semantic = report._semantic_key()
        cases = (
            ("artifact_bytes", {"artifact": replace(report.artifact, artifact_size_bytes=1)}),
            (
                "artifact_sha",
                {
                    "artifact": replace(
                        report.artifact, program_artifact_sha256="2" * 64
                    )
                },
            ),
            ("record_count", {"artifact": replace(report.artifact, record_count=1)}),
            ("relocation_count", {"artifact": replace(report.artifact, relocation_count=1)}),
            ("makespan", {"makespan_cycles": report.makespan_cycles + 1}),
            ("memory", {"memory": (replace(report.memory[0], lsu_hbm_read_bytes=1),)}),
            ("d2d", {"d2d": replace(report.d2d, logical_bytes=16)}),
            ("timing_claim", {"timing_execution": False}),
            ("structure_claim", {"dense_forward_structure_exact": False}),
            ("analytic_claim", {"analytic_work_exact": False}),
            ("program_io_claim", {"program_io_boundary_exact": False}),
            ("traffic_claim", {"traffic_accounting_exact": False}),
            ("functional_claim", {"compute_functional": True}),
            ("model_claim", {"model_functional": True}),
        )
        for name, changes in cases:
            with self.subTest(name=name), self.assertRaises(SchemaError):
                Stage2DenseForwardRuntimeReport.create(**(semantic | changes))
        report.validate_against(oracle)
        with self.assertRaisesRegex(SchemaError, "supplied oracle"):
            report.validate_against(_oracle(2))


if __name__ == "__main__":
    unittest.main()
