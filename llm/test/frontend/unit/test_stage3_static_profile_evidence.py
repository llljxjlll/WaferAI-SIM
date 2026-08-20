from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.registry import production_registry
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.stage3_dense_inference_oracle import (
    build_stage3_dense_inference_oracle,
)
from llm.frontend.wafer_frontend.schema.capability import CapabilityStatus
from llm.frontend.wafer_frontend.schema.common import ProfileKey
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec, InferOutput
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramIoMode,
    ProgramIoTargetKind,
)
from llm.frontend.wafer_frontend.schema.n4 import InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.n5 import IntraDieSchedulingContext
from llm.frontend.wafer_frontend.schema.policy import (
    PolicySelection,
    RegistryKind,
)
from llm.frontend.wafer_frontend.schema.s1_naive_evidence import (
    S1NaivePolicyEvidence,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.stage2_dense_forward_evidence import (
    Stage2DenseForwardArtifactEvidence,
    Stage2DenseForwardCompileEvidence,
    Stage2DenseForwardControlEvidence,
    Stage2DenseForwardCoreCount,
    Stage2DenseForwardNamedCount,
    Stage2DenseForwardProbeEvidence,
    Stage2DenseForwardRepeatEvidence,
    Stage2DenseForwardSidecarEvidence,
    Stage2DenseForwardToolEvidence,
)
from llm.frontend.wafer_frontend.schema.stage3_profile import (
    KvPageSpan,
    Stage3ProfileMode,
    Stage3StaticProfile,
    StaticRequestShape,
)
from llm.frontend.wafer_frontend.schema.stage3_static_profile_evidence import (
    STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
    STAGE3_STATIC_PROFILE_MARKER_SCHEMA_VERSION,
    STAGE3_STATIC_PROFILE_RUNTIME_REPORT_SCHEMA_VERSION,
    Stage3StaticAttentionEvidence,
    Stage3StaticProfileRuntimeReport,
    _CASE_GOLDENS,
    _ZERO_D2D,
)

from _fixtures import valid_spec


_DIGEST = "1" * 64
_REQUESTS = {
    Stage3ProfileMode.PREFILL: (("prefill_0", 8, 0, 8),),
    Stage3ProfileMode.MIXED: (
        ("decode_0", 0, 1, 4),
        ("decode_1", 0, 1, 8),
        ("decode_2", 0, 1, 12),
        ("decode_3", 0, 1, 16),
        ("prefill_0", 1, 0, 1),
        ("prefill_1", 3, 0, 3),
    ),
    Stage3ProfileMode.DECODE: tuple(
        (f"decode_{index}", 0, 1, context)
        for index, context in enumerate((4, 8, 12, 16, 20, 24, 28, 32))
    ),
}


def _policy() -> S1NaivePolicyEvidence:
    registry = production_registry()
    selections = (
        registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
        registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE,
            "direct_all_gather",
        ).selection,
        registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection,
    )
    planning = InterDiePlanningContext.create(
        producer_pass="test_stage3_static_profile_evidence",
        fused_policy=selections[0],
        standalone_policy=selections[1],
    )
    scheduling = IntraDieSchedulingContext.create(
        producer_pass="test_stage3_static_profile_evidence",
        policy=selections[2],
    )
    result = S1NaivePolicyEvidence(
        selections,
        planning.id,
        scheduling.id,
    )
    result.validate_against(planning, scheduling)
    return result


def _profile(mode: Stage3ProfileMode) -> Stage3StaticProfile:
    requests = []
    page_start = 0
    for request_ref, prefill, decode, context in _REQUESTS[mode]:
        page_count = (context + 15) // 16
        requests.append(
            StaticRequestShape(
                request_ref=request_ref,
                prefill_tokens=prefill,
                decode_tokens=decode,
                context_tokens=context,
                kv_span=KvPageSpan(page_start, page_count, 16),
            )
        )
        page_start += page_count
    values = tuple(requests)
    return Stage3StaticProfile.create(
        key=ProfileKey(
            prefill_tokens=sum(item.prefill_tokens for item in values),
            decode_tokens=sum(item.decode_tokens for item in values),
            num_seqs=len(values),
            context_sum=sum(item.context_tokens for item in values),
            context_max=max(item.context_tokens for item in values),
            kv_pages=sum(item.kv_span.page_count for item in values),
            expert_load=None,
        ),
        requests=values,
    )


def _oracle(mode: Stage3ProfileMode):
    profile = _profile(mode)
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
        role="both" if mode is Stage3ProfileMode.MIXED else mode.value,
        tp=1,
        sp=False,
    )
    raw["workload"]["infer"].update(  # type: ignore[index]
        output="logits",
        profile={
            "prefill_tokens": profile.key.prefill_tokens,
            "decode_tokens": profile.key.decode_tokens,
            "num_seqs": profile.key.num_seqs,
            "context_sum": profile.key.context_sum,
            "context_max": profile.key.context_max,
            "kv_pages": profile.key.kv_pages,
            "expert_load": None,
        },
    )
    template = build_ir0(
        from_data(ExperimentSpec, raw, path="spec"),
        exact_profiles=(profile,),
    )
    return build_stage3_dense_inference_oracle(
        template, profile, tp_degree=1
    )


def _report(mode: Stage3ProfileMode):
    oracle = _oracle(mode)
    golden = _CASE_GOLDENS[mode]
    artifact = golden["artifact"]
    hbm_probe_count, sram_probe_count = {
        Stage3ProfileMode.PREFILL: (0, 1),
        Stage3ProfileMode.MIXED: (16, 1),
        Stage3ProfileMode.DECODE: (32, 1),
    }[mode]
    probes = tuple(
        sorted(
            (
                *(
                    Stage2DenseForwardProbeEvidence(
                        f"probe.hbm.{index:02d}",
                        ProgramIoTargetKind.HBM,
                        64,
                        _DIGEST,
                        _DIGEST,
                        True,
                        True,
                        True,
                    )
                    for index in range(hbm_probe_count)
                ),
                *(
                    Stage2DenseForwardProbeEvidence(
                        f"probe.sram.{index:02d}",
                        ProgramIoTargetKind.SRAM,
                        64,
                        _DIGEST,
                        _DIGEST,
                        True,
                        True,
                        True,
                    )
                    for index in range(sram_probe_count)
                ),
            ),
            key=lambda item: item.probe_id,
        )
    )
    control = Stage2DenseForwardControlEvidence(
        (Stage2DenseForwardCoreCount(0, 2),),
        (Stage2DenseForwardCoreCount(0, 1),),
        tuple(
            Stage2DenseForwardNamedCount(name, 0)
            for name in ("collective", "global", "p2p", "timing")
        ),
        True,
    )
    repeat_fields = {
        "makespan_cycles": golden["makespan"],
        "marker_digest": golden["marker_digest"],
        "memory_digest": canonical_digest(golden["memory"]),
        "probe_digest": canonical_digest(probes),
        "control_digest": canonical_digest(control),
        "d2d_digest": canonical_digest(_ZERO_D2D),
    }
    report = Stage3StaticProfileRuntimeReport.create(
        baseline_epoch=STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
        profile_mode=mode,
        tp_degree=1,
        infer_output=InferOutput.LOGITS,
        capability_status=CapabilityStatus.E2E_TIMING,
        static_profile_id=oracle.static_profile.id,
        static_profile_digest=canonical_digest(oracle.static_profile),
        oracle_id=oracle.id,
        oracle_digest=canonical_digest(oracle),
        policy=_policy(),
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
        attention=Stage3StaticAttentionEvidence(*golden["attention"]),
        memory=golden["memory"],
        probes=probes,
        control=control,
        d2d=_ZERO_D2D,
        marker_schema_version=STAGE3_STATIC_PROFILE_MARKER_SCHEMA_VERSION,
        repeat_count=2,
        makespan_cycles=golden["makespan"],
        repeats=tuple(
            Stage2DenseForwardRepeatEvidence(index, **repeat_fields)
            for index in range(2)
        ),
        timing_execution=True,
        dense_forward_structure_exact=True,
        static_request_shape_exact=True,
        analytic_work_exact=True,
        program_io_boundary_exact=True,
        traffic_accounting_exact=True,
        compute_functional=False,
        model_functional=False,
    )
    return report, oracle


class Stage3StaticProfileEvidenceTest(unittest.TestCase):
    def test_all_profiles_strict_roundtrip_and_oracle_closure(self) -> None:
        self.assertEqual(
            STAGE3_STATIC_PROFILE_RUNTIME_REPORT_SCHEMA_VERSION,
            "wafer_frontend.stage3_static_profile_runtime_report/v1alpha2",
        )
        for mode in Stage3ProfileMode:
            with self.subTest(mode=mode):
                report, oracle = _report(mode)
                report.validate_against(oracle)
                self.assertEqual(
                    loads_dataclass(
                        Stage3StaticProfileRuntimeReport,
                        canonical_json(report),
                        path="report",
                    ),
                    report,
                )

    def test_strict_serde_and_stable_id_fail_closed(self) -> None:
        report, _ = _report(Stage3ProfileMode.PREFILL)
        raw = json.loads(canonical_json(report))
        raw["unexpected"] = 1
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(
                Stage3StaticProfileRuntimeReport,
                json.dumps(raw),
                path="report",
            )
        raw = json.loads(canonical_json(report))
        raw.pop("policy")
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(
                Stage3StaticProfileRuntimeReport,
                json.dumps(raw),
                path="report",
            )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                report,
                schema_version=(
                    "wafer_frontend.stage3_static_profile_runtime_report/"
                    "v1alpha1"
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(report, id="forged").validate()

    def test_policy_identity_is_exact_and_fail_closed(self) -> None:
        report, _oracle = _report(Stage3ProfileMode.DECODE)
        selection = report.policy.selections[0]

        def forged(**changes: object) -> PolicySelection:
            return PolicySelection.create(
                kind=changes.get("kind", selection.kind),
                name=changes.get("name", selection.name),
                implementation_id=changes.get(
                    "implementation_id", selection.implementation_id
                ),
                implementation_schema_version=changes.get(
                    "implementation_schema_version",
                    selection.implementation_schema_version,
                ),
                configuration_digest=changes.get(
                    "configuration_digest", selection.configuration_digest
                ),
                capability_ids=changes.get(
                    "capability_ids", selection.capability_ids
                ),
            )

        cases = {
            "reordered": report.policy.selections[::-1],
            "kind": (forged(kind=report.policy.selections[2].kind),)
            + report.policy.selections[1:],
            "name": (forged(name="not_naive"),)
            + report.policy.selections[1:],
            "implementation": (
                forged(implementation_id="wafer_frontend.policy.fake"),
            )
            + report.policy.selections[1:],
            "implementation_schema": (
                forged(implementation_schema_version="fake/v1"),
            )
            + report.policy.selections[1:],
            "configuration": (forged(configuration_digest="2" * 64),)
            + report.policy.selections[1:],
            "capability": (forged(capability_ids=("s1.fake",)),)
            + report.policy.selections[1:],
            "interface": (
                replace(selection, interface_version="fake/v1"),
            )
            + report.policy.selections[1:],
        }
        semantic = report._semantic_key()
        for name, policies in cases.items():
            with self.subTest(name=name), self.assertRaises(SchemaError):
                policy = S1NaivePolicyEvidence(
                    policies,
                    report.policy.planning_context_id,
                    report.policy.scheduling_context_id,
                )
                Stage3StaticProfileRuntimeReport.create(
                    **(semantic | {"policy": policy})
                )

    def test_profile_runtime_and_claim_tamper_matrix(self) -> None:
        report, oracle = _report(Stage3ProfileMode.MIXED)
        semantic = report._semantic_key()
        cases = (
            ("artifact", {"artifact": replace(report.artifact, record_count=1)}),
            ("attention", {"attention": replace(
                report.attention, query_key_pairs_per_record=1
            )}),
            ("memory", {"memory": (replace(
                report.memory[0], lsu_hbm_read_bytes=1
            ),)}),
            ("sidecar", {"sidecar": replace(
                report.sidecar, hbm_probe_count=0
            )}),
            ("makespan", {"makespan_cycles": report.makespan_cycles + 1}),
            ("shape_claim", {"static_request_shape_exact": False}),
            ("functional", {"compute_functional": True}),
            ("model", {"model_functional": True}),
        )
        for name, changes in cases:
            with self.subTest(name=name), self.assertRaises(SchemaError):
                Stage3StaticProfileRuntimeReport.create(**(semantic | changes))
        report.validate_against(oracle)
        with self.assertRaisesRegex(SchemaError, "supplied oracle/profile"):
            report.validate_against(_oracle(Stage3ProfileMode.DECODE))


if __name__ == "__main__":
    unittest.main()
