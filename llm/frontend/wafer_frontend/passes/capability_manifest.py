"""Pure construction of capability truth from a case matrix and evidence."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.capability import (
    CAPABILITY_MANIFEST_SCHEMA_VERSION,
    AcceptanceDefinition,
    AcceptanceClaim,
    CapabilityDefinition,
    CapabilityClaim,
    CapabilityManifest,
    CapabilityStage,
    CapabilityStatus,
    CaseDefinition,
    CaseEvidence,
    CaseMatrix,
    capability_status_rank,
)
from ..schema.common import stable_artifact_id, validate_nonempty
from ..schema.serde import canonical_digest
from ..schema.stage1a_evidence import (
    STAGE1A_BASELINE_EPOCH,
    Stage1aCase,
    Stage1aOracle,
    Stage1aRuntimeReport,
)
from ..schema.stage2_dense_forward_evidence import (
    STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
    Stage2DenseForwardRuntimeReport,
)
from ..schema.stage2_dense_forward_oracle import Stage2DenseForwardOracle
from ..schema.stage3_dense_inference_oracle import Stage3DenseInferenceOracle
from ..schema.stage3_profile import Stage3ProfileMode
from ..schema.stage3_static_profile_evidence import (
    STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
    Stage3StaticProfileRuntimeReport,
)
from ..schema.stage4_pd_case_matrix import Stage4PdCaseMatrix
from ..schema.stage4_pd_evidence import STAGE4_PD_BASELINE_EPOCH



_DEVELOPMENT_PLAN_REF = "notes/frontend/S1-S3后续开发方案.md"
_STAGE1A_CAPABILITY_BY_CASE = {
    Stage1aCase.P1: "s1.foundation.parameter_dma_compute",
    Stage1aCase.K1: "s1.foundation.kv_cross_action",
    Stage1aCase.PD1: "s1.foundation.synthetic_pd_state",
}
_STAGE1A_CASE_ID_BY_CASE = {
    case: f"case.s1.foundation.{capability.rsplit('.', 1)[-1]}"
    for case, capability in _STAGE1A_CAPABILITY_BY_CASE.items()
}
_STATE_FAIL_CLOSED_CAPABILITY = "s1.foundation.state_fail_closed"
_STATE_FAIL_CLOSED_CASE = "case.s1.foundation.state_fail_closed"
_DENSE_FORWARD_CAPABILITY = "s1.foundation.dense_forward_closure"
_DENSE_FORWARD_FAIL_CLOSED_CAPABILITY = (
    "s1.foundation.dense_forward_fail_closed"
)
_DENSE_FORWARD_CASE_BY_TP = {
    tp_degree: f"case.s1.foundation.dense_forward_tp{tp_degree}"
    for tp_degree in (1, 2, 4)
}
_DENSE_FORWARD_FAIL_CLOSED_CASE = (
    "case.s1.foundation.dense_forward_fail_closed"
)
_DENSE_FORWARD_INPUT_REFS = (
    _DEVELOPMENT_PLAN_REF,
    "llm/test/frontend/integration/run_stage2_dense_forward.py",
    "llm/test/frontend/integration/stage2_dense_forward_cases.py",
    "notes/frontend/S2开发报告.md",
)
_DENSE_FORWARD_FAIL_CLOSED_INPUT_REFS = (
    _DEVELOPMENT_PLAN_REF,
    "llm/test/frontend/unit/test_stage2_artifact_schema.py",
    "llm/test/frontend/unit/test_stage2_coarse_lowering.py",
    "llm/test/frontend/unit/test_stage2_dense_forward_evidence.py",
    "llm/test/frontend/unit/test_stage2_dense_forward_graph.py",
    "notes/frontend/S2开发报告.md",
)
_STATIC_PROFILE_FAIL_CLOSED_CAPABILITY = (
    "s1.foundation.static_profile_fail_closed"
)
_STATIC_PROFILE_FAIL_CLOSED_CASE = (
    "case.s1.foundation.static_profile_fail_closed"
)
_STATIC_PROFILE_CASE_BY_MODE = {
    Stage3ProfileMode.PREFILL: "case.s1.f_p1",
    Stage3ProfileMode.DECODE: "case.s1.f_d1",
    Stage3ProfileMode.MIXED: "case.s1.f_m1",
}
_STATIC_PROFILE_FAIL_CLOSED_INPUT_REFS = (
    _DEVELOPMENT_PLAN_REF,
    "llm/test/frontend/integration/run_stage3_static_profiles.py",
    "llm/test/frontend/integration/stage3_decode_cases.py",
    "llm/test/frontend/unit/test_stage3_profile_selection.py",
    "llm/test/frontend/unit/test_stage3_static_profile_evidence.py",
)
_STAGE4_PROMOTED_CASES = ("case.s1.kv_handoff", "case.s1.pd")


def _definition(
    key: str,
    stage: CapabilityStage,
    case_ids: tuple[str, ...],
    *,
    numerator: int = 0,
    denominator: int = 1,
) -> CapabilityDefinition:
    return CapabilityDefinition(
        key=key,
        stage=stage,
        required_case_ids=tuple(sorted(case_ids)),
        score_numerator=numerator,
        score_denominator=denominator,
        notes=("evidence is derived; declarations never imply support",),
    )


def _case(
    case_id: str,
    stage: CapabilityStage,
    capability_ids: tuple[str, ...],
    *,
    maximum_status: CapabilityStatus = CapabilityStatus.E2E_TIMING,
    input_refs: tuple[str, ...] = (_DEVELOPMENT_PLAN_REF,),
) -> CaseDefinition:
    return CaseDefinition(
        id=case_id,
        stage=stage,
        maximum_status=maximum_status,
        capability_ids=tuple(sorted(capability_ids)),
        input_refs=tuple(sorted(input_refs)),
        description=f"frozen evidence case {case_id}",
    )


def build_s1_s3_case_matrix() -> CaseMatrix:
    """Build the checked S1--S3 truth matrix from the development plan."""

    e1 = "case.s1.baseline.e1_tp2"
    e2 = "case.s1.baseline.e2_tp4"
    single_die = "case.s1.baseline.single_die_unit"
    capabilities = [
        _definition(
            "s1.gemm_collective.naive",
            CapabilityStage.S1,
            (e1, e2),
            numerator=1,
            denominator=2,
        ),
        _definition(
            "s1.gemm_collective.optimized",
            CapabilityStage.S1,
            ("case.s1.optimized_ablation",),
            numerator=1,
            denominator=2,
        ),
        _definition(
            "s1.kv_handoff",
            CapabilityStage.S1,
            ("case.s1.kv_handoff",),
            numerator=1,
        ),
        _definition(
            "s1.mesh",
            CapabilityStage.S1,
            (e2,),
            numerator=1,
        ),
        _definition(
            "s1.pd",
            CapabilityStage.S1,
            ("case.s1.pd",),
            numerator=1,
        ),
        _definition(
            "s1.single_die",
            CapabilityStage.S1,
            (single_die,),
            numerator=1,
        ),
        _definition(
            "s1.tp_sp",
            CapabilityStage.S1,
            (e1, e2),
            numerator=1,
        ),
        _definition(
            "s1.validation_ladder",
            CapabilityStage.S1,
            ("case.s1.validation_ladder",),
        ),
    ]
    cases = [
        _case(
            e1,
            CapabilityStage.S1,
            ("s1.gemm_collective.naive", "s1.tp_sp"),
            input_refs=(
                "notes/frontend/examples/hardware_2x1.json",
                "notes/frontend/examples/naive_dense_tp2.yaml",
            ),
        ),
        _case(
            e2,
            CapabilityStage.S1,
            ("s1.gemm_collective.naive", "s1.mesh", "s1.tp_sp"),
            input_refs=(
                "notes/frontend/examples/hardware_2x2.json",
                "notes/frontend/examples/naive_dense_tp4.yaml",
            ),
        ),
        _case(
            single_die,
            CapabilityStage.S1,
            ("s1.single_die",),
            maximum_status=CapabilityStatus.UNIT_ONLY,
            input_refs=("llm/test/frontend/unit/test_dense_ir0_validator.py",),
        ),
        _case(
            "case.s1.kv_handoff",
            CapabilityStage.S1,
            ("s1.kv_handoff",),
        ),
        _case(
            "case.s1.optimized_ablation",
            CapabilityStage.S1,
            ("s1.gemm_collective.optimized",),
        ),
        _case("case.s1.pd", CapabilityStage.S1, ("s1.pd",)),
        _case(
            "case.s1.validation_ladder",
            CapabilityStage.S1,
            ("s1.validation_ladder",),
        ),
    ]

    for case_name in (
        "f_d1",
        "f_m1",
        "f_p1",
        "f_p2",
        "f_pdf",
        "f_pdr",
        "f_pds",
        "f_tf",
    ):
        capability_id = f"s1.naive_case.{case_name}"
        case_id = f"case.s1.{case_name}"
        capabilities.append(
            _definition(capability_id, CapabilityStage.S1, (case_id,))
        )
        cases.append(_case(case_id, CapabilityStage.S1, (capability_id,)))
    for case_name in ("t0", "t1", "t2", "t3", "t4", "t5"):
        capability_id = f"s2.naive_case.{case_name}"
        case_id = f"case.s2.{case_name}"
        capabilities.append(
            _definition(capability_id, CapabilityStage.S2, (case_id,))
        )
        cases.append(_case(case_id, CapabilityStage.S2, (capability_id,)))
    for case_name in ("m0", "m1", "m2", "m3", "m4"):
        capability_id = f"s3.naive_case.{case_name}"
        case_id = f"case.s3.{case_name}"
        capabilities.append(
            _definition(capability_id, CapabilityStage.S3, (case_id,))
        )
        cases.append(_case(case_id, CapabilityStage.S3, (capability_id,)))

    acceptance = (
        AcceptanceDefinition(
            key="s1.acceptance.ablation",
            stage=CapabilityStage.S1,
            required_case_ids=(),
            blocking_capability_ids=("s1.gemm_collective.optimized",),
            notes=("requires a real inter by intra optimized ablation",),
        ),
        AcceptanceDefinition(
            key="s1.acceptance.e2e",
            stage=CapabilityStage.S1,
            required_case_ids=(),
            blocking_capability_ids=("s1.gemm_collective.naive",),
            notes=("requires the checked E1 and E2 timing cases",),
        ),
        AcceptanceDefinition(
            key="s1.acceptance.validation_ladder",
            stage=CapabilityStage.S1,
            required_case_ids=(),
            blocking_capability_ids=("s1.validation_ladder",),
            notes=("requires the complete analytic validation ladder",),
        ),
    )
    return CaseMatrix.create(
        capabilities=tuple(sorted(capabilities, key=lambda item: item.key)),
        acceptance=acceptance,
        cases=tuple(sorted(cases, key=lambda item: item.id)),
    )


def build_stage1a_case_matrix(stage0_matrix: CaseMatrix) -> CaseMatrix:
    """Extend reviewed Stage0 truth with zero-score foundation evidence."""

    if type(stage0_matrix) is not CaseMatrix:
        raise SchemaError("must be a CaseMatrix", path="stage0_matrix")
    stage0_matrix.validate("stage0_matrix")
    capabilities = tuple(
        _definition(
            _STAGE1A_CAPABILITY_BY_CASE[case],
            CapabilityStage.S1,
            (_STAGE1A_CASE_ID_BY_CASE[case],),
        )
        for case in Stage1aCase
    ) + (
        _definition(
            _STATE_FAIL_CLOSED_CAPABILITY,
            CapabilityStage.S1,
            (_STATE_FAIL_CLOSED_CASE,),
        ),
    )
    cases = tuple(
        _case(
            _STAGE1A_CASE_ID_BY_CASE[case],
            CapabilityStage.S1,
            (_STAGE1A_CAPABILITY_BY_CASE[case],),
            input_refs=(
                _DEVELOPMENT_PLAN_REF,
                "llm/test/frontend/integration/stage1a_state_cases.py",
            ),
        )
        for case in Stage1aCase
    ) + (
        _case(
            _STATE_FAIL_CLOSED_CASE,
            CapabilityStage.S1,
            (_STATE_FAIL_CLOSED_CAPABILITY,),
            maximum_status=CapabilityStatus.UNIT_ONLY,
            input_refs=(
                _DEVELOPMENT_PLAN_REF,
                "llm/test/frontend/integration/run_pd1_finalizer.py",
                "llm/test/frontend/integration/run_program_io_e1_t.py",
                "llm/test/frontend/integration/run_stage1a_negative_evidence.py",
                "llm/test/frontend/unit/test_persistent_state_schema.py",
                "llm/test/frontend/unit/test_program_io_hbm_validate.py",
                "llm/test/frontend/unit/test_stage1a_evidence.py",
                "llm/test/frontend/unit/test_state_dma_lowering.py",
                "llm/test/frontend/unit/test_state_transfer_lowering.py",
            ),
        ),
    )
    old_capability_ids = {
        definition.key for definition in stage0_matrix.capabilities
    }
    old_case_ids = {case.id for case in stage0_matrix.cases}
    if old_capability_ids.intersection(
        definition.key for definition in capabilities
    ) or old_case_ids.intersection(case.id for case in cases):
        raise SchemaError(
            "Stage1a foundation ids collide with Stage0",
            path="stage0_matrix",
        )
    return CaseMatrix.create(
        capabilities=tuple(
            sorted(
                (*stage0_matrix.capabilities, *capabilities),
                key=lambda item: item.key,
            )
        ),
        acceptance=stage0_matrix.acceptance,
        cases=tuple(
            sorted((*stage0_matrix.cases, *cases), key=lambda item: item.id)
        ),
    )


def build_stage2_case_matrix(stage1a_matrix: CaseMatrix) -> CaseMatrix:
    """Append zero-score dense-forward timing and fail-closed cases."""

    if type(stage1a_matrix) is not CaseMatrix:
        raise SchemaError("must be a CaseMatrix", path="stage1a_matrix")
    stage1a_matrix.validate("stage1a_matrix")
    capabilities = (
        _definition(
            _DENSE_FORWARD_CAPABILITY,
            CapabilityStage.S1,
            tuple(_DENSE_FORWARD_CASE_BY_TP.values()),
        ),
        _definition(
            _DENSE_FORWARD_FAIL_CLOSED_CAPABILITY,
            CapabilityStage.S1,
            (_DENSE_FORWARD_FAIL_CLOSED_CASE,),
        ),
    )
    cases = tuple(
        _case(
            _DENSE_FORWARD_CASE_BY_TP[tp_degree],
            CapabilityStage.S1,
            (_DENSE_FORWARD_CAPABILITY,),
            input_refs=_DENSE_FORWARD_INPUT_REFS,
        )
        for tp_degree in (1, 2, 4)
    ) + (
        _case(
            _DENSE_FORWARD_FAIL_CLOSED_CASE,
            CapabilityStage.S1,
            (_DENSE_FORWARD_FAIL_CLOSED_CAPABILITY,),
            maximum_status=CapabilityStatus.UNIT_ONLY,
            input_refs=_DENSE_FORWARD_FAIL_CLOSED_INPUT_REFS,
        ),
    )
    old_capability_ids = {
        definition.key for definition in stage1a_matrix.capabilities
    }
    old_case_ids = {case.id for case in stage1a_matrix.cases}
    if old_capability_ids.intersection(
        definition.key for definition in capabilities
    ) or old_case_ids.intersection(case.id for case in cases):
        raise SchemaError(
            "Stage2 dense-forward ids collide with Stage1a",
            path="stage1a_matrix",
        )
    return CaseMatrix.create(
        capabilities=tuple(
            sorted(
                (*stage1a_matrix.capabilities, *capabilities),
                key=lambda item: item.key,
            )
        ),
        acceptance=stage1a_matrix.acceptance,
        cases=tuple(
            sorted((*stage1a_matrix.cases, *cases), key=lambda item: item.id)
        ),
    )


def build_stage3_case_matrix(stage2_matrix: CaseMatrix) -> CaseMatrix:
    """Append a zero-score exact-profile fail-closed capability."""

    if type(stage2_matrix) is not CaseMatrix:
        raise SchemaError("must be a CaseMatrix", path="stage2_matrix")
    stage2_matrix.validate("stage2_matrix")
    capability = _definition(
        _STATIC_PROFILE_FAIL_CLOSED_CAPABILITY,
        CapabilityStage.S1,
        (_STATIC_PROFILE_FAIL_CLOSED_CASE,),
    )
    case = _case(
        _STATIC_PROFILE_FAIL_CLOSED_CASE,
        CapabilityStage.S1,
        (_STATIC_PROFILE_FAIL_CLOSED_CAPABILITY,),
        maximum_status=CapabilityStatus.UNIT_ONLY,
        input_refs=_STATIC_PROFILE_FAIL_CLOSED_INPUT_REFS,
    )
    if (
        capability.key in {item.key for item in stage2_matrix.capabilities}
        or case.id in {item.id for item in stage2_matrix.cases}
    ):
        raise SchemaError(
            "Stage3 static-profile ids collide with Stage2",
            path="stage2_matrix",
        )
    return CaseMatrix.create(
        capabilities=tuple(
            sorted(
                (*stage2_matrix.capabilities, capability),
                key=lambda item: item.key,
            )
        ),
        acceptance=stage2_matrix.acceptance,
        cases=tuple(
            sorted((*stage2_matrix.cases, case), key=lambda item: item.id)
        ),
    )


def _evidence_satisfies(
    case_maximum: CapabilityStatus, evidence: CaseEvidence | None
) -> bool:
    return evidence is not None and capability_status_rank(
        evidence.status
    ) >= capability_status_rank(case_maximum)



def build_stage0_capability_manifest(
    *,
    e1_artifact_digest: str,
    e1_report_digest: str,
    e2_artifact_digest: str,
    e2_report_digest: str,
    single_die_evidence_digest: str,
    baseline_epoch: str = "stage0-policy-provenance-v1",
) -> tuple[CaseMatrix, CapabilityManifest]:
    """Build the initial reviewed truth without inferring unsupported cases."""

    def timing_evidence(
        case_id: str,
        artifact_digest: str,
        report_digest: str,
    ) -> CaseEvidence:
        return CaseEvidence(
            case_id=case_id,
            status=CapabilityStatus.E2E_TIMING,
            evidence_digest=canonical_digest(
                {
                    "case_id": case_id,
                    "artifact_digest": artifact_digest,
                    "report_digest": report_digest,
                }
            ),
            artifact_digest=artifact_digest,
            report_digest=report_digest,
        )

    matrix = build_s1_s3_case_matrix()
    evidence = (
        timing_evidence(
            "case.s1.baseline.e1_tp2",
            e1_artifact_digest,
            e1_report_digest,
        ),
        timing_evidence(
            "case.s1.baseline.e2_tp4",
            e2_artifact_digest,
            e2_report_digest,
        ),
        CaseEvidence(
            case_id="case.s1.baseline.single_die_unit",
            status=CapabilityStatus.UNIT_ONLY,
            evidence_digest=single_die_evidence_digest,
            artifact_digest=None,
            report_digest=None,
        ),
    )
    manifest = build_capability_manifest(
        matrix,
        evidence,
        baseline_epoch=baseline_epoch,
    )
    manifest.validate_against(matrix)
    return matrix, manifest


def _stage1a_runtime_case_evidence(
    *,
    expected_case: Stage1aCase,
    oracle: Stage1aOracle | None,
    report: Stage1aRuntimeReport | None,
) -> CaseEvidence | None:
    path = expected_case.value.lower()
    if (oracle is None) != (report is None):
        raise SchemaError(
            "oracle and report must be supplied together",
            path=f"{path}_evidence",
        )
    if oracle is None:
        return None
    if type(oracle) is not Stage1aOracle:
        raise SchemaError("must be a Stage1aOracle", path=f"{path}_oracle")
    if type(report) is not Stage1aRuntimeReport:
        raise SchemaError(
            "must be a Stage1aRuntimeReport", path=f"{path}_report"
        )
    if oracle.case is not expected_case or report.case is not expected_case:
        raise SchemaError(
            f"must contain {expected_case.value} evidence",
            path=f"{path}_evidence.case",
        )
    report.validate_against(oracle, f"{path}_report")
    report_digest = canonical_digest(report)
    oracle_digest = canonical_digest(oracle)
    case_id = _STAGE1A_CASE_ID_BY_CASE[expected_case]
    evidence = CaseEvidence(
        case_id=case_id,
        status=report.capability_status,
        evidence_digest=canonical_digest(
            {
                "case_id": case_id,
                "oracle_id": oracle.id,
                "oracle_digest": oracle_digest,
                "report_id": report.id,
                "report_digest": report_digest,
                "program_artifact_sha256": (
                    report.artifact.program_artifact_sha256
                ),
            }
        ),
        artifact_digest=report.artifact.program_artifact_sha256,
        report_digest=report_digest,
    )
    evidence.validate(f"{path}_evidence")
    return evidence


def _stage2_dense_forward_case_evidence(
    *,
    expected_tp_degree: int,
    oracle: Stage2DenseForwardOracle | None,
    report: Stage2DenseForwardRuntimeReport | None,
) -> CaseEvidence | None:
    path = f"tp{expected_tp_degree}"
    if (oracle is None) != (report is None):
        raise SchemaError(
            "oracle and report must be supplied together",
            path=f"{path}_evidence",
        )
    if oracle is None:
        return None
    if type(oracle) is not Stage2DenseForwardOracle:
        raise SchemaError(
            "must be a Stage2DenseForwardOracle", path=f"{path}_oracle"
        )
    if type(report) is not Stage2DenseForwardRuntimeReport:
        raise SchemaError(
            "must be a Stage2DenseForwardRuntimeReport",
            path=f"{path}_report",
        )
    if (
        oracle.tp_degree != expected_tp_degree
        or report.tp_degree != expected_tp_degree
    ):
        raise SchemaError(
            f"must contain TP{expected_tp_degree} evidence",
            path=f"{path}_evidence.tp_degree",
        )
    report.validate_against(oracle, f"{path}_report")
    oracle_digest = canonical_digest(oracle)
    report_digest = canonical_digest(report)
    case_id = _DENSE_FORWARD_CASE_BY_TP[expected_tp_degree]
    evidence = CaseEvidence(
        case_id=case_id,
        status=report.capability_status,
        evidence_digest=canonical_digest(
            {
                "case_id": case_id,
                "tp_degree": expected_tp_degree,
                "oracle_id": oracle.id,
                "oracle_digest": oracle_digest,
                "report_id": report.id,
                "report_digest": report_digest,
                "program_artifact_sha256": (
                    report.artifact.program_artifact_sha256
                ),
            }
        ),
        artifact_digest=report.artifact.program_artifact_sha256,
        report_digest=report_digest,
    )
    evidence.validate(f"{path}_evidence")
    return evidence


def _stage3_static_profile_case_evidence(
    *,
    expected_mode: Stage3ProfileMode,
    oracle: Stage3DenseInferenceOracle | None,
    report: Stage3StaticProfileRuntimeReport | None,
) -> CaseEvidence | None:
    path = f"stage3_{expected_mode.value}"
    if (oracle is None) != (report is None):
        raise SchemaError(
            "oracle and report must be supplied together",
            path=f"{path}_evidence",
        )
    if oracle is None:
        return None
    if type(oracle) is not Stage3DenseInferenceOracle:
        raise SchemaError(
            "must be a Stage3DenseInferenceOracle", path=f"{path}_oracle"
        )
    if type(report) is not Stage3StaticProfileRuntimeReport:
        raise SchemaError(
            "must be a Stage3StaticProfileRuntimeReport",
            path=f"{path}_report",
        )
    if (
        oracle.static_profile.mode is not expected_mode
        or report.profile_mode is not expected_mode
    ):
        raise SchemaError(
            f"must contain {expected_mode.value} evidence",
            path=f"{path}_evidence.profile_mode",
        )
    report.validate_against(oracle, f"{path}_report")
    report_digest = canonical_digest(report)
    case_id = _STATIC_PROFILE_CASE_BY_MODE[expected_mode]
    evidence = CaseEvidence(
        case_id=case_id,
        status=report.capability_status,
        evidence_digest=canonical_digest(
            {
                "case_id": case_id,
                "profile_mode": expected_mode,
                "static_profile_id": oracle.static_profile.id,
                "static_profile_digest": canonical_digest(
                    oracle.static_profile
                ),
                "oracle_id": oracle.id,
                "oracle_digest": canonical_digest(oracle),
                "report_id": report.id,
                "report_digest": report_digest,
                "program_artifact_sha256": (
                    report.artifact.program_artifact_sha256
                ),
            }
        ),
        artifact_digest=report.artifact.program_artifact_sha256,
        report_digest=report_digest,
    )
    evidence.validate(f"{path}_evidence")
    return evidence


def build_stage1a_capability_manifest(
    stage0_matrix: CaseMatrix,
    stage0_manifest: CapabilityManifest,
    *,
    p1_oracle: Stage1aOracle | None = None,
    p1_report: Stage1aRuntimeReport | None = None,
    k1_oracle: Stage1aOracle | None = None,
    k1_report: Stage1aRuntimeReport | None = None,
    pd1_oracle: Stage1aOracle | None = None,
    pd1_report: Stage1aRuntimeReport | None = None,
    state_fail_closed_evidence_digest: str | None = None,
) -> tuple[CaseMatrix, CapabilityManifest]:
    """Derive zero-score Stage1a foundation claims from strict evidence."""

    if type(stage0_matrix) is not CaseMatrix:
        raise SchemaError("must be a CaseMatrix", path="stage0_matrix")
    if type(stage0_manifest) is not CapabilityManifest:
        raise SchemaError(
            "must be a CapabilityManifest", path="stage0_manifest"
        )
    stage0_manifest.validate_against(stage0_matrix, "stage0_manifest")
    matrix = build_stage1a_case_matrix(stage0_matrix)
    additions = tuple(
        evidence
        for evidence in (
            _stage1a_runtime_case_evidence(
                expected_case=Stage1aCase.P1,
                oracle=p1_oracle,
                report=p1_report,
            ),
            _stage1a_runtime_case_evidence(
                expected_case=Stage1aCase.K1,
                oracle=k1_oracle,
                report=k1_report,
            ),
            _stage1a_runtime_case_evidence(
                expected_case=Stage1aCase.PD1,
                oracle=pd1_oracle,
                report=pd1_report,
            ),
        )
        if evidence is not None
    )
    if state_fail_closed_evidence_digest is not None:
        negative = CaseEvidence(
            case_id=_STATE_FAIL_CLOSED_CASE,
            status=CapabilityStatus.UNIT_ONLY,
            evidence_digest=state_fail_closed_evidence_digest,
            artifact_digest=None,
            report_digest=None,
        )
        negative.validate("state_fail_closed_evidence")
        additions = (*additions, negative)
    evidence = tuple(
        sorted(
            (*stage0_manifest.evidence, *additions),
            key=lambda item: item.case_id,
        )
    )
    manifest = build_capability_manifest(
        matrix,
        evidence,
        baseline_epoch=STAGE1A_BASELINE_EPOCH,
    )
    manifest.validate_against(matrix)
    return matrix, manifest


def build_stage2_capability_manifest(
    stage1a_matrix: CaseMatrix,
    stage1a_manifest: CapabilityManifest,
    *,
    tp1_oracle: Stage2DenseForwardOracle | None = None,
    tp1_report: Stage2DenseForwardRuntimeReport | None = None,
    tp2_oracle: Stage2DenseForwardOracle | None = None,
    tp2_report: Stage2DenseForwardRuntimeReport | None = None,
    tp4_oracle: Stage2DenseForwardOracle | None = None,
    tp4_report: Stage2DenseForwardRuntimeReport | None = None,
    dense_forward_fail_closed_evidence_digest: str | None = None,
) -> tuple[CaseMatrix, CapabilityManifest]:
    """Derive zero-score Stage2 dense-forward claims from exact evidence."""

    if type(stage1a_matrix) is not CaseMatrix:
        raise SchemaError("must be a CaseMatrix", path="stage1a_matrix")
    if type(stage1a_manifest) is not CapabilityManifest:
        raise SchemaError(
            "must be a CapabilityManifest", path="stage1a_manifest"
        )
    stage1a_manifest.validate_against(stage1a_matrix, "stage1a_manifest")
    matrix = build_stage2_case_matrix(stage1a_matrix)
    additions = tuple(
        evidence
        for evidence in (
            _stage2_dense_forward_case_evidence(
                expected_tp_degree=1,
                oracle=tp1_oracle,
                report=tp1_report,
            ),
            _stage2_dense_forward_case_evidence(
                expected_tp_degree=2,
                oracle=tp2_oracle,
                report=tp2_report,
            ),
            _stage2_dense_forward_case_evidence(
                expected_tp_degree=4,
                oracle=tp4_oracle,
                report=tp4_report,
            ),
        )
        if evidence is not None
    )
    if dense_forward_fail_closed_evidence_digest is not None:
        negative = CaseEvidence(
            case_id=_DENSE_FORWARD_FAIL_CLOSED_CASE,
            status=CapabilityStatus.UNIT_ONLY,
            evidence_digest=dense_forward_fail_closed_evidence_digest,
            artifact_digest=None,
            report_digest=None,
        )
        negative.validate("dense_forward_fail_closed_evidence")
        additions = (*additions, negative)
    evidence = tuple(
        sorted(
            (*stage1a_manifest.evidence, *additions),
            key=lambda item: item.case_id,
        )
    )
    manifest = build_capability_manifest(
        matrix,
        evidence,
        baseline_epoch=STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
    )
    manifest.validate_against(matrix)
    return matrix, manifest


def build_stage3_capability_manifest(
    stage2_matrix: CaseMatrix,
    stage2_manifest: CapabilityManifest,
    *,
    prefill_oracle: Stage3DenseInferenceOracle | None = None,
    prefill_report: Stage3StaticProfileRuntimeReport | None = None,
    decode_oracle: Stage3DenseInferenceOracle | None = None,
    decode_report: Stage3StaticProfileRuntimeReport | None = None,
    mixed_oracle: Stage3DenseInferenceOracle | None = None,
    mixed_report: Stage3StaticProfileRuntimeReport | None = None,
    static_profile_fail_closed_evidence_digest: str | None = None,
) -> tuple[CaseMatrix, CapabilityManifest]:
    """Derive exact F-P1/F-D1/F-M1 claims from Stage3 evidence."""

    if type(stage2_matrix) is not CaseMatrix:
        raise SchemaError("must be a CaseMatrix", path="stage2_matrix")
    if type(stage2_manifest) is not CapabilityManifest:
        raise SchemaError(
            "must be a CapabilityManifest", path="stage2_manifest"
        )
    stage2_manifest.validate_against(stage2_matrix, "stage2_manifest")
    matrix = build_stage3_case_matrix(stage2_matrix)
    additions = tuple(
        evidence
        for evidence in (
            _stage3_static_profile_case_evidence(
                expected_mode=Stage3ProfileMode.PREFILL,
                oracle=prefill_oracle,
                report=prefill_report,
            ),
            _stage3_static_profile_case_evidence(
                expected_mode=Stage3ProfileMode.DECODE,
                oracle=decode_oracle,
                report=decode_report,
            ),
            _stage3_static_profile_case_evidence(
                expected_mode=Stage3ProfileMode.MIXED,
                oracle=mixed_oracle,
                report=mixed_report,
            ),
        )
        if evidence is not None
    )
    if static_profile_fail_closed_evidence_digest is not None:
        negative = CaseEvidence(
            case_id=_STATIC_PROFILE_FAIL_CLOSED_CASE,
            status=CapabilityStatus.UNIT_ONLY,
            evidence_digest=static_profile_fail_closed_evidence_digest,
            artifact_digest=None,
            report_digest=None,
        )
        negative.validate("static_profile_fail_closed_evidence")
        additions = (*additions, negative)
    evidence = tuple(
        sorted(
            (*stage2_manifest.evidence, *additions),
            key=lambda item: item.case_id,
        )
    )
    manifest = build_capability_manifest(
        matrix,
        evidence,
        baseline_epoch=STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
    )
    manifest.validate_against(matrix)
    return matrix, manifest


def build_stage4_capability_manifest(
    stage3_matrix: CaseMatrix,
    stage3_manifest: CapabilityManifest,
    stage4_matrix: Stage4PdCaseMatrix,
) -> tuple[CaseMatrix, CapabilityManifest]:
    """Promote only PD and KV handoff from one complete Stage4 matrix."""

    if type(stage3_matrix) is not CaseMatrix:
        raise SchemaError("must be a CaseMatrix", path="stage3_matrix")
    if type(stage3_manifest) is not CapabilityManifest:
        raise SchemaError(
            "must be a CapabilityManifest", path="stage3_manifest"
        )
    if type(stage4_matrix) is not Stage4PdCaseMatrix:
        raise SchemaError(
            "must be a Stage4PdCaseMatrix", path="stage4_matrix"
        )
    stage3_manifest.validate_against(stage3_matrix, "stage3_manifest")
    stage4_matrix.validate("stage4_matrix")
    case_ids = {case.id for case in stage3_matrix.cases}
    if any(case_id not in case_ids for case_id in _STAGE4_PROMOTED_CASES):
        raise SchemaError(
            "Stage3 matrix is missing the S1 PD capability cases",
            path="stage3_matrix",
        )
    if any(
        evidence.case_id in _STAGE4_PROMOTED_CASES
        for evidence in stage3_manifest.evidence
    ):
        raise SchemaError(
            "Stage3 manifest already contains Stage4 promotion evidence",
            path="stage3_manifest.evidence",
        )

    additions: tuple[CaseEvidence, ...] = ()
    if stage4_matrix.stage4_ready:
        report_witnesses = tuple(
            (
                report.case_id,
                report.id,
                canonical_digest(report),
                report.artifact.program_artifact_sha256,
            )
            for report in stage4_matrix.reports
        )
        artifact_digest = canonical_digest(
            tuple(item[3] for item in report_witnesses)
        )
        report_digest = canonical_digest(stage4_matrix)
        additions = tuple(
            CaseEvidence(
                case_id=case_id,
                status=stage4_matrix.capability_status,
                evidence_digest=canonical_digest(
                    {
                        "case_id": case_id,
                        "stage4_matrix_id": stage4_matrix.id,
                        "stage4_matrix_digest": report_digest,
                        "reports": report_witnesses,
                    }
                ),
                artifact_digest=artifact_digest,
                report_digest=report_digest,
            )
            for case_id in _STAGE4_PROMOTED_CASES
        )
        for index, evidence in enumerate(additions):
            evidence.validate(f"stage4_evidence[{index}]")

    evidence = tuple(
        sorted(
            (*stage3_manifest.evidence, *additions),
            key=lambda item: item.case_id,
        )
    )
    manifest = build_capability_manifest(
        stage3_matrix,
        evidence,
        baseline_epoch=STAGE4_PD_BASELINE_EPOCH,
    )
    manifest.validate_against(stage3_matrix)
    return stage3_matrix, manifest


def build_capability_manifest(
    case_matrix: CaseMatrix,
    evidence: tuple[CaseEvidence, ...],
    *,
    baseline_epoch: str,
) -> CapabilityManifest:
    """Derive claims; callers cannot directly assert status, score, or acceptance."""

    if type(case_matrix) is not CaseMatrix:
        raise SchemaError("must be a CaseMatrix", path="case_matrix")
    case_matrix.validate("case_matrix")
    validate_nonempty(baseline_epoch, "baseline_epoch")
    if type(evidence) is not tuple:
        raise SchemaError("must be an immutable tuple", path="evidence")
    evidence_by_case: dict[str, CaseEvidence] = {}
    previous_case_id: str | None = None
    case_by_id = {case.id: case for case in case_matrix.cases}
    for index, item in enumerate(evidence):
        item_path = f"evidence[{index}]"
        if type(item) is not CaseEvidence:
            raise SchemaError("must be a CaseEvidence", path=item_path)
        item.validate(item_path)
        if previous_case_id is not None and item.case_id <= previous_case_id:
            raise SchemaError(
                "case_id values must be unique and strictly increasing",
                path=f"{item_path}.case_id",
            )
        previous_case_id = item.case_id
        case = case_by_id.get(item.case_id)
        if case is None:
            raise SchemaError(
                "references an unknown case", path=f"{item_path}.case_id"
            )
        if capability_status_rank(item.status) > capability_status_rank(
            case.maximum_status
        ):
            raise SchemaError(
                "evidence exceeds the case's maximum proof level",
                path=f"{item_path}.status",
            )
        evidence_by_case[item.case_id] = item

    claims: list[CapabilityClaim] = []
    satisfied_capabilities: set[str] = set()
    for definition in case_matrix.capabilities:
        available = tuple(
            evidence_by_case[case_id]
            for case_id in definition.required_case_ids
            if case_id in evidence_by_case
        )
        if len(available) != len(definition.required_case_ids):
            status = CapabilityStatus.UNSUPPORTED
            cited_evidence = ()
        else:
            status = min(available, key=lambda item: capability_status_rank(item.status)).status
            cited_evidence = tuple(item.case_id for item in available)
        satisfied = all(
            _evidence_satisfies(case_by_id[case_id].maximum_status, evidence_by_case.get(case_id))
            for case_id in definition.required_case_ids
        )
        if satisfied:
            satisfied_capabilities.add(definition.key)
        claim = CapabilityClaim(
            key=definition.key,
            stage=definition.stage,
            status=status,
            evidence_case_ids=cited_evidence,
            score_numerator=definition.score_numerator if satisfied else 0,
            score_denominator=definition.score_denominator,
            notes=definition.notes,
        )
        claim.validate(f"claims[{len(claims)}]")
        claims.append(claim)

    acceptance_claims: list[AcceptanceClaim] = []
    for definition in case_matrix.acceptance:
        available_case_ids = tuple(
            case_id
            for case_id in definition.required_case_ids
            if case_id in evidence_by_case
        )
        cases_satisfied = all(
            _evidence_satisfies(case_by_id[case_id].maximum_status, evidence_by_case.get(case_id))
            for case_id in definition.required_case_ids
        )
        blocking = tuple(
            capability_id
            for capability_id in definition.blocking_capability_ids
            if capability_id not in satisfied_capabilities
        )
        claim = AcceptanceClaim(
            key=definition.key,
            stage=definition.stage,
            passed=cases_satisfied and not blocking,
            evidence_case_ids=available_case_ids,
            blocking_capability_ids=blocking,
            notes=definition.notes,
        )
        claim.validate(f"acceptance_claims[{len(acceptance_claims)}]")
        acceptance_claims.append(claim)

    semantic_key = {
        "baseline_epoch": baseline_epoch,
        "case_matrix_id": case_matrix.id,
        "case_matrix_digest": canonical_digest(case_matrix),
        "evidence": evidence,
        "claims": tuple(claims),
        "acceptance_claims": tuple(acceptance_claims),
    }
    result = CapabilityManifest(
        schema_version=CAPABILITY_MANIFEST_SCHEMA_VERSION,
        producer_pass="capability_manifest_builder",
        id=stable_artifact_id(
            "capability_manifest",
            semantic_key,
            schema_version=CAPABILITY_MANIFEST_SCHEMA_VERSION,
        ),
        **semantic_key,
    )
    result.validate()
    return result
