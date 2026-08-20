from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_stage4_pd_case_matrix as public_build_stage4_pd_case_matrix,
)
from llm.frontend.wafer_frontend.passes.stage4_pd_case_matrix import (
    build_stage4_pd_case_matrix,
)
from llm.frontend.wafer_frontend.schema import (
    Stage4PdCaseMatrix as PublicStage4PdCaseMatrix,
)
from llm.frontend.wafer_frontend.schema.capability import CapabilityStatus
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.stage4_pd import (
    Stage4KvReshardKind,
)
from llm.frontend.wafer_frontend.schema.stage4_pd_case_matrix import (
    STAGE4_PD_CASE_MATRIX_SCHEMA_VERSION,
    Stage4PdCaseMatrix,
    stage4_pd_case_key,
)
from llm.frontend.wafer_frontend.schema.stage4_pd_evidence import (
    Stage4PdNamedDigest,
)

from test_stage4_lower_program_segmented_transfer import _pipeline
from test_stage4_pd_evidence import _artifact, _recreate, _report


def _matrix_reports():
    fused, _fused_plan, _fused_oracle, _fused_manifest = _report(
        1, 1, fused=True
    )
    pds, _pds_plan, _pds_oracle, _pds_manifest = _report(1, 1)
    pds = _recreate(pds, case_id="case.stage4.pds.tp1")

    pdr, _pdr_plan, _pdr_oracle, _fixture_manifest = _report(2, 1)
    pdr_manifest = _pipeline()[2].manifest
    pdr_inputs = tuple(
        Stage4PdNamedDigest(item.name, canonical_digest(pdr_manifest))
        if item.name == "manifest"
        else item
        for item in pdr.input_digests
    )
    pdr = _recreate(
        pdr,
        case_id="case.stage4.pdr.tp2_to_tp1",
        artifact=_artifact(pdr_manifest),
        input_digests=pdr_inputs,
    )
    return fused, pds, pdr


class Stage4PdCaseMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fused, cls.pds, cls.pdr = _matrix_reports()

    def test_empty_partial_and_full_readiness_are_exact(self) -> None:
        self.assertIs(PublicStage4PdCaseMatrix, Stage4PdCaseMatrix)
        self.assertIs(
            public_build_stage4_pd_case_matrix,
            build_stage4_pd_case_matrix,
        )
        self.assertEqual(
            STAGE4_PD_CASE_MATRIX_SCHEMA_VERSION,
            "wafer_frontend.stage4_pd_case_matrix/v1alpha2",
        )

        empty = build_stage4_pd_case_matrix()
        one = build_stage4_pd_case_matrix((self.fused,))
        two = build_stage4_pd_case_matrix((self.pds, self.fused))
        for matrix, expected_reports in (
            (empty, ()),
            (one, (self.fused,)),
            (two, (self.fused, self.pds)),
        ):
            with self.subTest(count=len(expected_reports)):
                self.assertEqual(matrix.reports, expected_reports)
                self.assertIs(
                    matrix.capability_status, CapabilityStatus.UNSUPPORTED
                )
                self.assertFalse(matrix.stage4_ready)
                matrix.validate()

        full = build_stage4_pd_case_matrix(
            (self.pdr, self.fused, self.pds)
        )
        self.assertEqual(full.reports, (self.fused, self.pds, self.pdr))
        self.assertIs(full.capability_status, CapabilityStatus.E2E_TIMING)
        self.assertTrue(full.stage4_ready)
        self.assertEqual(
            tuple(stage4_pd_case_key(report) for report in full.reports),
            (
                (
                    self.fused.mode,
                    Stage4KvReshardKind.NONE,
                ),
                (
                    self.pds.mode,
                    Stage4KvReshardKind.ONE_TO_ONE,
                ),
                (
                    self.pdr.mode,
                    Stage4KvReshardKind.GATHER,
                ),
            ),
        )
        for field in ("case_id", "id", "plan_id", "oracle_id"):
            values = tuple(getattr(report, field) for report in full.reports)
            self.assertEqual(len(values), len(set(values)))
        manifest_ids = tuple(
            report.artifact.linked_manifest_id for report in full.reports
        )
        self.assertEqual(len(manifest_ids), len(set(manifest_ids)))
        self.assertEqual(
            build_stage4_pd_case_matrix(tuple(reversed(full.reports))), full
        )
        self.assertEqual(
            loads_dataclass(
                Stage4PdCaseMatrix,
                canonical_json(full),
                path="matrix",
            ),
            full,
        )

    def test_strict_serde_order_identity_and_readiness_fail_closed(self) -> None:
        full = build_stage4_pd_case_matrix(
            (self.fused, self.pds, self.pdr)
        )
        raw = json.loads(canonical_json(full))
        raw["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(
                Stage4PdCaseMatrix, json.dumps(raw), path="matrix"
            )
        del raw["unexpected"]
        del raw["reports"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(
                Stage4PdCaseMatrix, json.dumps(raw), path="matrix"
            )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(full, schema_version="v0").validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(full, id="forged").validate()
        with self.assertRaisesRegex(SchemaError, "PD-F/PDS/PDR order"):
            replace(full, reports=tuple(reversed(full.reports))).validate()
        with self.assertRaisesRegex(SchemaError, "reports must be unique"):
            build_stage4_pd_case_matrix((self.fused, self.pds, self.pds))
        with self.assertRaisesRegex(SchemaError, "only the exact"):
            replace(full, stage4_ready=False).validate()
        partial = build_stage4_pd_case_matrix((self.fused,))
        with self.assertRaisesRegex(SchemaError, "only the exact"):
            replace(
                partial,
                capability_status=CapabilityStatus.E2E_TIMING,
                stage4_ready=True,
            ).validate()
        with self.assertRaisesRegex(SchemaError, "immutable tuple"):
            build_stage4_pd_case_matrix([self.fused])  # type: ignore[arg-type]

    def test_case_epoch_tool_and_input_provenance_fail_closed(self) -> None:
        wrong_case = _recreate(self.pds, case_id="case.stage4.not_pds")
        with self.assertRaisesRegex(SchemaError, "case.stage4.pds.tp1"):
            build_stage4_pd_case_matrix((self.fused, wrong_case, self.pdr))

        wrong_kind = _recreate(
            self.pdr, reshard=Stage4KvReshardKind.SCATTER
        )
        with self.assertRaisesRegex(SchemaError, "PD-F, PDS equal-TP"):
            build_stage4_pd_case_matrix((self.fused, self.pds, wrong_kind))

        moving_epoch = replace(self.pdr, baseline_epoch="moving")
        with self.assertRaisesRegex(SchemaError, "s1-n-v1"):
            build_stage4_pd_case_matrix(
                (self.fused, self.pds, moving_epoch)
            )

        changed_tools = tuple(
            Stage4PdNamedDigest(item.name, "2" * 64)
            if item.name == "runner"
            else item
            for item in self.pdr.tool_digests
        )
        wrong_tool = _recreate(self.pdr, tool_digests=changed_tools)
        with self.assertRaisesRegex(SchemaError, "exact tool digest set"):
            build_stage4_pd_case_matrix(
                (self.fused, self.pds, wrong_tool)
            )

        changed_policy = replace(
            self.pdr.policy, planning_context_id="different"
        )
        changed_policy_inputs = tuple(
            Stage4PdNamedDigest(item.name, canonical_digest(changed_policy))
            if item.name == "policy"
            else item
            for item in self.pdr.input_digests
        )
        wrong_policy = _recreate(
            self.pdr,
            policy=changed_policy,
            input_digests=changed_policy_inputs,
        )
        with self.assertRaisesRegex(SchemaError, "exact policy identity"):
            build_stage4_pd_case_matrix(
                (self.fused, self.pds, wrong_policy)
            )

        wrong_inputs = replace(
            self.pdr, input_digests=self.pdr.input_digests[:-1]
        )
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            build_stage4_pd_case_matrix(
                (self.fused, self.pds, wrong_inputs)
            )

        duplicate_manifest = _recreate(
            self.pdr,
            artifact=replace(
                self.pdr.artifact,
                linked_manifest_id=self.pds.artifact.linked_manifest_id,
            ),
        )
        with self.assertRaisesRegex(SchemaError, "manifest_id must be unique"):
            build_stage4_pd_case_matrix(
                (self.fused, self.pds, duplicate_manifest)
            )
        for field_name in ("plan_id", "oracle_id"):
            with self.subTest(identity=field_name):
                duplicate_identity = _recreate(
                    self.pdr,
                    **{
                        field_name: getattr(self.pds, field_name),
                    },
                )
                with self.assertRaisesRegex(
                    SchemaError, f"{field_name} must be unique"
                ):
                    build_stage4_pd_case_matrix(
                        (self.fused, self.pds, duplicate_identity)
                    )


if __name__ == "__main__":
    unittest.main()
