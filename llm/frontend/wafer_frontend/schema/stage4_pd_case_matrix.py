"""Exact Stage 4 PD runtime-report readiness matrix."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .capability import CapabilityStatus
from .common import stable_artifact_id
from .stage4_pd import Stage4KvReshardKind, Stage4PdMode
from .stage4_pd_evidence import Stage4PdRuntimeReport


STAGE4_PD_CASE_MATRIX_SCHEMA_VERSION = (
    "wafer_frontend.stage4_pd_case_matrix/v1alpha2"
)

_CASE_ORDER = (
    (Stage4PdMode.FUSED, Stage4KvReshardKind.NONE),
    (Stage4PdMode.SEPARATED, Stage4KvReshardKind.ONE_TO_ONE),
    (Stage4PdMode.SEPARATED, Stage4KvReshardKind.GATHER),
)
_CASE_RANK = {key: index for index, key in enumerate(_CASE_ORDER)}
_EXPECTED_CASE_IDS = {
    (Stage4PdMode.FUSED, Stage4KvReshardKind.NONE):
        "case.stage4.pd_f.tp1",
    (Stage4PdMode.SEPARATED, Stage4KvReshardKind.ONE_TO_ONE):
        "case.stage4.pds.tp1",
    (Stage4PdMode.SEPARATED, Stage4KvReshardKind.GATHER):
        "case.stage4.pdr.tp2_to_tp1",
}


def stage4_pd_case_key(
    report: Stage4PdRuntimeReport,
) -> tuple[Stage4PdMode, Stage4KvReshardKind]:
    """Return the frozen PD-F/PDS/PDR case key for one typed report."""

    if type(report) is not Stage4PdRuntimeReport:
        raise SchemaError(
            "must be a Stage4PdRuntimeReport", path="stage4_pd_report"
        )
    return report.mode, report.reshard


@dataclass(frozen=True, slots=True)
class Stage4PdCaseMatrix:
    """Readiness derived only from validated reports and frozen case IDs.

    Runtime reports do not carry TP degrees directly, so the PDR TP2-to-TP1
    boundary is identified by its typed GATHER report plus exact case ID.
    """

    schema_version: str
    producer_pass: str
    id: str
    reports: tuple[Stage4PdRuntimeReport, ...]
    capability_status: CapabilityStatus
    stage4_ready: bool

    @classmethod
    def create(
        cls,
        *,
        reports: tuple[Stage4PdRuntimeReport, ...],
    ) -> "Stage4PdCaseMatrix":
        complete = len(reports) == len(_CASE_ORDER)
        semantic_key = {
            "reports": reports,
            "capability_status": (
                CapabilityStatus.E2E_TIMING
                if complete
                else CapabilityStatus.UNSUPPORTED
            ),
            "stage4_ready": complete,
        }
        result = cls(
            schema_version=STAGE4_PD_CASE_MATRIX_SCHEMA_VERSION,
            producer_pass="stage4_pd_case_matrix",
            id=stable_artifact_id(
                "stage4_pd_case_matrix",
                semantic_key,
                schema_version=STAGE4_PD_CASE_MATRIX_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "reports": self.reports,
            "capability_status": self.capability_status,
            "stage4_ready": self.stage4_ready,
        }

    def validate(self, path: str = "stage4_pd_case_matrix") -> None:
        if self.schema_version != STAGE4_PD_CASE_MATRIX_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "stage4_pd_case_matrix":
            raise SchemaError(
                "must be 'stage4_pd_case_matrix'",
                path=f"{path}.producer_pass",
            )
        if type(self.reports) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.reports"
            )
        if len(self.reports) > len(_CASE_ORDER):
            raise SchemaError(
                "cannot contain more than PD-F/PDS/PDR",
                path=f"{path}.reports",
            )

        keys: list[tuple[Stage4PdMode, Stage4KvReshardKind]] = []
        identity_columns: dict[str, list[str]] = {
            "case_id": [],
            "report_id": [],
            "plan_id": [],
            "oracle_id": [],
            "manifest_id": [],
        }
        epochs: list[str] = []
        tool_sets: list[object] = []
        policy_sets: list[object] = []
        input_name_sets: list[tuple[str, ...]] = []
        for index, report in enumerate(self.reports):
            report_path = f"{path}.reports[{index}]"
            if type(report) is not Stage4PdRuntimeReport:
                raise SchemaError(
                    "must be a Stage4PdRuntimeReport", path=report_path
                )
            report.validate(report_path)
            key = stage4_pd_case_key(report)
            if key not in _CASE_RANK:
                raise SchemaError(
                    "must be PD-F, PDS equal-TP, or PDR TP2-to-TP1 gather",
                    path=f"{report_path}.reshard",
                )
            if report.case_id != _EXPECTED_CASE_IDS[key]:
                raise SchemaError(
                    f"must be {_EXPECTED_CASE_IDS[key]!r} for this case",
                    path=f"{report_path}.case_id",
                )
            keys.append(key)
            identity_columns["case_id"].append(report.case_id)
            identity_columns["report_id"].append(report.id)
            identity_columns["plan_id"].append(report.plan_id)
            identity_columns["oracle_id"].append(report.oracle_id)
            identity_columns["manifest_id"].append(
                report.artifact.linked_manifest_id
            )
            epochs.append(report.baseline_epoch)
            tool_sets.append(report.tool_digests)
            policy_sets.append(report.policy)
            input_name_sets.append(
                tuple(item.name for item in report.input_digests)
            )

        expected_keys = tuple(sorted(set(keys), key=_CASE_RANK.__getitem__))
        if tuple(keys) != expected_keys:
            raise SchemaError(
                "reports must be unique and in PD-F/PDS/PDR order",
                path=f"{path}.reports",
            )
        for name, values in identity_columns.items():
            if len(values) != len(set(values)):
                raise SchemaError(
                    f"{name} must be unique across cases",
                    path=f"{path}.reports",
                )
        if epochs and len(set(epochs)) != 1:
            raise SchemaError(
                "reports must share one baseline epoch",
                path=f"{path}.reports",
            )
        if tool_sets and any(value != tool_sets[0] for value in tool_sets[1:]):
            raise SchemaError(
                "reports must share one exact tool digest set",
                path=f"{path}.reports",
            )
        if policy_sets and any(
            value != policy_sets[0] for value in policy_sets[1:]
        ):
            raise SchemaError(
                "reports must share one exact policy identity",
                path=f"{path}.reports",
            )
        if input_name_sets and any(
            value != input_name_sets[0] for value in input_name_sets[1:]
        ):
            raise SchemaError(
                "reports must share one canonical input digest name set",
                path=f"{path}.reports",
            )

        complete = tuple(keys) == _CASE_ORDER
        expected_status = (
            CapabilityStatus.E2E_TIMING
            if complete
            else CapabilityStatus.UNSUPPORTED
        )
        if type(self.stage4_ready) is not bool:
            raise SchemaError(
                "must be a bool", path=f"{path}.stage4_ready"
            )
        if (
            self.stage4_ready is not complete
            or self.capability_status is not expected_status
        ):
            raise SchemaError(
                "only the exact PD-F/PDS/PDR matrix is Stage 4 ready",
                path=path,
            )
        expected_id = stable_artifact_id(
            "stage4_pd_case_matrix",
            self._semantic_key(),
            schema_version=STAGE4_PD_CASE_MATRIX_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


__all__ = [
    "STAGE4_PD_CASE_MATRIX_SCHEMA_VERSION",
    "Stage4PdCaseMatrix",
    "stage4_pd_case_key",
]
