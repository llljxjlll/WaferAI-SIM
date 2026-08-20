"""Aggregate existing Stage 4 runtime reports into a readiness matrix."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.stage4_pd import Stage4KvReshardKind, Stage4PdMode
from ..schema.stage4_pd_case_matrix import Stage4PdCaseMatrix
from ..schema.stage4_pd_evidence import Stage4PdRuntimeReport


_CASE_RANK = {
    (Stage4PdMode.FUSED, Stage4KvReshardKind.NONE): 0,
    (Stage4PdMode.SEPARATED, Stage4KvReshardKind.ONE_TO_ONE): 1,
    (Stage4PdMode.SEPARATED, Stage4KvReshardKind.GATHER): 2,
}


def build_stage4_pd_case_matrix(
    reports: tuple[Stage4PdRuntimeReport, ...] = (),
) -> Stage4PdCaseMatrix:
    """Build a canonical matrix without creating or upgrading evidence."""

    if type(reports) is not tuple:
        raise SchemaError("must be an immutable tuple", path="reports")
    ranked: list[tuple[int, Stage4PdRuntimeReport]] = []
    for index, report in enumerate(reports):
        if type(report) is not Stage4PdRuntimeReport:
            raise SchemaError(
                "must be a Stage4PdRuntimeReport",
                path=f"reports[{index}]",
            )
        report.validate(f"reports[{index}]")
        key = report.mode, report.reshard
        rank = _CASE_RANK.get(key)
        if rank is None:
            raise SchemaError(
                "must be PD-F, PDS equal-TP, or PDR TP2-to-TP1 gather",
                path=f"reports[{index}].reshard",
            )
        ranked.append((rank, report))
    canonical = tuple(
        report for _rank, report in sorted(ranked, key=lambda item: item[0])
    )
    return Stage4PdCaseMatrix.create(reports=canonical)


__all__ = ["build_stage4_pd_case_matrix"]
