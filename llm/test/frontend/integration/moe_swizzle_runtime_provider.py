"""W9 handoff boundary for W11/W12 MoE runtime artifacts.

No artifact is synthesized here.  A production provider must lower/link the
exact case branch and return its ProgramIo files.  Provider preflight is
strictly ordered after the actual 168-run calibration evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MoeCalibrationStatus,
)

from run_moe_swizzle_calibration import MoeSwizzleCalibrationRunEvidence
from moe_swizzle_runtime_suite import (
    MoeSwizzleRuntimeBranch,
    MoeSwizzleRuntimeBranchPlan,
    MoeSwizzleRuntimeCasePlan,
    MoeSwizzleRuntimeSuitePlan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def moe_swizzle_calibration_run_digest(
    evidence: MoeSwizzleCalibrationRunEvidence,
) -> str:
    evidence.validate("moe_runtime_provider.calibration")
    return canonical_digest({
        "profile": evidence.profile,
        "matching_tool_sha256": evidence.matching_tool_sha256,
        "hardware_sha256": evidence.hardware_sha256,
        "simulation_sha256": evidence.simulation_sha256,
        "mapping_sha256": evidence.mapping_sha256,
        "run_count": evidence.run_count,
        "raw_output_sha256": tuple(_sha256(item) for item in evidence.raw_output_paths),
        "artifacts": tuple(
            (
                item.key,
                item.production_source_ref,
                item.program_sha256,
                item.linked_manifest_sha256,
                item.program_io_sha256,
            )
            for item in evidence.artifacts
        ),
    })


@dataclass(frozen=True, slots=True)
class MoeSwizzleRuntimeExecutable:
    case_ref: str
    branch: MoeSwizzleRuntimeBranch
    same_work_digest: str
    deployment_digest: str
    workload_selection_ref: str | None
    source_workload_selection_id: str | None
    production_source_ref: str
    source_execution_id: str
    program: Path
    linked_manifest: Path
    program_io: Path
    hardware_config: Path
    simulation_config: Path
    mapping_config: Path
    expected_probe_count: int
    expected_probe_bytes: int
    expected_physical_root_count: int
    expected_ack_count: int
    expected_done_count: int

    def validate_against(
        self,
        case: MoeSwizzleRuntimeCasePlan,
        branch: MoeSwizzleRuntimeBranchPlan,
        path: str = "moe_swizzle_runtime_executable",
    ) -> None:
        if (
            self.case_ref != case.id
            or self.branch is not branch.branch
            or self.same_work_digest != canonical_digest(case.same_work)
            or self.deployment_digest != canonical_digest(branch)
            or self.workload_selection_ref != branch.workload_selection_ref
            or self.source_workload_selection_id != branch.workload_selection_ref
            or self.source_execution_id != case.execution.id
        ):
            raise SchemaError("executable source/branch provenance drifted", path=path)
        for name in ("workload_selection_ref", "source_workload_selection_id"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise SchemaError(
                    "joint selection lineage must be nonempty or absent",
                    path=f"{path}.{name}",
                )
        if type(self.production_source_ref) is not str or not self.production_source_ref:
            raise SchemaError("requires a production linked source ref", path=f"{path}.production_source_ref")
        for name in (
            "program", "linked_manifest", "program_io", "hardware_config",
            "simulation_config", "mapping_config",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_file():
                raise SchemaError("must be an existing regular file", path=f"{path}.{name}")
        for name in (
            "expected_probe_count", "expected_probe_bytes",
            "expected_physical_root_count", "expected_ack_count",
            "expected_done_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SchemaError("must be a positive production-derived count", path=f"{path}.{name}")


class MoeSwizzleRuntimeArtifactProvider(Protocol):
    def materialize(
        self,
        case: MoeSwizzleRuntimeCasePlan,
        branch: MoeSwizzleRuntimeBranchPlan,
        output_root: Path,
    ) -> MoeSwizzleRuntimeExecutable: ...


class PendingW9MoeSwizzleRuntimeProvider:
    """Fail-closed placeholder until W9 hands off linked production sources."""

    def materialize(
        self,
        case: MoeSwizzleRuntimeCasePlan,
        branch: MoeSwizzleRuntimeBranchPlan,
        output_root: Path,
    ) -> MoeSwizzleRuntimeExecutable:
        del case, branch, output_root
        raise SchemaError(
            "W9 production linked MoE artifact provider is not installed",
            path="moe_swizzle_runtime_provider",
        )


def preflight_moe_swizzle_runtime_provider(
    *,
    calibration: MoeSwizzleCalibrationRunEvidence,
    suite: MoeSwizzleRuntimeSuitePlan,
    provider: MoeSwizzleRuntimeArtifactProvider,
    output_root: Path,
) -> tuple[MoeSwizzleRuntimeExecutable, ...]:
    """Materialize every branch only after actual calibration is exact."""

    calibration.validate("moe_runtime_provider.calibration")
    suite.validate("moe_runtime_provider.suite")
    if (
        calibration.profile.status is not MoeCalibrationStatus.MEASURED
        or calibration.profile != suite.calibration_profile
        or moe_swizzle_calibration_run_digest(calibration)
        != suite.calibration_run_digest
    ):
        raise SchemaError(
            "suite/provider requires the exact actual 168-run calibration",
            path="moe_swizzle_runtime_provider.calibration",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    result = []
    for case_index, case in enumerate(suite.cases):
        for branch_index, branch in enumerate(case.branches):
            branch_root = output_root / (
                f"case-{case_index:02d}-{case.spec.name}-{case.execution.mode.value}-"
                f"{case.scope.value}-{branch.branch.value}"
            )
            branch_root.mkdir(parents=True, exist_ok=True)
            executable = provider.materialize(case, branch, branch_root)
            executable.validate_against(
                case,
                branch,
                f"moe_runtime_provider.executables[{case_index}][{branch_index}]",
            )
            result.append(executable)
    return tuple(result)


__all__ = [
    "MoeSwizzleRuntimeArtifactProvider",
    "MoeSwizzleRuntimeExecutable",
    "PendingW9MoeSwizzleRuntimeProvider",
    "moe_swizzle_calibration_run_digest",
    "preflight_moe_swizzle_runtime_provider",
]
