"""Closed producer-layer source for a standard UNFUSED comparison program."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import CommandFragment, LinkedProgramManifest
from .common import stable_artifact_id
from .ir1 import IR1
from .swizzle_unfused import UnfusedComparisonPlan, UnfusedComparisonProjection
from .swizzle_unfused_abi import UnfusedComparisonCoreABI
from .swizzle_unfused_lowering import (
    UnfusedComparisonLoweredProgram,
    UnfusedComparisonOperandABI,
)


UNFUSED_COMPARISON_STANDARD_LINKED_SCHEMA_VERSION = (
    "wafer_frontend.unfused_comparison_standard_linked/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class UnfusedComparisonStandardLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    ir1: IR1
    plan: UnfusedComparisonPlan
    projection: UnfusedComparisonProjection
    lowered: UnfusedComparisonLoweredProgram
    core_abi: UnfusedComparisonCoreABI
    operand_abi: UnfusedComparisonOperandABI
    fragment: CommandFragment
    manifest: LinkedProgramManifest

    @classmethod
    def create(cls, **semantic: object) -> "UnfusedComparisonStandardLinkedProgram":
        result = cls(
            schema_version=UNFUSED_COMPARISON_STANDARD_LINKED_SCHEMA_VERSION,
            producer_pass="unfused_comparison_standard_linker",
            id=stable_artifact_id(
                "unfused_comparison_standard_linked", semantic,
                schema_version=UNFUSED_COMPARISON_STANDARD_LINKED_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "ir1", "plan", "projection", "lowered", "core_abi",
            "operand_abi", "fragment", "manifest",
        )}

    def validate(self, path: str = "unfused_comparison_standard_linked") -> None:
        if self.schema_version != UNFUSED_COMPARISON_STANDARD_LINKED_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "unfused_comparison_standard_linker":
            raise SchemaError("requires exact producer", path=f"{path}.producer_pass")
        self.ir1.validate(f"{path}.ir1")
        self.plan.validate(f"{path}.plan")
        self.projection.validate(f"{path}.projection")
        self.lowered.validate(f"{path}.lowered")
        self.core_abi.validate(f"{path}.core_abi")
        self.operand_abi.validate(f"{path}.operand_abi")
        self.fragment.validate(f"{path}.fragment")
        self.manifest.validate(f"{path}.manifest")
        expected = stable_artifact_id(
            "unfused_comparison_standard_linked", self._semantic_key(),
            schema_version=UNFUSED_COMPARISON_STANDARD_LINKED_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(self, path: str = "unfused_comparison_standard_linked") -> None:
        self.validate(path)
        from ..lowering.swizzle_unfused_standard import (
            link_unfused_comparison_manifest,
            lower_unfused_comparison_fragment,
        )
        self.projection.validate_against(self.ir1, self.plan, f"{path}.projection")
        self.lowered.validate_against(self.plan, self.projection, f"{path}.lowered")
        self.core_abi.validate_against(self.ir1, self.plan, self.projection, f"{path}.core_abi")
        self.operand_abi.validate_against(self.ir1, self.plan, self.projection, f"{path}.operand_abi")
        fragment = lower_unfused_comparison_fragment(
            self.ir1, self.plan, self.projection, self.lowered,
            self.core_abi, self.operand_abi,
        )
        if self.fragment != fragment:
            raise SchemaError("fragment is not the exact producer result", path=f"{path}.fragment")
        manifest = link_unfused_comparison_manifest(
            self.ir1, self.plan, self.projection, self.lowered,
            self.core_abi, self.operand_abi, fragment,
        )
        if self.manifest != manifest:
            raise SchemaError("manifest is not the exact producer result", path=f"{path}.manifest")


__all__ = [
    "UNFUSED_COMPARISON_STANDARD_LINKED_SCHEMA_VERSION",
    "UnfusedComparisonStandardLinkedProgram",
]
