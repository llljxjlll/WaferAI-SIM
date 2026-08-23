"""Closed producer-layer source for a standard linked Swizzle program."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import CommandFragment, LinkedProgramManifest
from .common import stable_artifact_id, validate_nonempty
from .ir1 import IR1
from .swizzle_abi import SwizzleCoreAddressABI
from .swizzle_ir2 import (
    SwizzleIr2Projection,
    admits_wang_4rank_packed_layout,
)
from .swizzle_lowering import SwizzleLoweredProgram
from .swizzle_operand_abi import SwizzleOperandABI
from .swizzle_plan import SwizzleFusionPlan


SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_standard_linked_program/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class SwizzleStandardLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    ir1: IR1
    plan: SwizzleFusionPlan
    projection: SwizzleIr2Projection
    lowered: SwizzleLoweredProgram
    core_abi: SwizzleCoreAddressABI
    operand_abi: SwizzleOperandABI
    fragment: CommandFragment
    manifest: LinkedProgramManifest

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleStandardLinkedProgram":
        result = cls(
            schema_version=SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
            producer_pass="swizzle_standard_linker",
            id=stable_artifact_id(
                "swizzle_standard_linked_program",
                semantic,
                schema_version=SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "ir1", "plan", "projection", "lowered", "core_abi",
                "operand_abi", "fragment", "manifest",
            )
        }

    def validate(self, path: str = "swizzle_standard_linked_program") -> None:
        if self.schema_version != SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "swizzle_standard_linker":
            raise SchemaError("requires exact dedicated producer", path=f"{path}.producer_pass")
        validate_nonempty(self.id, f"{path}.id")
        self.ir1.validate(f"{path}.ir1")
        self.plan.validate(f"{path}.plan")
        self.projection.validate(f"{path}.projection")
        self.lowered.validate(f"{path}.lowered")
        self.core_abi.validate(f"{path}.core_abi")
        self.operand_abi.validate(f"{path}.operand_abi")
        self.fragment.validate(f"{path}.fragment")
        self.manifest.validate(f"{path}.manifest")
        packed_layouts = {
            "swizzle_standard_terminal_root/v1",
            "swizzle_standard_terminal_subview/v1",
            "swizzle_standard_storage_root/v1",
            "swizzle_standard_storage_subview/v1",
        }
        if (
            not admits_wang_4rank_packed_layout(self.projection)
            and any(
                item.layout in packed_layouts
                for item in self.fragment.buffer_abi
            )
        ):
            raise SchemaError(
                "packed BufferABI layouts require exact four-rank Wang AG/RS",
                path=f"{path}.fragment.buffer_abi",
            )
        expected = stable_artifact_id(
            "swizzle_standard_linked_program",
            self._semantic_key(),
            schema_version=SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(self, path: str = "swizzle_standard_linked_program") -> None:
        self.validate(path)
        from ..lowering.swizzle_standard import (
            link_swizzle_standard_manifest,
            lower_swizzle_standard_fragment,
        )

        expected_fragment = lower_swizzle_standard_fragment(
            self.ir1,
            self.plan,
            self.projection,
            self.lowered,
            self.core_abi,
            self.operand_abi,
        )
        if self.fragment != expected_fragment:
            raise SchemaError(
                "embedded fragment is not the exact producer result",
                path=f"{path}.fragment",
            )
        expected_manifest = link_swizzle_standard_manifest(
            self.ir1,
            self.plan,
            self.projection,
            self.lowered,
            self.core_abi,
            self.operand_abi,
            self.fragment,
        )
        if self.manifest != expected_manifest:
            raise SchemaError(
                "embedded manifest is not the exact producer result",
                path=f"{path}.manifest",
            )


__all__ = [
    "SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION",
    "SwizzleStandardLinkedProgram",
]
