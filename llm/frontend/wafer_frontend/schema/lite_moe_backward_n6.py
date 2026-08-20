"""Typed N6 carriers for the isolated S3-Lite MoE backward preview."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import CommandFragment, LinkedProgramManifest
from .common import stable_artifact_id
from .lite_moe import LiteMoeStaticTrace
from .lite_moe_backward import LiteMoeBackwardOverlay
from .lite_moe_execution import LiteMoeGlobalDag, LiteMoeProjection, LiteMoeScheduled
from .lite_moe_n4 import LiteMoeN4IR1
from .lite_moe_n6 import LiteMoeN6Intent


LITE_MOE_BACKWARD_LOWERED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_backward_lowered_program/v1alpha1"
)
LITE_MOE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_backward_linked_program/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardLoweredProgram:
    schema_version: str
    producer_pass: str
    id: str
    n4: LiteMoeN4IR1
    projection: LiteMoeProjection
    schedule: LiteMoeScheduled
    global_dag: LiteMoeGlobalDag
    n6_intent: LiteMoeN6Intent
    trace: LiteMoeStaticTrace
    overlay: LiteMoeBackwardOverlay
    fragments: tuple[CommandFragment, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeBackwardLoweredProgram":
        semantic["fragments"] = tuple(
            sorted(semantic["fragments"], key=lambda item: item.id)  # type: ignore[index,union-attr]
        )
        result = cls(
            LITE_MOE_BACKWARD_LOWERED_PROGRAM_SCHEMA_VERSION,
            "lite_moe_backward_lower_program",
            stable_artifact_id(
                "s3_lite_moe_backward_lowered_program",
                semantic,
                schema_version=LITE_MOE_BACKWARD_LOWERED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "lite_moe_backward_lowered_program") -> None:
        if (
            self.schema_version != LITE_MOE_BACKWARD_LOWERED_PROGRAM_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_backward_lower_program"
        ):
            raise SchemaError("unsupported lowered schema/producer", path=path)
        from ..lowering.lite_moe_backward import lower_lite_moe_backward

        expected = lower_lite_moe_backward(
            self.overlay,
            self.n4,
            self.projection,
            self.schedule,
            self.global_dag,
            self.n6_intent,
            self.trace,
        )
        if self.fragments != expected:
            raise SchemaError(
                "fragments are not the exact production backward quotient",
                path=f"{path}.fragments",
            )
        if (
            len(self.fragments) != 28
            or sum(len(fragment.claimed_action_ids) for fragment in self.fragments) != 32
        ):
            raise SchemaError(
                "backward lowering must carry exact 28 leaves/32 claims",
                path=f"{path}.fragments",
            )
        expected_id = stable_artifact_id(
            "s3_lite_moe_backward_lowered_program",
            self._semantic(),
            schema_version=LITE_MOE_BACKWARD_LOWERED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable lowered id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeBackwardLoweredProgram
    manifest: LinkedProgramManifest

    @classmethod
    def create(
        cls,
        *,
        source: LiteMoeBackwardLoweredProgram,
        manifest: LinkedProgramManifest,
    ) -> "LiteMoeBackwardLinkedProgram":
        semantic = {"source": source, "manifest": manifest}
        result = cls(
            LITE_MOE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION,
            "lite_moe_backward_manifest_linker",
            stable_artifact_id(
                "s3_lite_moe_backward_linked_program",
                semantic,
                schema_version=LITE_MOE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {"source": self.source, "manifest": self.manifest}

    def validate(self, path: str = "lite_moe_backward_linked_program") -> None:
        if (
            self.schema_version != LITE_MOE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_backward_manifest_linker"
        ):
            raise SchemaError("unsupported linked schema/producer", path=path)
        if type(self.source) is not LiteMoeBackwardLoweredProgram:
            raise SchemaError(
                "must embed LiteMoeBackwardLoweredProgram", path=f"{path}.source"
            )
        self.source.validate(f"{path}.source")
        if type(self.manifest) is not LinkedProgramManifest:
            raise SchemaError("must embed LinkedProgramManifest", path=f"{path}.manifest")
        self.manifest.validate(f"{path}.manifest")
        from ..lowering.lite_moe_backward_linker import link_lite_moe_backward_manifest

        if self.manifest != link_lite_moe_backward_manifest(self.source):
            raise SchemaError(
                "manifest is not the exact production backward quotient",
                path=f"{path}.manifest",
            )
        expected_id = stable_artifact_id(
            "s3_lite_moe_backward_linked_program",
            self._semantic(),
            schema_version=LITE_MOE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable linked id", path=f"{path}.id")


__all__ = [
    "LITE_MOE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION",
    "LITE_MOE_BACKWARD_LOWERED_PROGRAM_SCHEMA_VERSION",
    "LiteMoeBackwardLinkedProgram",
    "LiteMoeBackwardLoweredProgram",
]
