"""Dedicated pre-link N6 carrier for the S2-Lite LM-head train case."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..lowering.context import LoweringContext
from .artifact_manifest import (
    CommandFragment,
    LinkedProgramManifest,
    RegionManifest,
)
from .common import stable_artifact_id
from .ir2 import BufferOwnership
from .n6 import _leaf_fragment, _validate_lowered_fragments
from .train_global_action import S2LiteTrainGlobalAction


S2_LITE_TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_train_lowered_program/v1alpha1"
)
S2_LITE_TRAIN_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_train_linked_program/v1alpha1"
)


def s2_lite_train_lowering_context(
    source: S2LiteTrainGlobalAction,
) -> LoweringContext:
    """Recover the one exact DP1 lowering context from the Lite carrier."""

    if type(source) is not S2LiteTrainGlobalAction:
        raise SchemaError(
            "must be an S2LiteTrainGlobalAction", path="source"
        )
    source.validate("source")
    if len(source.source.replicas) != 1 or len(source.global_dags) != 1:
        raise SchemaError(
            "S2-Lite lowering requires one exact DP1 source",
            path="source",
        )
    scheduled = source.source.replicas[0]
    projected = scheduled.projected
    context = LoweringContext(
        ir1=projected.graph,
        fusion_plans=projected.fusion_plans,
        standalone_plans=projected.standalone_plans,
        projection=projected.projection,
        schedule_set=scheduled.schedule_set,
        global_dag=source.global_dags[0],
    )
    context.validate("s2_lite_lowering_context")
    aliased = tuple(
        binding
        for schedule in context.schedule_set.schedules
        for binding in schedule.buffer_bindings
        if binding.ownership is BufferOwnership.ALIASED
    )
    if len(aliased) != 1:
        raise SchemaError(
            "S2-Lite lowering requires one exact trainable-state alias",
            path="s2_lite_lowering_context.schedule_set",
        )
    return context


@dataclass(frozen=True, slots=True)
class S2LiteTrainLoweredProgram:
    """Canonical production fragments for one Lite GlobalAction timeline."""

    schema_version: str
    producer_pass: str
    id: str
    source: S2LiteTrainGlobalAction
    lowering_context: LoweringContext
    fragments: tuple[CommandFragment | RegionManifest, ...]

    @classmethod
    def create(
        cls,
        *,
        source: S2LiteTrainGlobalAction,
        lowering_context: LoweringContext,
        fragments: tuple[CommandFragment | RegionManifest, ...],
    ) -> "S2LiteTrainLoweredProgram":
        semantic_key = {
            "source": source,
            "lowering_context": lowering_context,
            "fragments": tuple(
                sorted(fragments, key=lambda item: _leaf_fragment(item).id)
            ),
        }
        result = cls(
            schema_version=S2_LITE_TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
            producer_pass="s2_lite_train_lowering",
            id=stable_artifact_id(
                "s2_lite_train_lowered_program",
                semantic_key,
                schema_version=S2_LITE_TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source": self.source,
            "lowering_context": self.lowering_context,
            "fragments": self.fragments,
        }

    def validate(self, path: str = "s2_lite_train_lowered_program") -> None:
        if self.schema_version != S2_LITE_TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "s2_lite_train_lowering":
            raise SchemaError(
                "must be 's2_lite_train_lowering'",
                path=f"{path}.producer_pass",
            )
        if type(self.source) is not S2LiteTrainGlobalAction:
            raise SchemaError(
                "must embed one exact S2LiteTrainGlobalAction",
                path=f"{path}.source",
            )
        self.source.validate(f"{path}.source")
        if type(self.lowering_context) is not LoweringContext:
            raise SchemaError(
                "must be a LoweringContext",
                path=f"{path}.lowering_context",
            )
        self.lowering_context.validate(f"{path}.lowering_context")
        if self.lowering_context != s2_lite_train_lowering_context(self.source):
            raise SchemaError(
                "must preserve the exact Lite GlobalAction lowering context",
                path=f"{path}.lowering_context",
            )
        if self.fragments != tuple(
            sorted(self.fragments, key=lambda item: _leaf_fragment(item).id)
        ):
            raise SchemaError(
                "fragments must use canonical leaf-id order",
                path=f"{path}.fragments",
            )
        _validate_lowered_fragments(
            self.fragments,
            self.lowering_context,
            f"{path}.fragments",
        )
        expected_id = stable_artifact_id(
            "s2_lite_train_lowered_program",
            self._semantic_key(),
            schema_version=S2_LITE_TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: S2LiteTrainGlobalAction,
        path: str = "s2_lite_train_lowered_program",
    ) -> None:
        self.validate(path)
        if self.source != source:
            raise SchemaError(
                "must embed the exact Lite GlobalAction source",
                path=f"{path}.source",
            )


@dataclass(frozen=True, slots=True)
class S2LiteTrainLinkedProgram:
    """One production-linked manifest for the single Lite timeline."""

    schema_version: str
    producer_pass: str
    id: str
    source: S2LiteTrainLoweredProgram
    manifest: LinkedProgramManifest

    @classmethod
    def create(
        cls,
        *,
        source: S2LiteTrainLoweredProgram,
        manifest: LinkedProgramManifest,
    ) -> "S2LiteTrainLinkedProgram":
        semantic_key = {"source": source, "manifest": manifest}
        result = cls(
            schema_version=S2_LITE_TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
            producer_pass="s2_lite_train_manifest_linker",
            id=stable_artifact_id(
                "s2_lite_train_linked_program",
                semantic_key,
                schema_version=S2_LITE_TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {"source": self.source, "manifest": self.manifest}

    def validate(self, path: str = "s2_lite_train_linked_program") -> None:
        if self.schema_version != S2_LITE_TRAIN_LINKED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "s2_lite_train_manifest_linker":
            raise SchemaError(
                "must be 's2_lite_train_manifest_linker'",
                path=f"{path}.producer_pass",
            )
        if type(self.source) is not S2LiteTrainLoweredProgram:
            raise SchemaError(
                "must embed one exact S2LiteTrainLoweredProgram",
                path=f"{path}.source",
            )
        self.source.validate(f"{path}.source")
        if type(self.manifest) is not LinkedProgramManifest:
            raise SchemaError(
                "must embed one LinkedProgramManifest",
                path=f"{path}.manifest",
            )
        context = self.source.lowering_context
        self.manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            self.source.fragments,
            f"{path}.manifest",
        )
        from ..lowering.linker import NaiveManifestLinker

        expected = NaiveManifestLinker().link(
            context,
            self.source.fragments,
        )
        if self.manifest != expected:
            raise SchemaError(
                "manifest must equal the production Lite link quotient",
                path=f"{path}.manifest",
            )
        expected_id = stable_artifact_id(
            "s2_lite_train_linked_program",
            self._semantic_key(),
            schema_version=S2_LITE_TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: S2LiteTrainLoweredProgram,
        path: str = "s2_lite_train_linked_program",
    ) -> None:
        self.validate(path)
        if self.source != source:
            raise SchemaError(
                "must embed the exact Lite lowering source",
                path=f"{path}.source",
            )


__all__ = [
    "S2_LITE_TRAIN_LINKED_PROGRAM_SCHEMA_VERSION",
    "S2_LITE_TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION",
    "S2LiteTrainLinkedProgram",
    "S2LiteTrainLoweredProgram",
    "s2_lite_train_lowering_context",
]
