"""N6 manifest-link producer orchestration."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.interfaces import ManifestLinker
from ..schema.artifact_manifest import LinkedProgramManifest
from ..schema.n6 import (
    LinkedProgramBundle,
    LinkedProgramProfile,
    LoweredProgramBundle,
    LoweredProgramProfile,
    Stage4LinkedProgram,
    Stage4LoweredProgram,
)


def _resolve_linker(linker: ManifestLinker | None) -> ManifestLinker:
    if linker is None:
        from ..lowering.linker import NaiveManifestLinker

        return NaiveManifestLinker()
    return linker


def _link_profile(
    source: LoweredProgramProfile,
    linker: ManifestLinker,
) -> LinkedProgramProfile:
    source.validate("source")
    manifest = linker.link(source.lowering_context, source.fragments)
    if type(manifest) is not LinkedProgramManifest:
        raise SchemaError(
            "linker must return a LinkedProgramManifest",
            path="manifest",
        )
    result = LinkedProgramProfile.create(source=source, manifest=manifest)
    result.validate_against(source)
    return result


def link_profile(
    source: LoweredProgramProfile,
    linker: ManifestLinker | None = None,
) -> LinkedProgramProfile:
    """Link one exact canonical Lowered profile."""

    if type(source) is not LoweredProgramProfile:
        raise SchemaError(
            "must be a LoweredProgramProfile",
            path="source",
        )
    return _link_profile(source, _resolve_linker(linker))


def link_bundle(
    source: LoweredProgramBundle,
    linker: ManifestLinker | None = None,
) -> LinkedProgramBundle:
    """Link every Lowered profile exactly once in source tuple order."""

    if type(source) is not LoweredProgramBundle:
        raise SchemaError(
            "must be a LoweredProgramBundle",
            path="source",
        )
    source.validate("source")
    resolved = _resolve_linker(linker)
    entries = tuple(_link_profile(entry, resolved) for entry in source.entries)
    result = LinkedProgramBundle.create(source=source, entries=entries)
    result.validate_against(source)
    return result


def link_stage4(source: Stage4LoweredProgram) -> Stage4LinkedProgram:
    """Link one formal Stage 4 lowering through the production linker."""

    if type(source) is not Stage4LoweredProgram:
        raise SchemaError(
            "must be a Stage4LoweredProgram",
            path="source",
        )
    source.validate("source")
    manifest = _resolve_linker(None).link(
        source.lowering_context,
        source.fragments,
    )
    if type(manifest) is not LinkedProgramManifest:
        raise SchemaError(
            "linker must return a LinkedProgramManifest",
            path="manifest",
        )
    result = Stage4LinkedProgram.create(
        source=source,
        manifest=manifest,
    )
    result.validate_against(source)
    return result


__all__ = ["link_bundle", "link_profile", "link_stage4"]
