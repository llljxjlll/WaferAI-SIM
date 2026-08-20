"""Typed one-manifest link entry for S2-Lite rooted all-reduce."""

from ..errors import SchemaError
from ..lowering.lite_train_rooted_ar_linker import link_s2_lite_rooted_ar_manifest
from ..schema.lite_train_rooted_ar_n6 import S2LiteRootedArLinkedProgram, S2LiteRootedArLoweredProgram


def link_s2_lite_rooted_ar(source: S2LiteRootedArLoweredProgram) -> S2LiteRootedArLinkedProgram:
    if type(source) is not S2LiteRootedArLoweredProgram:
        raise SchemaError("must be an S2LiteRootedArLoweredProgram", path="source")
    source.validate("source")
    return S2LiteRootedArLinkedProgram.create(source=source, manifest=link_s2_lite_rooted_ar_manifest(source))


__all__ = ["link_s2_lite_rooted_ar"]
