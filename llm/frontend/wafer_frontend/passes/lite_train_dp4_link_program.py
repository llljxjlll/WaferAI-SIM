"""Typed one-manifest link entry for S2-Lite DP4 tree AllReduce."""

from ..errors import SchemaError
from ..lowering.lite_train_dp4_linker import link_s2_lite_dp4_tree_ar_manifest
from ..schema.lite_train_dp4_n6 import (
    S2LiteDp4TreeArLinkedProgram,
    S2LiteDp4TreeArLoweredProgram,
)


def link_s2_lite_dp4_tree_ar(
    source: S2LiteDp4TreeArLoweredProgram,
) -> S2LiteDp4TreeArLinkedProgram:
    if type(source) is not S2LiteDp4TreeArLoweredProgram:
        raise SchemaError("must be an S2LiteDp4TreeArLoweredProgram", path="source")
    source.validate("source")
    return S2LiteDp4TreeArLinkedProgram.create(
        source=source,
        manifest=link_s2_lite_dp4_tree_ar_manifest(source),
    )


__all__ = ["link_s2_lite_dp4_tree_ar"]
