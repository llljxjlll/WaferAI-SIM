"""Identity implementation for the first intra-die graph-refine boundary."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.ir1 import IR1
from ..schema.ir2 import IR2ProjectionResult
from ..schema.intra_die_refine import (
    IntraDieOptimizationOptions,
    IntraDieRefineContext,
    IntraDieRefineContract,
    SplitKRefineOptions,
)

from ..schema.split_k_refine import SplitKRefinedProjection

IDENTITY_INTRADIE_REFINE_POLICY_SCHEMA_VERSION = (
    "wafer_frontend.identity_intra_die_refine_policy/v1"
)


class IdentityIntraDieRefinePolicy:
    """Return the exact projected IR2 graph without semantic rewriting."""

    schema_version = IDENTITY_INTRADIE_REFINE_POLICY_SCHEMA_VERSION

    def refine(self, projection: IR2ProjectionResult, ir1: IR1) -> IR2ProjectionResult:
        if type(projection) is not IR2ProjectionResult:
            raise SchemaError("must be an IR2ProjectionResult", path="projection")
        if type(ir1) is not IR1:
            raise SchemaError("must be an IR1 artifact", path="ir1")
        projection.validate("projection")
        if projection.source_ir1_id != ir1.id:
            raise SchemaError(
                "projection references a different IR-1",
                path="projection.source_ir1_id",
            )
        return projection
    def refine_graph(
        self, projection: IR2ProjectionResult, ir1: IR1,
        context: IntraDieRefineContext,
    ) -> SplitKRefinedProjection | None:
        """Return the selected v2 view while canonical projection stays exact."""
        if type(context) is not IntraDieRefineContext:
            raise SchemaError(
                "must be an IntraDieRefineContext",
                path="intra_die_refine_context",
            )
        self.refine(projection, ir1)
        context.validate("intra_die_refine_context")
        if context.contract is not IntraDieRefineContract.SPLIT_K_REDUCE_DOUBLE_BUFFER_V2:
            return None
        if (
            type(context.options) is SplitKRefineOptions
            and context.options.split_k_parts == 1
        ):
            return None
        from .split_k_intra_die_refine import refine_split_k_projection
        return refine_split_k_projection(projection, context.options, ir1)


__all__ = [
    "IDENTITY_INTRADIE_REFINE_POLICY_SCHEMA_VERSION",
    "IdentityIntraDieRefinePolicy",
]
