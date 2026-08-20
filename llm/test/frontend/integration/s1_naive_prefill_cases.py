"""Current-epoch production builders for the S1-N F-P1/F-P2 cases."""

from __future__ import annotations

from dataclasses import dataclass

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    REGION_MANIFEST_SCHEMA_VERSION,
    CommandFragment,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
)
from llm.frontend.wafer_frontend.schema.s1_naive_evidence import (
    S1_N_BASELINE_EPOCH,
    S1NaivePolicyEvidence,
    S1NaivePrefillCase,
)

from stage2_dense_forward_cases import (
    Stage2DenseForwardCase,
    build_stage2_dense_forward_case,
)


@dataclass(frozen=True, slots=True)
class S1NaivePrefillCurrentCase:
    baseline_epoch: str
    case: S1NaivePrefillCase
    tp_degree: int
    source: Stage2DenseForwardCase
    policy: S1NaivePolicyEvidence

    def validate(self, path: str = "s1_naive_prefill_current_case") -> None:
        if self.baseline_epoch != S1_N_BASELINE_EPOCH:
            raise SchemaError(
                f"must be {S1_N_BASELINE_EPOCH!r}",
                path=f"{path}.baseline_epoch",
            )
        if type(self.case) is not S1NaivePrefillCase:
            raise SchemaError(
                "must be a S1NaivePrefillCase", path=f"{path}.case"
            )
        expected_tp = {
            S1NaivePrefillCase.F_P1: 1,
            S1NaivePrefillCase.F_P2: 2,
        }[self.case]
        if self.tp_degree != expected_tp:
            raise SchemaError(
                f"must be exactly {expected_tp}", path=f"{path}.tp_degree"
            )
        if type(self.source) is not Stage2DenseForwardCase:
            raise SchemaError(
                "must be a Stage2DenseForwardCase", path=f"{path}.source"
            )
        source = self.source
        source.oracle.validate_against_template(source.template)
        if source.oracle.tp_degree != self.tp_degree:
            raise SchemaError(
                "oracle TP degree differs", path=f"{path}.source.oracle"
            )
        profile = source.oracle.profile
        if (
            profile.prefill_tokens != 8
            or profile.decode_tokens != 0
            or profile.num_seqs != 1
            or profile.context_sum != 8
            or profile.context_max != 8
            or profile.kv_pages != 1
        ):
            raise SchemaError(
                "must be the exact tiny pure-prefill profile",
                path=f"{path}.source.oracle.profile",
            )
        if (
            source.lowered_bundle.schema_version
            != LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION
            or source.linked_bundle.schema_version
            != LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION
            or source.manifest.schema_version
            != LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION
        ):
            raise SchemaError(
                "does not use the current N6 wrapper/manifest epoch", path=path
            )
        if (
            source.lowered_bundle.entries != (source.lowered,)
            or source.linked_bundle.entries != (source.profile,)
            or source.profile.manifest != source.manifest
        ):
            raise SchemaError(
                "must preserve one exact lower/link profile", path=path
            )
        for index, fragment in enumerate(source.lowered.fragments):
            leaf = fragment.fragment if type(fragment) is RegionManifest else fragment
            if type(leaf) is not CommandFragment:
                raise SchemaError(
                    "must contain only command/region leaves",
                    path=f"{path}.source.lowered.fragments[{index}]",
                )
            if leaf.schema_version != COMMAND_FRAGMENT_SCHEMA_VERSION:
                raise SchemaError(
                    "command leaf is not current epoch",
                    path=f"{path}.source.lowered.fragments[{index}]",
                )
            if (
                type(fragment) is RegionManifest
                and fragment.schema_version != REGION_MANIFEST_SCHEMA_VERSION
            ):
                raise SchemaError(
                    "region leaf is not current epoch",
                    path=f"{path}.source.lowered.fragments[{index}]",
                )
        if type(self.policy) is not S1NaivePolicyEvidence:
            raise SchemaError(
                "must be a S1NaivePolicyEvidence", path=f"{path}.policy"
            )
        self.policy.validate_against(
            source.planning_context,
            source.scheduling_context,
            f"{path}.policy",
        )
        if source.profile.scheduling_context_id != source.scheduling_context.id:
            raise SchemaError(
                "linked profile lost scheduling policy context identity",
                path=f"{path}.source.profile.scheduling_context_id",
            )
        collective_bytes = (
            source.oracle.collectives.all_gather.group_payload_bytes_total
            + source.oracle.collectives.reduce_scatter.group_payload_bytes_total
        )
        if (self.case is S1NaivePrefillCase.F_P1) is not (
            source.oracle.graph.collective_node_count == 0
            and collective_bytes == 0
        ):
            raise SchemaError(
                "collective analytic boundary differs from F-P1/F-P2",
                path=f"{path}.source.oracle.collectives",
            )


def build_s1_naive_prefill_case(
    case: S1NaivePrefillCase,
) -> S1NaivePrefillCurrentCase:
    if type(case) is not S1NaivePrefillCase:
        raise TypeError("case must be a S1NaivePrefillCase")
    tp_degree = {
        S1NaivePrefillCase.F_P1: 1,
        S1NaivePrefillCase.F_P2: 2,
    }[case]
    source = build_stage2_dense_forward_case(tp_degree)
    policy = S1NaivePolicyEvidence(
        selections=(
            source.planning_context.fused_policy,
            source.planning_context.standalone_policy,
            source.scheduling_context.policy,
        ),
        planning_context_id=source.planning_context.id,
        scheduling_context_id=source.scheduling_context.id,
    )
    result = S1NaivePrefillCurrentCase(
        baseline_epoch=S1_N_BASELINE_EPOCH,
        case=case,
        tp_degree=tp_degree,
        source=source,
        policy=policy,
    )
    result.validate()
    return result


__all__ = [
    "S1NaivePrefillCurrentCase",
    "build_s1_naive_prefill_case",
]
