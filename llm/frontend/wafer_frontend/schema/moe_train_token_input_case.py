"""One immutable full-V MoE TRAIN vocabulary input bound to frozen routing.

WorkloadRunRequest does not carry token IDs: this separate workload *input*
artifact is explicitly selected for a canary and signed by original case,
model, TRAIN steps and four source route traces.  ProgramIO must derive its
INT32 SRAM bytes from this sole input, never from a separately typed trace.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id
from .serde import canonical_digest
from .workload_materialization import WorkloadMaterializationManifest
from .workload_run import WorkloadFamily


_SCHEMA = "wafer_frontend.moe_train_token_input_case/v1alpha1"


@dataclass(frozen=True, slots=True)
class MoeTrainTokenInputCase:
    schema_version: str
    id: str
    source_case_id: str
    source_request_digest: str
    source_logical_graph_digest: str
    route_trace_ids: tuple[str, ...]
    route_trace_digests: tuple[str, ...]
    sequence_length: int
    vocabulary_size: int
    tokens_by_step: tuple[tuple[int, ...], tuple[int, ...]]

    @classmethod
    def create(cls, manifest: WorkloadMaterializationManifest, *,
               tokens_by_step: tuple[tuple[int, ...], tuple[int, ...]]):
        manifest.validate("moe_token_input_source")
        if (manifest.request.family is not WorkloadFamily.MOE_TRAINING
                or manifest.request.steps.training is None):
            raise SchemaError("full-V input requires an original MoE TRAIN case",
                              path="moe_train_token_input_case.source")
        traces = manifest.logical_graph.route_traces
        semantic = dict(
            source_case_id=manifest.request.case_id,
            source_request_digest=canonical_digest(manifest.request),
            source_logical_graph_digest=manifest.logical_graph_digest,
            route_trace_ids=tuple(item.id for item in traces),
            route_trace_digests=tuple(canonical_digest(item) for item in traces),
            sequence_length=manifest.request.steps.training.sequence_length,
            vocabulary_size=manifest.request.model.vocabulary_size,
            tokens_by_step=tokens_by_step,
        )
        result = cls(_SCHEMA, stable_artifact_id(
            "moe_train_token_input_case", semantic, schema_version=_SCHEMA,
        ), **semantic)
        result.validate_against(manifest)
        return result

    def validate_against(self, manifest: WorkloadMaterializationManifest) -> None:
        manifest.validate("moe_token_input_source")
        req = manifest.request
        train = req.steps.training
        traces = manifest.logical_graph.route_traces
        if (self.schema_version != _SCHEMA or req.family is not
            WorkloadFamily.MOE_TRAINING or train is None
                or train.step_count != 2 or len(self.tokens_by_step) != 2
                or self.source_case_id != req.case_id
                or self.source_request_digest != canonical_digest(req)
                or self.source_logical_graph_digest !=
                    manifest.logical_graph_digest
                or self.route_trace_ids != tuple(item.id for item in traces)
                or self.route_trace_digests != tuple(canonical_digest(item)
                                                      for item in traces)
                or len(traces) != 2 * req.model.num_layers
                or self.sequence_length != train.sequence_length
                or self.vocabulary_size != req.model.vocabulary_size
                or any(trace.token_count != self.sequence_length for trace
                       in traces)):
            raise SchemaError("token workload input loses original MoE request/frozen routing source",
                              path="moe_train_token_input_case.source")
        for step, token_ids in enumerate(self.tokens_by_step):
            if (type(token_ids) is not tuple
                    or len(token_ids) != self.sequence_length
                    or not all(type(value) is int and 0 <= value <
                               self.vocabulary_size for value in token_ids)
                    or not any(value != 0 for value in token_ids)
                    or len(set(token_ids)) == len(token_ids)):
                raise SchemaError("nonzero scatter canary needs legal full-V token IDs and a repeat",
                                  path=f"moe_train_token_input_case.step{step}")
        semantic = dict(
            source_case_id=self.source_case_id,
            source_request_digest=self.source_request_digest,
            source_logical_graph_digest=self.source_logical_graph_digest,
            route_trace_ids=self.route_trace_ids,
            route_trace_digests=self.route_trace_digests,
            sequence_length=self.sequence_length,
            vocabulary_size=self.vocabulary_size,
            tokens_by_step=self.tokens_by_step,
        )
        if self.id != stable_artifact_id(
            "moe_train_token_input_case", semantic, schema_version=_SCHEMA,
        ):
            raise SchemaError("source token input content and artifact ID drifted",
                              path="moe_train_token_input_case.id")


__all__ = ["MoeTrainTokenInputCase"]
