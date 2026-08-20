"""Independent analytic contract for the N6.1 forward-only train graph."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import TrainStructure


TRAIN_FORWARD_ORACLE_SCHEMA_VERSION = (
    "wafer_frontend.train_forward_oracle/v1alpha1"
)


def _uint_fields(value: object, path: str) -> None:
    for name in value.__dataclass_fields__:  # type: ignore[attr-defined]
        validate_uint64(getattr(value, name), f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class TrainForwardParameterMetrics:
    unique_tensor_count: int
    unique_elements: int
    unique_bytes: int
    tp_placed_elements: int
    tp_placed_bytes: int
    dp_replicated_bytes: int

    def validate(self, path: str) -> None:
        _uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class TrainForwardGraphMetrics:
    node_count: int
    value_count: int
    edge_count: int
    fusion_candidate_count: int
    collective_node_count: int
    parameter_declaration_count: int
    kv_declaration_count: int

    def validate(self, path: str) -> None:
        _uint_fields(self, path)
        if self.kv_declaration_count != 0:
            raise SchemaError(
                "forward-only train must not declare KV state",
                path=f"{path}.kv_declaration_count",
            )


@dataclass(frozen=True, slots=True)
class TrainForwardCollectiveMetrics:
    node_count: int
    logical_tensor_bytes_per_node: int
    group_payload_bytes_per_microbatch: int
    cluster_step_group_payload_bytes: int

    def validate(self, path: str) -> None:
        _uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class TrainForwardCeMetrics:
    logical_rows: int
    rank_rows: int
    label_dtype: DType
    loss_dtype: DType
    logical_label_bytes: int
    rank_label_bytes: int
    logical_loss_bytes: int
    rank_loss_bytes: int

    def validate(self, path: str) -> None:
        for name in (
            "logical_rows",
            "rank_rows",
            "logical_label_bytes",
            "rank_label_bytes",
            "logical_loss_bytes",
            "rank_loss_bytes",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.logical_rows == 0 or self.rank_rows == 0:
            raise SchemaError("row counts must be positive", path=path)
        if self.label_dtype is not DType.INT32:
            raise SchemaError("labels must be INT32", path=f"{path}.label_dtype")
        if self.loss_dtype is not DType.FP32:
            raise SchemaError("loss must be FP32", path=f"{path}.loss_dtype")
        if self.logical_label_bytes != 4 * self.logical_rows:
            raise SchemaError("must equal 4 * logical_rows", path=f"{path}.logical_label_bytes")
        if self.rank_label_bytes != 4 * self.rank_rows:
            raise SchemaError("must equal 4 * rank_rows", path=f"{path}.rank_label_bytes")
        if self.logical_loss_bytes != 4 * self.logical_rows:
            raise SchemaError("must equal 4 * logical_rows", path=f"{path}.logical_loss_bytes")
        if self.rank_loss_bytes != 4 * self.rank_rows:
            raise SchemaError("must equal 4 * rank_rows", path=f"{path}.rank_loss_bytes")


@dataclass(frozen=True, slots=True)
class TrainForwardOracle:
    schema_version: str
    producer_pass: str
    id: str
    source_spec_digest: str
    train_structure: TrainStructure
    tp_degree: int
    dp_degree: int
    sequence_parallel: bool
    tokens_per_microbatch: int
    parameters: TrainForwardParameterMetrics
    graph: TrainForwardGraphMetrics
    gemm_flops_per_microbatch: int
    attention_query_key_pairs_per_microbatch: int
    attention_flops_per_microbatch: int
    logical_forward_flops_per_microbatch: int
    rank_forward_flops_per_microbatch: int
    cluster_forward_flops_per_step: int
    collectives: TrainForwardCollectiveMetrics
    ce: TrainForwardCeMetrics

    @classmethod
    def create(cls, **semantic_key: object) -> "TrainForwardOracle":
        result = cls(
            schema_version=TRAIN_FORWARD_ORACLE_SCHEMA_VERSION,
            producer_pass="train_forward_oracle",
            id=stable_artifact_id(
                "train_forward_oracle",
                semantic_key,
                schema_version=TRAIN_FORWARD_ORACLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "train_forward_oracle") -> None:
        if self.schema_version != TRAIN_FORWARD_ORACLE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "train_forward_oracle":
            raise SchemaError("must be 'train_forward_oracle'", path=f"{path}.producer_pass")
        validate_nonempty(self.source_spec_digest, f"{path}.source_spec_digest")
        if type(self.train_structure) is not TrainStructure:
            raise SchemaError("must be a TrainStructure", path=f"{path}.train_structure")
        self.train_structure.validate(f"{path}.train_structure")
        for name in (
            "tp_degree",
            "dp_degree",
            "tokens_per_microbatch",
            "gemm_flops_per_microbatch",
            "attention_query_key_pairs_per_microbatch",
            "attention_flops_per_microbatch",
            "logical_forward_flops_per_microbatch",
            "rank_forward_flops_per_microbatch",
            "cluster_forward_flops_per_step",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.tp_degree == 0 or self.dp_degree == 0 or self.tokens_per_microbatch == 0:
            raise SchemaError("degrees and token count must be positive", path=path)
        if type(self.sequence_parallel) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.sequence_parallel")
        for name, expected in (
            ("parameters", TrainForwardParameterMetrics),
            ("graph", TrainForwardGraphMetrics),
            ("collectives", TrainForwardCollectiveMetrics),
            ("ce", TrainForwardCeMetrics),
        ):
            value = getattr(self, name)
            if type(value) is not expected:
                raise SchemaError(f"must be a {expected.__name__}", path=f"{path}.{name}")
            value.validate(f"{path}.{name}")
        if self.parameters.dp_replicated_bytes != self.parameters.tp_placed_bytes * self.dp_degree:
            raise SchemaError("must equal TP placed bytes * dp", path=f"{path}.parameters.dp_replicated_bytes")
        if self.logical_forward_flops_per_microbatch != self.gemm_flops_per_microbatch + self.attention_flops_per_microbatch:
            raise SchemaError("must equal GEMM + attention FLOPs", path=f"{path}.logical_forward_flops_per_microbatch")
        if self.rank_forward_flops_per_microbatch * self.tp_degree != self.logical_forward_flops_per_microbatch:
            raise SchemaError("rank FLOPs must quotient logical FLOPs by TP", path=f"{path}.rank_forward_flops_per_microbatch")
        if self.cluster_forward_flops_per_step != (
            self.logical_forward_flops_per_microbatch
            * self.dp_degree
            * self.train_structure.micro_batch_count
        ):
            raise SchemaError("cluster step FLOPs are not exact", path=f"{path}.cluster_forward_flops_per_step")
        if self.collectives.cluster_step_group_payload_bytes != (
            self.collectives.group_payload_bytes_per_microbatch
            * self.dp_degree
            * self.train_structure.micro_batch_count
        ):
            raise SchemaError("cluster collective bytes are not exact", path=f"{path}.collectives.cluster_step_group_payload_bytes")
        expected_id = stable_artifact_id(
            "train_forward_oracle",
            self._semantic_key(),
            schema_version=TRAIN_FORWARD_ORACLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against_spec(self, spec: object, *, path: str = "train_forward_oracle") -> None:
        from ..passes.train_forward_oracle import build_train_forward_oracle

        expected = build_train_forward_oracle(spec)  # type: ignore[arg-type]
        if self != expected:
            raise SchemaError("does not exactly match the source spec", path=path)

    def validate_against_ir0(self, spec: object, graph: object, *, path: str = "train_forward_oracle") -> None:
        from ..passes.train_forward_oracle import validate_train_forward_oracle_against_ir0

        validate_train_forward_oracle_against_ir0(self, spec, graph, path=path)  # type: ignore[arg-type]


__all__ = [
    "TRAIN_FORWARD_ORACLE_SCHEMA_VERSION",
    "TrainForwardCeMetrics",
    "TrainForwardCollectiveMetrics",
    "TrainForwardGraphMetrics",
    "TrainForwardOracle",
    "TrainForwardParameterMetrics",
]
