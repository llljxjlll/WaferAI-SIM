"""Independent analytic oracle for the Stage 2 Dense forward graph."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import ProfileKey, stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import InferOutput
from .logical import IR0Template


STAGE2_DENSE_FORWARD_ORACLE_SCHEMA_VERSION = (
    "wafer_frontend.stage2_dense_forward_oracle/v1alpha1"
)


def _validate_uint_fields(value: object, path: str) -> None:
    for field_name in value.__dataclass_fields__:  # type: ignore[attr-defined]
        validate_uint64(getattr(value, field_name), f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class DenseParameterMetrics:
    unique_tensor_count: int
    unique_elements: int
    unique_bytes: int
    placed_elements: int
    placed_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseGraphMetrics:
    node_count: int
    collective_node_count: int
    parameter_declaration_count: int
    kv_declaration_count: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseGemmOpMetrics:
    flops: int
    memory_read_bytes: int
    memory_write_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseGemmMetrics:
    qkv: DenseGemmOpMetrics
    attention_output: DenseGemmOpMetrics
    gate_up: DenseGemmOpMetrics
    down: DenseGemmOpMetrics
    lm_head: DenseGemmOpMetrics

    def validate(self, path: str) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if type(value) is not DenseGemmOpMetrics:
                raise SchemaError(
                    "must be a DenseGemmOpMetrics", path=f"{path}.{field_name}"
                )
            value.validate(f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class DenseEmbeddingMetrics:
    rows: int
    vector_ops: int
    sfu_ops: int
    memory_read_bytes: int
    memory_write_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseRmsNormMetrics:
    rows: int
    vector_ops: int
    sfu_ops: int
    memory_read_bytes: int
    memory_write_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseRopeQkMetrics:
    rotated_elements: int
    vector_ops: int
    sfu_ops: int
    memory_read_bytes: int
    memory_write_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseAttentionMetrics:
    query_key_pairs: int
    softmax_elements: int
    qk_matmul_flops: int
    value_matmul_flops: int
    vector_ops: int
    sfu_ops: int
    activation_memory_read_bytes: int
    activation_memory_write_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseSwiGluMetrics:
    output_elements: int
    vector_ops: int
    sfu_ops: int
    memory_read_bytes: int
    memory_write_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseResidualMetrics:
    output_elements: int
    vector_ops: int
    sfu_ops: int
    memory_read_bytes: int
    memory_write_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseGreedyMetrics:
    sample_count: int
    comparisons: int
    vector_ops: int
    sfu_ops: int
    memory_read_bytes: int
    memory_write_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseForwardWorkMetrics:
    gemm: DenseGemmMetrics
    embedding: DenseEmbeddingMetrics
    rms_norm: DenseRmsNormMetrics
    rope_qk: DenseRopeQkMetrics
    attention: DenseAttentionMetrics
    swiglu: DenseSwiGluMetrics
    residual: DenseResidualMetrics
    greedy: DenseGreedyMetrics

    def validate(self, path: str) -> None:
        expected = (
            ("gemm", DenseGemmMetrics),
            ("embedding", DenseEmbeddingMetrics),
            ("rms_norm", DenseRmsNormMetrics),
            ("rope_qk", DenseRopeQkMetrics),
            ("attention", DenseAttentionMetrics),
            ("swiglu", DenseSwiGluMetrics),
            ("residual", DenseResidualMetrics),
            ("greedy", DenseGreedyMetrics),
        )
        for field_name, expected_type in expected:
            value = getattr(self, field_name)
            if type(value) is not expected_type:
                raise SchemaError(
                    f"must be a {expected_type.__name__}",
                    path=f"{path}.{field_name}",
                )
            value.validate(f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class DenseCollectiveKindMetrics:
    node_count: int
    logical_tensor_bytes_per_node: int
    rank_payload_bytes_total: int
    group_payload_bytes_total: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class DenseCollectiveMetrics:
    all_gather: DenseCollectiveKindMetrics
    reduce_scatter: DenseCollectiveKindMetrics

    def validate(self, path: str) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if type(value) is not DenseCollectiveKindMetrics:
                raise SchemaError(
                    "must be a DenseCollectiveKindMetrics",
                    path=f"{path}.{field_name}",
                )
            value.validate(f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class DenseKvMetrics:
    logical_read_bytes: int
    logical_write_bytes: int
    rank_read_bytes: int
    rank_write_bytes: int

    def validate(self, path: str) -> None:
        _validate_uint_fields(self, path)


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardOracle:
    schema_version: str
    producer_pass: str
    id: str
    source_template_id: str
    profile: ProfileKey
    tp_degree: int
    infer_output: InferOutput
    parameters: DenseParameterMetrics
    graph: DenseGraphMetrics
    logical_work: DenseForwardWorkMetrics
    rank_work: DenseForwardWorkMetrics
    collectives: DenseCollectiveMetrics
    kv: DenseKvMetrics

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage2DenseForwardOracle":
        result = cls(
            schema_version=STAGE2_DENSE_FORWARD_ORACLE_SCHEMA_VERSION,
            producer_pass="stage2_dense_forward_oracle",
            id=stable_artifact_id(
                "stage2_dense_forward_oracle",
                semantic_key,
                schema_version=STAGE2_DENSE_FORWARD_ORACLE_SCHEMA_VERSION,
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

    def validate(self, path: str = "stage2_dense_forward_oracle") -> None:
        if self.schema_version != STAGE2_DENSE_FORWARD_ORACLE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "stage2_dense_forward_oracle":
            raise SchemaError(
                "must be 'stage2_dense_forward_oracle'",
                path=f"{path}.producer_pass",
            )
        validate_nonempty(self.source_template_id, f"{path}.source_template_id")
        if type(self.profile) is not ProfileKey:
            raise SchemaError("must be a ProfileKey", path=f"{path}.profile")
        self.profile.validate(f"{path}.profile")
        validate_uint64(self.tp_degree, f"{path}.tp_degree")
        if self.tp_degree == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.tp_degree")
        if type(self.infer_output) is not InferOutput:
            raise SchemaError("must be an InferOutput", path=f"{path}.infer_output")
        for field_name, expected_type in (
            ("parameters", DenseParameterMetrics),
            ("graph", DenseGraphMetrics),
            ("logical_work", DenseForwardWorkMetrics),
            ("rank_work", DenseForwardWorkMetrics),
            ("collectives", DenseCollectiveMetrics),
            ("kv", DenseKvMetrics),
        ):
            value = getattr(self, field_name)
            if type(value) is not expected_type:
                raise SchemaError(
                    f"must be a {expected_type.__name__}",
                    path=f"{path}.{field_name}",
                )
            value.validate(f"{path}.{field_name}")
        expected_id = stable_artifact_id(
            "stage2_dense_forward_oracle",
            self._semantic_key(),
            schema_version=STAGE2_DENSE_FORWARD_ORACLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against_template(
        self,
        template: IR0Template,
        *,
        path: str = "stage2_dense_forward_oracle",
    ) -> None:
        from ..passes.stage2_dense_forward_oracle import (
            build_stage2_dense_forward_oracle,
        )

        self.validate(path)
        expected = build_stage2_dense_forward_oracle(
            template, self.profile, tp_degree=self.tp_degree
        )
        for field_name in self.__dataclass_fields__:
            if getattr(self, field_name) != getattr(expected, field_name):
                raise SchemaError(
                    "does not match the source template",
                    path=f"{path}.{field_name}",
                )

    def validate_against_ir0(
        self,
        template: IR0Template,
        graph: object,
        *,
        path: str = "stage2_dense_forward_oracle",
    ) -> None:
        from ..passes.stage2_dense_forward_oracle import (
            validate_stage2_oracle_against_ir0,
        )

        validate_stage2_oracle_against_ir0(
            self, template, graph, path=path  # type: ignore[arg-type]
        )

    def validate_against_ir1(
        self,
        template: IR0Template,
        source: object,
        graph: object,
        *,
        path: str = "stage2_dense_forward_oracle",
    ) -> None:
        from ..passes.stage2_dense_forward_oracle import (
            validate_stage2_oracle_against_ir1,
        )

        validate_stage2_oracle_against_ir1(
            self,
            template,
            source,  # type: ignore[arg-type]
            graph,  # type: ignore[arg-type]
            path=path,
        )


__all__ = [
    "STAGE2_DENSE_FORWARD_ORACLE_SCHEMA_VERSION",
    "DenseAttentionMetrics",
    "DenseCollectiveKindMetrics",
    "DenseCollectiveMetrics",
    "DenseEmbeddingMetrics",
    "DenseForwardWorkMetrics",
    "DenseGemmMetrics",
    "DenseGemmOpMetrics",
    "DenseGraphMetrics",
    "DenseGreedyMetrics",
    "DenseKvMetrics",
    "DenseParameterMetrics",
    "DenseResidualMetrics",
    "DenseRmsNormMetrics",
    "DenseRopeQkMetrics",
    "DenseSwiGluMetrics",
    "Stage2DenseForwardOracle",
]
