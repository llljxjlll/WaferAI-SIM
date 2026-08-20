"""Independent analytic oracle for Stage 3 static Dense inference profiles."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import InferOutput
from .logical import IR0Template
from .stage2_dense_forward_oracle import (
    DenseCollectiveMetrics,
    DenseForwardWorkMetrics,
    DenseGraphMetrics,
    DenseParameterMetrics,
)
from .stage3_profile import Stage3StaticProfile


STAGE3_DENSE_INFERENCE_ORACLE_SCHEMA_VERSION = (
    "wafer_frontend.stage3_dense_inference_oracle/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class Stage3KvMetrics:
    page_count: int
    page_size_tokens: int
    logical_read_bytes: int
    logical_write_bytes: int
    rank_read_bytes: int
    rank_write_bytes: int
    logical_reserved_bytes: int
    rank_reserved_bytes: int

    def validate(self, path: str = "kv") -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.page_count == 0 or self.page_size_tokens == 0:
            raise SchemaError("page geometry must be non-zero", path=path)
        if self.logical_write_bytes == 0 or self.logical_reserved_bytes == 0:
            raise SchemaError("KV write/reserved bytes must be non-zero", path=path)
        if self.logical_reserved_bytes < self.logical_write_bytes:
            raise SchemaError(
                "reserved bytes must cover write bytes",
                path=f"{path}.logical_reserved_bytes",
            )


@dataclass(frozen=True, slots=True)
class Stage3DenseInferenceOracle:
    schema_version: str
    producer_pass: str
    id: str
    source_template_id: str
    static_profile: Stage3StaticProfile
    tp_degree: int
    infer_output: InferOutput
    parameters: DenseParameterMetrics
    graph: DenseGraphMetrics
    logical_work: DenseForwardWorkMetrics
    rank_work: DenseForwardWorkMetrics
    collectives: DenseCollectiveMetrics
    kv: Stage3KvMetrics

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage3DenseInferenceOracle":
        result = cls(
            schema_version=STAGE3_DENSE_INFERENCE_ORACLE_SCHEMA_VERSION,
            producer_pass="stage3_dense_inference_oracle",
            id=stable_artifact_id(
                "stage3_dense_inference_oracle",
                semantic_key,
                schema_version=STAGE3_DENSE_INFERENCE_ORACLE_SCHEMA_VERSION,
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

    def validate(self, path: str = "stage3_dense_inference_oracle") -> None:
        if self.schema_version != STAGE3_DENSE_INFERENCE_ORACLE_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "stage3_dense_inference_oracle":
            raise SchemaError(
                "must be 'stage3_dense_inference_oracle'",
                path=f"{path}.producer_pass",
            )
        validate_nonempty(self.source_template_id, f"{path}.source_template_id")
        if type(self.static_profile) is not Stage3StaticProfile:
            raise SchemaError(
                "must be a Stage3StaticProfile", path=f"{path}.static_profile"
            )
        self.static_profile.validate(f"{path}.static_profile")
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
            ("kv", Stage3KvMetrics),
        ):
            value = getattr(self, field_name)
            if type(value) is not expected_type:
                raise SchemaError(
                    f"must be a {expected_type.__name__}",
                    path=f"{path}.{field_name}",
                )
            value.validate(f"{path}.{field_name}")
        if self.kv.rank_read_bytes * self.tp_degree != self.kv.logical_read_bytes:
            raise SchemaError(
                "rank read bytes must exactly partition logical read bytes",
                path=f"{path}.kv.rank_read_bytes",
            )
        if self.kv.rank_write_bytes * self.tp_degree != self.kv.logical_write_bytes:
            raise SchemaError(
                "rank write bytes must exactly partition logical write bytes",
                path=f"{path}.kv.rank_write_bytes",
            )
        if (
            self.kv.rank_reserved_bytes * self.tp_degree
            != self.kv.logical_reserved_bytes
        ):
            raise SchemaError(
                "rank reserved bytes must exactly partition logical reserved bytes",
                path=f"{path}.kv.rank_reserved_bytes",
            )
        expected_id = stable_artifact_id(
            "stage3_dense_inference_oracle",
            self._semantic_key(),
            schema_version=STAGE3_DENSE_INFERENCE_ORACLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )

    def validate_against_template(
        self,
        template: IR0Template,
        *,
        path: str = "stage3_dense_inference_oracle",
    ) -> None:
        from ..passes.stage3_dense_inference_oracle import (
            build_stage3_dense_inference_oracle,
        )

        self.validate(path)
        expected = build_stage3_dense_inference_oracle(
            template,
            self.static_profile,
            tp_degree=self.tp_degree,
        )
        for field_name in self.__dataclass_fields__:
            if getattr(self, field_name) != getattr(expected, field_name):
                raise SchemaError(
                    "does not match the source template/profile",
                    path=f"{path}.{field_name}",
                )


__all__ = [
    "STAGE3_DENSE_INFERENCE_ORACLE_SCHEMA_VERSION",
    "Stage3KvMetrics",
    "Stage3DenseInferenceOracle",
]
