"""Typed contract and independent oracle for S2-Lite LM-head training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import TrainOptimizer


S2_LITE_LM_HEAD_TRAIN_CASE_ID = "case.s2_lite.lm_head_train"
S2_LITE_BASELINE_EPOCH = "s2-lite-v1"
S2_LITE_LM_HEAD_TRAIN_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_lm_head_train/v1alpha1"
)
S2_LITE_LM_HEAD_TRAIN_ORACLE_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_lm_head_train_oracle/v1alpha1"
)


class S2LiteTrainCoverage(str, Enum):
    LM_HEAD_ONLY = "lm_head_only"
    FULL_MODEL = "full_model"


class S2LiteTrainStage(str, Enum):
    CE_BACKWARD = "ce_backward"
    LM_HEAD_WGRAD = "lm_head_wgrad"
    SGD_UPDATE = "sgd_update"


_EXACT_STAGES = (
    S2LiteTrainStage.CE_BACKWARD,
    S2LiteTrainStage.LM_HEAD_WGRAD,
    S2LiteTrainStage.SGD_UPDATE,
)


def _validate_sha256(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a canonical lowercase SHA-256", path=path)


@dataclass(frozen=True, slots=True)
class S2LiteLmHeadTrainContract:
    """The only training coverage admitted by the two-day S2-Lite slice."""

    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    source_spec_digest: str
    coverage: S2LiteTrainCoverage
    backbone_frozen: bool
    embedding_frozen: bool
    optimizer: TrainOptimizer
    learning_rate: float
    momentum: float
    dp_degree: int
    tp_degree: int
    pp_degree: int
    ep_degree: int
    micro_batch_count: int
    step_count: int
    micro_batch_size: int
    sequence_length: int
    hidden_size: int
    vocabulary_size: int
    activation_dtype: DType
    label_dtype: DType
    loss_gradient_dtype: DType
    weight_dtype: DType
    weight_gradient_dtype: DType
    stages: tuple[S2LiteTrainStage, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "S2LiteLmHeadTrainContract":
        result = cls(
            schema_version=S2_LITE_LM_HEAD_TRAIN_SCHEMA_VERSION,
            producer_pass="s2_lite_lm_head_train_contract",
            id=stable_artifact_id(
                "s2_lite_lm_head_train",
                semantic_key,
                schema_version=S2_LITE_LM_HEAD_TRAIN_SCHEMA_VERSION,
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

    def validate(self, path: str = "s2_lite_lm_head_train") -> None:
        if self.schema_version != S2_LITE_LM_HEAD_TRAIN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_lm_head_train_contract":
            raise SchemaError(
                "must be 's2_lite_lm_head_train_contract'",
                path=f"{path}.producer_pass",
            )
        if self.case_id != S2_LITE_LM_HEAD_TRAIN_CASE_ID:
            raise SchemaError(
                f"must be {S2_LITE_LM_HEAD_TRAIN_CASE_ID!r}",
                path=f"{path}.case_id",
            )
        _validate_sha256(self.source_spec_digest, f"{path}.source_spec_digest")
        if type(self.coverage) is not S2LiteTrainCoverage:
            raise SchemaError(
                "must be an S2LiteTrainCoverage", path=f"{path}.coverage"
            )
        if self.coverage is not S2LiteTrainCoverage.LM_HEAD_ONLY:
            raise SchemaError(
                "only LM_HEAD_ONLY is supported", path=f"{path}.coverage"
            )
        for name in ("backbone_frozen", "embedding_frozen"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise SchemaError("must be a bool", path=f"{path}.{name}")
            if not value:
                raise SchemaError(
                    "must be true for LM_HEAD_ONLY", path=f"{path}.{name}"
                )
        if type(self.optimizer) is not TrainOptimizer:
            raise SchemaError("must be a TrainOptimizer", path=f"{path}.optimizer")
        if self.optimizer is not TrainOptimizer.SGD:
            raise SchemaError("only SGD is supported", path=f"{path}.optimizer")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise SchemaError(
                "must be a finite positive float", path=f"{path}.learning_rate"
            )
        if type(self.momentum) is not float or self.momentum != 0.0:
            raise SchemaError(
                "S2-Lite SGD requires momentum=0", path=f"{path}.momentum"
            )
        for name in (
            "dp_degree",
            "tp_degree",
            "pp_degree",
            "ep_degree",
            "micro_batch_count",
            "step_count",
            "micro_batch_size",
            "sequence_length",
            "hidden_size",
            "vocabulary_size",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        for name in (
            "dp_degree",
            "tp_degree",
            "pp_degree",
            "ep_degree",
            "micro_batch_count",
            "step_count",
        ):
            if getattr(self, name) != 1:
                raise SchemaError("must equal 1", path=f"{path}.{name}")
        for name in (
            "micro_batch_size",
            "sequence_length",
            "hidden_size",
            "vocabulary_size",
        ):
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")
        expected_dtypes = (
            ("activation_dtype", DType.FP16),
            ("label_dtype", DType.INT32),
            ("loss_gradient_dtype", DType.FP32),
            ("weight_dtype", DType.FP16),
            ("weight_gradient_dtype", DType.FP32),
        )
        for name, expected in expected_dtypes:
            value = getattr(self, name)
            if type(value) is not DType or value is not expected:
                raise SchemaError(
                    f"must be {expected.value}", path=f"{path}.{name}"
                )
        if self.stages != _EXACT_STAGES:
            raise SchemaError(
                "must be CE_BACKWARD -> LM_HEAD_WGRAD -> SGD_UPDATE",
                path=f"{path}.stages",
            )
        expected_id = stable_artifact_id(
            "s2_lite_lm_head_train",
            self._semantic_key(),
            schema_version=S2_LITE_LM_HEAD_TRAIN_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class S2LiteCeBackwardMetrics:
    element_count: int
    logits_read_bytes: int
    labels_read_bytes: int
    loss_gradient_read_bytes: int
    logits_gradient_write_bytes: int

    def validate(self, path: str) -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class S2LiteLmHeadWgradMetrics:
    weight_element_count: int
    hidden_read_bytes: int
    logits_gradient_read_bytes: int
    weight_gradient_write_bytes: int
    floating_point_ops: int

    def validate(self, path: str) -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class S2LiteSgdMetrics:
    element_count: int
    weight_read_bytes: int
    gradient_read_bytes: int
    updated_weight_write_bytes: int
    floating_point_ops: int

    def validate(self, path: str) -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class S2LiteLmHeadTrainOracle:
    """Analytic work/byte oracle derived before a training DAG exists."""

    schema_version: str
    producer_pass: str
    id: str
    source_contract_id: str
    source_spec_digest: str
    case_id: str
    stages: tuple[S2LiteTrainStage, ...]
    dependency_count: int
    logical_rows: int
    hidden_elements: int
    logits_elements: int
    lm_head_weight_elements: int
    lm_head_weight_bytes: int
    lm_head_weight_gradient_bytes: int
    ce_backward: S2LiteCeBackwardMetrics
    lm_head_wgrad: S2LiteLmHeadWgradMetrics
    sgd_update: S2LiteSgdMetrics
    total_read_bytes: int
    total_write_bytes: int
    total_floating_point_ops: int

    @classmethod
    def create(cls, **semantic_key: object) -> "S2LiteLmHeadTrainOracle":
        result = cls(
            schema_version=S2_LITE_LM_HEAD_TRAIN_ORACLE_SCHEMA_VERSION,
            producer_pass="s2_lite_lm_head_train_oracle",
            id=stable_artifact_id(
                "s2_lite_lm_head_train_oracle",
                semantic_key,
                schema_version=S2_LITE_LM_HEAD_TRAIN_ORACLE_SCHEMA_VERSION,
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

    def validate(self, path: str = "s2_lite_lm_head_train_oracle") -> None:
        if self.schema_version != S2_LITE_LM_HEAD_TRAIN_ORACLE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_lm_head_train_oracle":
            raise SchemaError(
                "must be 's2_lite_lm_head_train_oracle'",
                path=f"{path}.producer_pass",
            )
        validate_nonempty(self.source_contract_id, f"{path}.source_contract_id")
        _validate_sha256(self.source_spec_digest, f"{path}.source_spec_digest")
        if self.case_id != S2_LITE_LM_HEAD_TRAIN_CASE_ID:
            raise SchemaError(
                f"must be {S2_LITE_LM_HEAD_TRAIN_CASE_ID!r}",
                path=f"{path}.case_id",
            )
        if self.stages != _EXACT_STAGES:
            raise SchemaError("stage order is not exact", path=f"{path}.stages")
        for name in (
            "dependency_count",
            "logical_rows",
            "hidden_elements",
            "logits_elements",
            "lm_head_weight_elements",
            "lm_head_weight_bytes",
            "lm_head_weight_gradient_bytes",
            "total_read_bytes",
            "total_write_bytes",
            "total_floating_point_ops",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.dependency_count != len(self.stages) - 1:
            raise SchemaError(
                "must encode the exact linear stage dependencies",
                path=f"{path}.dependency_count",
            )
        if (
            self.logical_rows == 0
            or self.hidden_elements == 0
            or self.logits_elements == 0
            or self.lm_head_weight_elements == 0
            or self.hidden_elements % self.logical_rows != 0
            or self.logits_elements % self.logical_rows != 0
        ):
            raise SchemaError(
                "logical tensor geometry must be positive and row-aligned",
                path=path,
            )
        hidden_size = self.hidden_elements // self.logical_rows
        vocabulary_size = self.logits_elements // self.logical_rows
        if self.lm_head_weight_elements != hidden_size * vocabulary_size:
            raise SchemaError(
                "LM-head weight geometry is not exact",
                path=f"{path}.lm_head_weight_elements",
            )
        if (
            self.lm_head_weight_bytes != 2 * self.lm_head_weight_elements
            or self.lm_head_weight_gradient_bytes
            != 4 * self.lm_head_weight_elements
        ):
            raise SchemaError(
                "LM-head FP16 weight/FP32 gradient bytes are not exact",
                path=f"{path}.lm_head_weight_bytes",
            )
        for name, expected_type in (
            ("ce_backward", S2LiteCeBackwardMetrics),
            ("lm_head_wgrad", S2LiteLmHeadWgradMetrics),
            ("sgd_update", S2LiteSgdMetrics),
        ):
            value = getattr(self, name)
            if type(value) is not expected_type:
                raise SchemaError(
                    f"must be a {expected_type.__name__}", path=f"{path}.{name}"
                )
            value.validate(f"{path}.{name}")
        if (
            self.ce_backward.element_count != self.logits_elements
            or self.ce_backward.logits_read_bytes != 2 * self.logits_elements
            or self.ce_backward.labels_read_bytes != 4 * self.logical_rows
            or self.ce_backward.loss_gradient_read_bytes
            != 4 * self.logical_rows
            or self.ce_backward.logits_gradient_write_bytes
            != 2 * self.logits_elements
        ):
            raise SchemaError(
                "CE backward tensor bytes are not exact",
                path=f"{path}.ce_backward",
            )
        if (
            self.lm_head_wgrad.weight_element_count
            != self.lm_head_weight_elements
            or self.lm_head_wgrad.hidden_read_bytes
            != 2 * self.hidden_elements
            or self.lm_head_wgrad.logits_gradient_read_bytes
            != 2 * self.logits_elements
            or self.lm_head_wgrad.weight_gradient_write_bytes
            != self.lm_head_weight_gradient_bytes
            or self.lm_head_wgrad.floating_point_ops
            != 2 * self.logical_rows * hidden_size * vocabulary_size
        ):
            raise SchemaError(
                "LM-head WGRAD work/bytes are not exact",
                path=f"{path}.lm_head_wgrad",
            )
        if self.sgd_update.floating_point_ops != 2 * self.sgd_update.element_count:
            raise SchemaError(
                "SGD must perform multiply-and-subtract per element",
                path=f"{path}.sgd_update.floating_point_ops",
            )
        expected_reads = (
            self.ce_backward.logits_read_bytes
            + self.ce_backward.labels_read_bytes
            + self.ce_backward.loss_gradient_read_bytes
            + self.lm_head_wgrad.hidden_read_bytes
            + self.lm_head_wgrad.logits_gradient_read_bytes
            + self.sgd_update.weight_read_bytes
            + self.sgd_update.gradient_read_bytes
        )
        if self.total_read_bytes != expected_reads:
            raise SchemaError("read-byte total is not exact", path=f"{path}.total_read_bytes")
        expected_writes = (
            self.ce_backward.logits_gradient_write_bytes
            + self.lm_head_wgrad.weight_gradient_write_bytes
            + self.sgd_update.updated_weight_write_bytes
        )
        if self.total_write_bytes != expected_writes:
            raise SchemaError("write-byte total is not exact", path=f"{path}.total_write_bytes")
        expected_flops = (
            self.lm_head_wgrad.floating_point_ops
            + self.sgd_update.floating_point_ops
        )
        if self.total_floating_point_ops != expected_flops:
            raise SchemaError("FLOP total is not exact", path=f"{path}.total_floating_point_ops")
        if (
            self.sgd_update.element_count != self.lm_head_weight_elements
            or self.sgd_update.weight_read_bytes != self.lm_head_weight_bytes
            or self.sgd_update.gradient_read_bytes
            != self.lm_head_weight_gradient_bytes
            or self.sgd_update.updated_weight_write_bytes
            != self.lm_head_weight_bytes
        ):
            raise SchemaError(
                "SGD weight/gradient footprint is not exact", path=f"{path}.sgd_update"
            )
        expected_id = stable_artifact_id(
            "s2_lite_lm_head_train_oracle",
            self._semantic_key(),
            schema_version=S2_LITE_LM_HEAD_TRAIN_ORACLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against_contract(
        self,
        contract: S2LiteLmHeadTrainContract,
        *,
        path: str = "s2_lite_lm_head_train_oracle",
    ) -> None:
        from ..passes.lite_train import build_s2_lite_lm_head_train_oracle

        expected = build_s2_lite_lm_head_train_oracle(contract)
        if self != expected:
            raise SchemaError("does not exactly match the source contract", path=path)


__all__ = [
    "S2_LITE_BASELINE_EPOCH",
    "S2_LITE_LM_HEAD_TRAIN_CASE_ID",
    "S2_LITE_LM_HEAD_TRAIN_ORACLE_SCHEMA_VERSION",
    "S2_LITE_LM_HEAD_TRAIN_SCHEMA_VERSION",
    "S2LiteCeBackwardMetrics",
    "S2LiteLmHeadTrainContract",
    "S2LiteLmHeadTrainOracle",
    "S2LiteLmHeadWgradMetrics",
    "S2LiteSgdMetrics",
    "S2LiteTrainCoverage",
    "S2LiteTrainStage",
]
