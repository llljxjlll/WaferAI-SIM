"""Lossless carrier for the S2-Lite LM-head-only training IR-0."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id
from .ir0 import IR0, OpKind, OpPhase
from .lite_train import S2LiteLmHeadTrainContract, S2LiteLmHeadTrainOracle


S2_LITE_LM_HEAD_TRAIN_IR0_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_lm_head_train_ir0/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class S2LiteLmHeadTrainIR0:
    schema_version: str
    producer_pass: str
    id: str
    base_graph: IR0
    contract: S2LiteLmHeadTrainContract
    oracle: S2LiteLmHeadTrainOracle
    graph: IR0

    @classmethod
    def create(
        cls,
        *,
        base_graph: IR0,
        contract: S2LiteLmHeadTrainContract,
        oracle: S2LiteLmHeadTrainOracle,
        graph: IR0,
    ) -> "S2LiteLmHeadTrainIR0":
        semantic_key = {
            "base_graph": base_graph,
            "contract": contract,
            "oracle": oracle,
            "graph": graph,
        }
        result = cls(
            schema_version=S2_LITE_LM_HEAD_TRAIN_IR0_SCHEMA_VERSION,
            producer_pass="s2_lite_lm_head_train_ir0",
            id=stable_artifact_id(
                "s2_lite_lm_head_train_ir0",
                semantic_key,
                schema_version=S2_LITE_LM_HEAD_TRAIN_IR0_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "base_graph": self.base_graph,
            "contract": self.contract,
            "oracle": self.oracle,
            "graph": self.graph,
        }

    def validate(self, path: str = "s2_lite_lm_head_train_ir0") -> None:
        if self.schema_version != S2_LITE_LM_HEAD_TRAIN_IR0_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_lm_head_train_ir0":
            raise SchemaError(
                "must be 's2_lite_lm_head_train_ir0'",
                path=f"{path}.producer_pass",
            )
        if type(self.base_graph) is not IR0:
            raise SchemaError("must be an IR0", path=f"{path}.base_graph")
        if type(self.contract) is not S2LiteLmHeadTrainContract:
            raise SchemaError(
                "must be an S2LiteLmHeadTrainContract", path=f"{path}.contract"
            )
        if type(self.oracle) is not S2LiteLmHeadTrainOracle:
            raise SchemaError(
                "must be an S2LiteLmHeadTrainOracle", path=f"{path}.oracle"
            )
        if type(self.graph) is not IR0:
            raise SchemaError("must be an IR0", path=f"{path}.graph")
        from ..passes.validate_ir0 import DenseIR0Validator

        DenseIR0Validator.validate(self.base_graph, f"{path}.base_graph")
        self.contract.validate(f"{path}.contract")
        self.oracle.validate_against_contract(
            self.contract, path=f"{path}.oracle"
        )
        DenseIR0Validator.validate(self.graph, f"{path}.graph")
        if (
            self.base_graph.producer_pass != "train_forward_expand"
            or any(node.phase is not OpPhase.FWD for node in self.base_graph.nodes)
            or any(
                node.kind in (OpKind.CE_BACKWARD, OpKind.OPTIMIZER_UPDATE)
                for node in self.base_graph.nodes
            )
        ):
            raise SchemaError(
                "base graph must be the exact forward-only train graph",
                path=f"{path}.base_graph",
            )
        from ..passes.lite_train_graph import _append_s2_lite_lm_head_train

        expected_graph = _append_s2_lite_lm_head_train(
            self.base_graph, self.contract, self.oracle
        )
        if self.graph != expected_graph:
            raise SchemaError(
                "graph does not exactly derive from base/contract/oracle",
                path=f"{path}.graph",
            )
        expected_id = stable_artifact_id(
            "s2_lite_lm_head_train_ir0",
            self._semantic_key(),
            schema_version=S2_LITE_LM_HEAD_TRAIN_IR0_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


__all__ = [
    "S2_LITE_LM_HEAD_TRAIN_IR0_SCHEMA_VERSION",
    "S2LiteLmHeadTrainIR0",
]
