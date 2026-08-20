from __future__ import annotations

from dataclasses import replace
import json
import math
import unittest

from llm.frontend.wafer_frontend.errors import FrontendError, SchemaError
from llm.frontend.wafer_frontend.passes.lite_train import (
    build_s2_lite_lm_head_train_oracle,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.experiment import TrainOptimizer
from llm.frontend.wafer_frontend.schema.lite_train import (
    S2_LITE_BASELINE_EPOCH,
    S2_LITE_LM_HEAD_TRAIN_CASE_ID,
    S2_LITE_LM_HEAD_TRAIN_ORACLE_SCHEMA_VERSION,
    S2_LITE_LM_HEAD_TRAIN_SCHEMA_VERSION,
    S2LiteLmHeadTrainContract,
    S2LiteLmHeadTrainOracle,
    S2LiteTrainCoverage,
    S2LiteTrainStage,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass


def _tiny_contract() -> S2LiteLmHeadTrainContract:
    return S2LiteLmHeadTrainContract.create(
        case_id=S2_LITE_LM_HEAD_TRAIN_CASE_ID,
        source_spec_digest="1" * 64,
        coverage=S2LiteTrainCoverage.LM_HEAD_ONLY,
        backbone_frozen=True,
        embedding_frozen=True,
        optimizer=TrainOptimizer.SGD,
        learning_rate=0.001,
        momentum=0.0,
        dp_degree=1,
        tp_degree=1,
        pp_degree=1,
        ep_degree=1,
        micro_batch_count=1,
        step_count=1,
        micro_batch_size=1,
        sequence_length=8,
        hidden_size=16,
        vocabulary_size=32,
        activation_dtype=DType.FP16,
        label_dtype=DType.INT32,
        loss_gradient_dtype=DType.FP32,
        weight_dtype=DType.FP16,
        weight_gradient_dtype=DType.FP32,
        stages=(
            S2LiteTrainStage.CE_BACKWARD,
            S2LiteTrainStage.LM_HEAD_WGRAD,
            S2LiteTrainStage.SGD_UPDATE,
        ),
    )


class S2LiteLmHeadTrainTest(unittest.TestCase):
    def test_tiny_contract_and_independent_oracle_goldens(self) -> None:
        contract = _tiny_contract()
        oracle = build_s2_lite_lm_head_train_oracle(contract)

        self.assertEqual(
            S2_LITE_LM_HEAD_TRAIN_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_lm_head_train/v1alpha1",
        )
        self.assertEqual(
            S2_LITE_LM_HEAD_TRAIN_ORACLE_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_lm_head_train_oracle/v1alpha1",
        )
        self.assertEqual(contract.case_id, "case.s2_lite.lm_head_train")
        self.assertEqual(S2_LITE_BASELINE_EPOCH, "s2-lite-v1")
        self.assertEqual(
            oracle.stages,
            (
                S2LiteTrainStage.CE_BACKWARD,
                S2LiteTrainStage.LM_HEAD_WGRAD,
                S2LiteTrainStage.SGD_UPDATE,
            ),
        )
        self.assertEqual(
            (
                oracle.dependency_count,
                oracle.logical_rows,
                oracle.hidden_elements,
                oracle.logits_elements,
                oracle.lm_head_weight_elements,
                oracle.lm_head_weight_bytes,
                oracle.lm_head_weight_gradient_bytes,
            ),
            (2, 8, 128, 256, 512, 1024, 2048),
        )
        self.assertEqual(
            (
                oracle.ce_backward.element_count,
                oracle.ce_backward.logits_read_bytes,
                oracle.ce_backward.labels_read_bytes,
                oracle.ce_backward.loss_gradient_read_bytes,
                oracle.ce_backward.logits_gradient_write_bytes,
            ),
            (256, 512, 32, 32, 512),
        )
        self.assertEqual(
            (
                oracle.lm_head_wgrad.weight_element_count,
                oracle.lm_head_wgrad.hidden_read_bytes,
                oracle.lm_head_wgrad.logits_gradient_read_bytes,
                oracle.lm_head_wgrad.weight_gradient_write_bytes,
                oracle.lm_head_wgrad.floating_point_ops,
            ),
            (512, 256, 512, 2048, 8192),
        )
        self.assertEqual(
            (
                oracle.sgd_update.element_count,
                oracle.sgd_update.weight_read_bytes,
                oracle.sgd_update.gradient_read_bytes,
                oracle.sgd_update.updated_weight_write_bytes,
                oracle.sgd_update.floating_point_ops,
            ),
            (512, 1024, 2048, 1024, 1024),
        )
        self.assertEqual(
            (
                oracle.total_read_bytes,
                oracle.total_write_bytes,
                oracle.total_floating_point_ops,
            ),
            (4416, 3584, 9216),
        )
        oracle.validate_against_contract(contract)
        self.assertEqual(
            contract.id,
            _tiny_contract().id,
            "stable identity must not depend on construction order or process state",
        )

    def test_strict_round_trip_missing_unknown_enum_and_old_versions(self) -> None:
        contract = _tiny_contract()
        oracle = build_s2_lite_lm_head_train_oracle(contract)
        self.assertEqual(
            loads_dataclass(
                S2LiteLmHeadTrainContract,
                canonical_json(contract),
                path="contract",
            ),
            contract,
        )
        self.assertEqual(
            loads_dataclass(
                S2LiteLmHeadTrainOracle,
                canonical_json(oracle),
                path="oracle",
            ),
            oracle,
        )

        raw = json.loads(canonical_json(contract))
        with self.subTest("missing"):
            missing = dict(raw)
            missing.pop("stages")
            with self.assertRaises(FrontendError):
                loads_dataclass(
                    S2LiteLmHeadTrainContract,
                    json.dumps(missing),
                    path="contract",
                )
        with self.subTest("unknown-field"):
            unknown = dict(raw)
            unknown["fallback"] = True
            with self.assertRaises(FrontendError):
                loads_dataclass(
                    S2LiteLmHeadTrainContract,
                    json.dumps(unknown),
                    path="contract",
                )
        with self.subTest("unknown-enum"):
            unknown_enum = dict(raw)
            unknown_enum["coverage"] = "partial_model"
            with self.assertRaises(FrontendError):
                loads_dataclass(
                    S2LiteLmHeadTrainContract,
                    json.dumps(unknown_enum),
                    path="contract",
                )
        with self.subTest("old-contract-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    contract,
                    schema_version="wafer_frontend.s2_lite_lm_head_train/v1alpha0",
                ).validate()
        with self.subTest("old-oracle-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    oracle,
                    schema_version=(
                        "wafer_frontend.s2_lite_lm_head_train_oracle/v1alpha0"
                    ),
                ).validate()

    def test_contract_boundaries_fail_closed(self) -> None:
        contract = _tiny_contract()
        cases = {
            "full-model": {"coverage": S2LiteTrainCoverage.FULL_MODEL},
            "backbone-not-frozen": {"backbone_frozen": False},
            "embedding-not-frozen": {"embedding_frozen": False},
            "adamw": {"optimizer": TrainOptimizer.ADAMW},
            "no-optimizer": {"optimizer": TrainOptimizer.NONE},
            "momentum": {"momentum": 0.9},
            "zero-learning-rate": {"learning_rate": 0.0},
            "nan-learning-rate": {"learning_rate": math.nan},
            "infinite-learning-rate": {"learning_rate": math.inf},
            "dp": {"dp_degree": 2},
            "tp": {"tp_degree": 2},
            "pp": {"pp_degree": 2},
            "ep": {"ep_degree": 2},
            "multiple-microbatches": {"micro_batch_count": 2},
            "multiple-steps": {"step_count": 2},
            "reordered-stages": {"stages": tuple(reversed(contract.stages))},
            "missing-stage": {"stages": contract.stages[:-1]},
            "fp16-weight-gradient": {"weight_gradient_dtype": DType.FP16},
            "short-source-digest": {"source_spec_digest": "1" * 63},
            "uppercase-source-digest": {"source_spec_digest": "A" * 64},
            "nonhex-source-digest": {"source_spec_digest": "g" * 64},
        }
        for name, changes in cases.items():
            with self.subTest(name):
                with self.assertRaises(SchemaError):
                    replace(contract, **changes).validate()

    def test_stable_ids_and_source_recompute_reject_tamper(self) -> None:
        contract = _tiny_contract()
        oracle = build_s2_lite_lm_head_train_oracle(contract)
        with self.subTest("contract-id"):
            with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
                replace(contract, id="s2_lite_lm_head_train_forged").validate()
        with self.subTest("oracle-id"):
            with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
                replace(oracle, id="s2_lite_lm_head_train_oracle_forged").validate()
        with self.subTest("self-consistent-oracle-tamper"):
            other_contract = S2LiteLmHeadTrainContract.create(
                **{
                    **contract._semantic_key(),
                    "hidden_size": contract.hidden_size + 1,
                }
            )
            tampered = build_s2_lite_lm_head_train_oracle(other_contract)
            with self.assertRaisesRegex(SchemaError, "source contract"):
                tampered.validate_against_contract(contract)
        with self.subTest("wrong-source-contract"):
            other = S2LiteLmHeadTrainContract.create(
                **{
                    **contract._semantic_key(),
                    "source_spec_digest": "2" * 64,
                }
            )
            with self.assertRaisesRegex(SchemaError, "source contract"):
                oracle.validate_against_contract(other)


if __name__ == "__main__":
    unittest.main()
