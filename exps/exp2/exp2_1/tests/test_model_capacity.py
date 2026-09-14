from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_ROOT))

import capacity_model as capacity
import model_manifests as manifests


class ModelManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = manifests.load_model_manifests()

    def test_six_complete_manifests_in_stable_order(self) -> None:
        self.assertEqual(tuple(self.models), manifests.MODEL_ORDER)
        self.assertEqual(len(self.models), 6)
        for model in self.models.values():
            for field in manifests.REQUIRED_FIELDS:
                self.assertIn(field, model)
            self.assertEqual(sum(model["parameter_breakdown"].values()), model["parameter_count"])
            self.assertEqual(len(model["manifest_digest"]), 64)

    def test_source_snapshots_are_local_digest_scope(self) -> None:
        for model in self.models.values():
            self.assertTrue(model["sources"])
            for source in model["sources"]:
                self.assertIn("not upstream remote bytes", source["digest_scope"])
                path = manifests.SOURCE_DIR / source["local_snapshot"]
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                manifests.validate_source_snapshot(snapshot)
                self.assertEqual(snapshot["snapshot_digest"], source["snapshot_digest"])

    def test_digest_detects_mutation(self) -> None:
        changed = copy.deepcopy(self.models["llama3_8b"])
        changed["hidden_size"] += 1
        with self.assertRaises(manifests.ManifestError):
            manifests.validate_manifest(changed)

    def test_deepseek_is_mla_not_standard_gqa(self) -> None:
        model = self.models["deepseek_v3"]
        self.assertEqual(model["attention_type"], "MLA")
        self.assertIsNone(model["head_dim"])
        self.assertEqual(model["intermediate_size"], 2048)
        self.assertEqual(model["dense_intermediate_size"], 18432)
        self.assertEqual(model["dense_layer_prefix"], 3)
        self.assertEqual(model["mla_cache_width"], 576)
        self.assertEqual(capacity.kv_elements_per_token_per_layer(model), 576)

    def test_architecture_counts_are_frozen_integers(self) -> None:
        self.assertEqual(self.models["llama2_7b"]["parameter_count"], 6_738_415_616)
        self.assertEqual(self.models["llama3_8b"]["parameter_count"], 8_030_261_248)
        self.assertEqual(self.models["llama3_1_405b"]["parameter_count"], 405_853_388_800)
        self.assertEqual(self.models["mixtral_8x7b"]["parameter_count"], 46_702_792_704)
        self.assertEqual(self.models["deepseek_v3"]["parameter_count"], 671_026_404_352)
        self.assertEqual(self.models["gpt3_175b"]["parameter_count"], 174_604_259_328)


class CapacityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = manifests.load_model_manifests()

    def test_hardware_capacity_is_four_binary_16_gib_stacks(self) -> None:
        self.assertEqual(capacity.HBM_CAPACITY_BYTES, 64 * 1024**3)

    def test_training_adamw_and_gradient_are_not_tiny_sgd_proxy(self) -> None:
        row = capacity.audit_training_case(self.models["llama2_7b"], 2304)
        resident = 4 * self.models["llama2_7b"]["parameter_count"]
        self.assertEqual(row["resident_parameter_count"], resident)
        self.assertEqual(row["parameter_bytes"], 2 * resident)
        self.assertEqual(row["gradient_bytes"], 2 * resident)
        self.assertEqual(row["optimizer_bytes"], 12 * resident)
        self.assertEqual(row["memory_policy"]["optimizer"], "AdamW")
        self.assertEqual(row["memory_policy"]["optimizer_sharding"], "none")
        self.assertGreater(row["activation_peak_bytes"], 0)
        self.assertEqual(row["capacity_status"], "capacity_infeasible_projection")

    def test_moe_experts_partition_while_shared_components_replicate(self) -> None:
        model = self.models["mixtral_8x7b"]
        short = capacity.audit_training_case(model, 2304)
        factors = short["parameter_placement_factors"]
        self.assertEqual(factors["routed_experts"], 1)
        self.assertEqual(factors["attention"], 4)
        self.assertEqual(factors["embedding_and_lm_head"], 4)

    def test_per_block_checkpoint_long_sequence_is_larger(self) -> None:
        model = self.models["llama3_8b"]
        short = capacity.audit_training_case(model, 2304)
        long = capacity.audit_training_case(model, 36864)
        self.assertEqual(long["activation_peak_bytes"], 16 * short["activation_peak_bytes"])
        self.assertEqual(short["memory_policy"]["activation_checkpoint"], "per_transformer_block")

    def test_inference_weight_modes_and_standard_kv_formula(self) -> None:
        model = self.models["llama3_8b"]
        shared = capacity.audit_inference_case(model, 64, "global_shared_read_only")
        replicated = capacity.audit_inference_case(model, 64, "replicated_per_instance")
        weight_bytes = 2 * model["parameter_count"]
        self.assertEqual(shared["shared_weight_bytes"], weight_bytes)
        self.assertEqual(shared["replicated_weight_bytes"], 0)
        self.assertEqual(replicated["replicated_weight_bytes"], 6 * weight_bytes)
        self.assertEqual(replicated["total_resident_bytes"] - shared["total_resident_bytes"], 5 * weight_bytes)
        self.assertEqual(shared["kv_cache_formula"]["elements_per_token_per_layer"], 2 * 8 * 128)

    def test_inference_capacity_counts_two_decode_and_four_prefill_instances(self) -> None:
        model = self.models["deepseek_v3"]
        row = capacity.audit_inference_case(model, 512, "global_shared_read_only")
        sequence_tokens = 2 * 512 * 36864 + 4 * 2304
        expected = sequence_tokens * 61 * 576 * 2
        self.assertEqual(row["kv_cache_bytes"], expected)
        self.assertEqual(row["kv_cache_formula"]["representation"], "mla_compressed_latent_plus_rope_key")

    def test_complete_audit_has_12_training_and_24_inference_rows(self) -> None:
        audit = capacity.build_capacity_audit()
        self.assertEqual(audit["training_case_count"], 12)
        self.assertEqual(audit["inference_case_count"], 24)
        self.assertEqual(len(audit["capacity_audit_digest"]), 64)
        required = {
            "parameter_bytes", "optimizer_bytes", "gradient_bytes",
            "activation_peak_bytes", "kv_cache_bytes", "shared_weight_bytes",
            "replicated_weight_bytes", "total_resident_bytes",
            "hbm_capacity_bytes", "capacity_status",
        }
        for row in audit["training"] + audit["inference"]:
            self.assertTrue(required.issubset(row))


if __name__ == "__main__":
    unittest.main()
