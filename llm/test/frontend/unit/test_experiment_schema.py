from __future__ import annotations

from copy import deepcopy
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.schema.common import DType, UINT64_MAX
from llm.frontend.wafer_frontend.schema.experiment import (
    ExperimentSpec,
    InferOutput,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import valid_spec


class ExperimentSchemaTest(unittest.TestCase):
    def test_valid_mvp_spec_decodes_to_immutable_tuples(self) -> None:
        spec = from_data(ExperimentSpec, valid_spec(), path="spec")
        self.assertIsInstance(spec.parallel.instances, tuple)
        self.assertEqual(spec.workload.infer.profile.context_sum, 32)
        encoded = canonical_json(spec)
        self.assertEqual(loads_dataclass(ExperimentSpec, encoded, path="spec"), spec)

    def test_stage2_model_and_output_contract_is_required_and_exact(self) -> None:
        spec = from_data(ExperimentSpec, valid_spec(), path="spec")
        self.assertIs(spec.workload.infer.output, InferOutput.LOGITS)
        self.assertEqual(
            (
                spec.model.V,
                spec.model.rotary_dim,
                spec.model.tie_word_embeddings,
                spec.model.rms_norm_epsilon,
                spec.model.rope_theta,
                spec.model.max_position_embeddings,
            ),
            (512, 64, False, 1e-5, 10000.0, 4096),
        )

        for field in (
            "V",
            "rotary_dim",
            "tie_word_embeddings",
            "rms_norm_epsilon",
            "rope_theta",
            "max_position_embeddings",
        ):
            with self.subTest(missing=field):
                raw = valid_spec()
                del raw["model"][field]  # type: ignore[index]
                with self.assertRaisesRegex(SchemaError, field):
                    from_data(ExperimentSpec, raw, path="spec")

        raw = valid_spec()
        del raw["workload"]["infer"]["output"]  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "output"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_stage2_model_scope_and_old_version_fail_closed(self) -> None:
        raw = valid_spec()
        raw["schema_version"] = "wafer_frontend.experiment/v1alpha2"
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            from_data(ExperimentSpec, raw, path="spec")

        for field, value, message in (
            ("tie_word_embeddings", True, "tie_word_embeddings"),
            ("rotary_dim", 32, "rotary_dim"),
            ("rms_norm_epsilon", 0.0, "finite positive"),
            ("rope_theta", float("inf"), "finite positive"),
        ):
            with self.subTest(field=field):
                raw = valid_spec()
                raw["model"][field] = value  # type: ignore[index]
                with self.assertRaisesRegex(
                    (SchemaError, UnsupportedFeatureError), message
                ):
                    from_data(ExperimentSpec, raw, path="spec")

        raw = valid_spec()
        raw["workload"]["infer"]["output"] = "sample"  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_stage2_tiny_parameter_oracle_is_6224_elements_12448_bytes(self) -> None:
        raw = valid_spec()
        raw["model"].update(  # type: ignore[index,union-attr]
            {
                "V": 32,
                "H": 16,
                "I": 32,
                "NH": 4,
                "KVH": 4,
                "DH": 4,
                "rotary_dim": 4,
                "L": 2,
            }
        )
        spec = from_data(ExperimentSpec, raw, path="spec")
        self.assertEqual(spec.model.parameter_elements(), 6224)
        self.assertEqual(spec.model.parameter_bytes(), 12448)

    def test_all_profile_dimensions_are_explicit(self) -> None:
        raw = valid_spec()
        del raw["workload"]["infer"]["profile"]["context_sum"]  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "context_sum"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_t_bucket_cannot_replace_profile(self) -> None:
        raw = valid_spec()
        raw["workload"]["infer"] = {  # type: ignore[index]
            "source": "shape_dist",
            "output": "logits",
            "t_buckets": [32],
        }
        with self.assertRaises(SchemaError):
            from_data(ExperimentSpec, raw, path="spec")

    def test_unknown_field_and_enum_case_are_rejected(self) -> None:
        raw = valid_spec()
        raw["extra"] = 1
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(ExperimentSpec, raw, path="spec")
        raw = valid_spec()
        raw["model"]["dtype"] = "FP16"  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_policy_availability_is_not_in_the_syntax_schema(self) -> None:
        for field, value in (
            ("inter_die", "swizzle_topo"),
            ("intra_die", "optimized"),
        ):
            raw = valid_spec()
            raw["policy"][field] = value  # type: ignore[index]
            spec = from_data(ExperimentSpec, raw, path="spec")
            self.assertEqual(getattr(spec.policy, field).value, value)

    def test_compact_and_explicit_placement_contracts_are_strict(self) -> None:
        compact = from_data(ExperimentSpec, valid_spec(), path="spec")
        self.assertEqual(compact.placement.groups, ())

        raw = valid_spec()
        raw["placement"] = {
            "strategy": "explicit",
            "groups": [
                {
                    "instance_id": "P0",
                    "mesh_ref": "P0.mesh.tp",
                    "die_ids": [3, 0],
                }
            ],
        }
        explicit = from_data(ExperimentSpec, raw, path="spec")
        self.assertEqual(explicit.placement.groups[0].die_ids, (3, 0))
        self.assertEqual(
            loads_dataclass(ExperimentSpec, canonical_json(explicit), path="spec"),
            explicit,
        )

        invalid_placements = (
            {
                "strategy": "compact",
                "groups": [
                    {
                        "instance_id": "P0",
                        "mesh_ref": "P0.mesh.tp",
                        "die_ids": [0, 1],
                    }
                ],
            },
            {"strategy": "explicit", "groups": []},
            {
                "strategy": "explicit",
                "groups": [
                    {
                        "instance_id": "P0",
                        "mesh_ref": "P0.mesh.tp",
                        "die_ids": [0, 0],
                    }
                ],
            },
            {
                "strategy": "explicit",
                "groups": [
                    {"instance_id": "P1", "mesh_ref": "mesh", "die_ids": [0]},
                    {"instance_id": "P0", "mesh_ref": "mesh", "die_ids": [1]},
                ],
            },
        )
        for placement in invalid_placements:
            with self.subTest(placement=placement):
                changed = valid_spec()
                changed["placement"] = placement
                with self.assertRaises(SchemaError):
                    from_data(ExperimentSpec, changed, path="spec")

    def test_dynamic_and_multi_instance_modes_are_rejected(self) -> None:
        raw = valid_spec()
        raw["backend"]["dynamic_region_dispatch"] = True  # type: ignore[index]
        with self.assertRaises(UnsupportedFeatureError):
            from_data(ExperimentSpec, raw, path="spec")
        raw = valid_spec()
        raw["parallel"]["instances"].append(  # type: ignore[index,union-attr]
            dict(raw["parallel"]["instances"][0], id="P1")  # type: ignore[index]
        )
        with self.assertRaises(UnsupportedFeatureError):
            from_data(ExperimentSpec, raw, path="spec")

    def test_static_shape_distribution_round_trip_and_profile_ids(self) -> None:
        raw = valid_spec()
        profile = raw["workload"]["infer"].pop("profile")  # type: ignore[index,union-attr]
        raw["workload"]["infer"].update({  # type: ignore[index,union-attr]
            "source": "shape_dist",
            "shape_dist": {
                "profiles": [
                    {"key": profile, "weight": 0.25},
                    {
                        "key": dict(profile, prefill_tokens=64, context_sum=64, context_max=64),
                        "weight": 0.75,
                    },
                ]
            },
        })
        spec = from_data(ExperimentSpec, raw, path="spec")
        decoded = loads_dataclass(ExperimentSpec, canonical_json(spec), path="spec")
        self.assertEqual(decoded, spec)
        first = spec.workload.infer.profiles()[0]
        self.assertEqual(first.stable_id(), decoded.workload.infer.profiles()[0].stable_id())
        self.assertEqual(len(first.digest()), 64)

    def test_shape_distribution_rejects_empty_duplicate_and_invalid_weight(self) -> None:
        base = valid_spec()
        profile = base["workload"]["infer"]["profile"]  # type: ignore[index]
        for profiles in (
            [],
            [{"key": profile, "weight": 1.0}, {"key": deepcopy(profile), "weight": 2.0}],
            [{"key": profile, "weight": 0.0}],
        ):
            raw = valid_spec()
            raw["workload"]["infer"] = {  # type: ignore[index]
                "source": "shape_dist",
                "output": "logits",
                "profile": None,
                "shape_dist": {"profiles": profiles},
            }
            with self.assertRaises(SchemaError):
                from_data(ExperimentSpec, raw, path="spec")

    def test_shape_distribution_rejects_dynamic_arrival_fields(self) -> None:
        raw = valid_spec()
        profile = raw["workload"]["infer"].pop("profile")  # type: ignore[index,union-attr]
        raw["workload"]["infer"].update({  # type: ignore[index,union-attr]
            "source": "shape_dist",
            "shape_dist": {"profiles": [{"key": profile, "weight": 1.0}]},
            "arrival": {"kind": "poisson"},
        })
        with self.assertRaisesRegex(SchemaError, "arrival"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_model_intrinsic_shape_and_gqa_contracts_are_enforced(self) -> None:
        for field, value, message in (
            ("H", 255, "NH \* DH"),
            ("KVH", 8, "must not exceed NH"),
            ("KVH", 3, "NH must be divisible by KVH"),
        ):
            raw = valid_spec()
            raw["model"][field] = value  # type: ignore[index]
            with self.assertRaisesRegex(SchemaError, message):
                from_data(ExperimentSpec, raw, path="spec")

    def test_model_rejects_overflow_in_derived_dimensions(self) -> None:
        raw = valid_spec()
        raw["model"].update(  # type: ignore[index,union-attr]
            {
                "H": UINT64_MAX,
                "I": 1,
                "NH": 1,
                "KVH": 1,
                "DH": UINT64_MAX,
                "rotary_dim": UINT64_MAX,
                "L": 1,
            }
        )
        with self.assertRaisesRegex(SchemaError, "derived.Q"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_direct_experiment_validation_recurses_into_nested_specs(self) -> None:
        spec = from_data(ExperimentSpec, valid_spec(), path="spec")
        with self.assertRaisesRegex(SchemaError, "hardware.ref"):
            replace(spec, hardware=replace(spec.hardware, ref="")).validate()
        bad_reduction = replace(
            spec.backend.reduction_contract, accumulate=DType.FP16
        )
        with self.assertRaises(UnsupportedFeatureError):
            replace(
                spec,
                backend=replace(spec.backend, reduction_contract=bad_reduction),
            ).validate()

    def test_tp_divisibility_and_sequence_parallel_profiles_are_checked(self) -> None:
        cases = (
            ("H", lambda raw: raw["parallel"]["instances"][0].update(tp=3)),
            ("NH", lambda raw: raw["parallel"]["instances"][0].update(tp=8)),
            ("KVH", lambda raw: raw["parallel"]["instances"][0].update(tp=4)),
            (
                "I",
                lambda raw: (
                    raw["model"].update(KVH=4, I=510),
                    raw["parallel"]["instances"][0].update(tp=4),
                ),
            ),
        )
        for dimension, mutate in cases:
            raw = valid_spec()
            mutate(raw)  # type: ignore[arg-type]
            with self.assertRaisesRegex(SchemaError, f"dimension {dimension}"):
                from_data(ExperimentSpec, raw, path="spec")

        raw = valid_spec()
        raw["workload"]["infer"]["profile"]["prefill_tokens"] = 33  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "sequence-parallel"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_dense_profiles_reject_expert_load(self) -> None:
        raw = valid_spec()
        raw["workload"]["infer"]["profile"]["expert_load"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "Dense model"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_shape_distribution_weights_must_sum_to_one_without_normalization(self) -> None:
        raw = valid_spec()
        profile = raw["workload"]["infer"].pop("profile")  # type: ignore[index,union-attr]
        raw["workload"]["infer"].update(  # type: ignore[index,union-attr]
            {
                "source": "shape_dist",
                "shape_dist": {
                    "profiles": [
                        {"key": profile, "weight": 0.4},
                        {
                            "key": dict(
                                profile,
                                prefill_tokens=64,
                                context_sum=64,
                                context_max=64,
                            ),
                            "weight": 0.5,
                        },
                    ]
                },
            }
        )
        with self.assertRaisesRegex(SchemaError, "sum to 1"):
            from_data(ExperimentSpec, raw, path="spec")


if __name__ == "__main__":
    unittest.main()
