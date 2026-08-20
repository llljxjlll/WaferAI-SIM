from __future__ import annotations

from copy import deepcopy
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes.build_ir0 import (
    build_ir0,
    build_ir0_template_for_profile,
)
from llm.frontend.wafer_frontend.schema import (
    EXPERIMENT_SCHEMA_VERSION,
    PipelineSchedule as PublicPipelineSchedule,
    RecomputeMode as PublicRecomputeMode,
    TrainOptimizer as PublicTrainOptimizer,
    TrainStructure as PublicTrainStructure,
    TrainWorkloadSpec as PublicTrainWorkloadSpec,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    ExperimentSpec,
    TrainOptimizer,
    TrainWorkloadSpec,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    PipelineSchedule,
    RecomputeMode,
    TrainStructure,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import valid_spec


def _train_spec() -> dict[str, object]:
    raw = valid_spec()
    raw["workload"] = {
        "mode": "train",
        "infer": None,
        "train": {
            "global_batch": 4,
            "micro_batch": 1,
            "seq_len": 32,
            "backward": False,
            "optimizer": "none",
            "structure": {
                "micro_batch_count": 2,
                "pp_schedule": "gpipe",
                "interleave_chunks": 1,
                "recompute": "none",
            },
        },
    }
    instance = raw["parallel"]["instances"][0]  # type: ignore[index]
    instance.update(  # type: ignore[union-attr]
        role="train",
        dp=2,
        pp=1,
        ep=1,
    )
    return raw


class ExperimentTrainSchemaTest(unittest.TestCase):
    def test_forward_only_train_round_trip_and_infer_stability(self) -> None:
        self.assertEqual(
            EXPERIMENT_SCHEMA_VERSION,
            "wafer_frontend.experiment/v1alpha5",
        )
        self.assertIs(PublicTrainOptimizer, TrainOptimizer)
        self.assertIs(PublicTrainWorkloadSpec, TrainWorkloadSpec)
        self.assertIs(PublicPipelineSchedule, PipelineSchedule)
        self.assertIs(PublicRecomputeMode, RecomputeMode)
        self.assertIs(PublicTrainStructure, TrainStructure)

        train = from_data(ExperimentSpec, _train_spec(), path="spec")
        train.validate()
        self.assertIsNone(train.workload.infer)
        self.assertIsNotNone(train.workload.train)
        assert train.workload.train is not None
        self.assertEqual(
            (
                train.workload.train.global_batch,
                train.workload.train.structure.micro_batch_count,
                train.parallel.instances[0].dp,
            ),
            (4, 2, 2),
        )
        self.assertEqual(
            loads_dataclass(
                ExperimentSpec,
                canonical_json(train),
                path="spec",
            ),
            train,
        )

        infer = from_data(ExperimentSpec, valid_spec(), path="spec")
        infer.validate()
        self.assertIsNone(infer.workload.train)
        self.assertEqual(
            (infer.model.parameter_elements(), infer.model.parameter_bytes()),
            (train.model.parameter_elements(), train.model.parameter_bytes()),
        )
        self.assertEqual(
            loads_dataclass(
                ExperimentSpec,
                canonical_json(infer),
                path="spec",
            ),
            infer,
        )

    def test_tagged_union_and_old_version_fail_closed(self) -> None:
        raw = _train_spec()
        raw["schema_version"] = "wafer_frontend.experiment/v1alpha4"
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            from_data(ExperimentSpec, raw, path="spec")

        raw = _train_spec()
        raw["workload"]["train"] = None  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "required for train"):
            from_data(ExperimentSpec, raw, path="spec")

        raw = _train_spec()
        raw["workload"]["infer"] = deepcopy(  # type: ignore[index]
            valid_spec()["workload"]["infer"]  # type: ignore[index]
        )
        with self.assertRaisesRegex(SchemaError, "null for train"):
            from_data(ExperimentSpec, raw, path="spec")

        raw = valid_spec()
        raw["workload"]["train"] = _train_spec()["workload"]["train"]  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "null for infer"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_backward_optimizer_recompute_and_schedule_are_unsupported(self) -> None:
        mutations = (
            (("backward",), True, "backward"),
            (("optimizer",), "sgd", "optimizer"),
            (("optimizer",), "adamw", "optimizer"),
            (("structure", "recompute"), "selective", "recompute"),
            (("structure", "recompute"), "full", "recompute"),
            (("structure", "pp_schedule"), "1f1b", "pp_schedule"),
            (
                ("structure", "pp_schedule"),
                "interleaved_1f1b",
                "pp_schedule",
            ),
            (("structure", "interleave_chunks"), 2, "interleave_chunks"),
        )
        for field_path, value, message in mutations:
            with self.subTest(field=field_path, value=value):
                raw = _train_spec()
                train = raw["workload"]["train"]  # type: ignore[index]
                if len(field_path) == 1:
                    train[field_path[0]] = value  # type: ignore[index]
                else:
                    train[field_path[0]][field_path[1]] = value  # type: ignore[index]
                with self.assertRaisesRegex(
                    UnsupportedFeatureError, message
                ):
                    from_data(ExperimentSpec, raw, path="spec")

    def test_train_parallel_and_batch_geometry_are_exact(self) -> None:
        for field in ("replicas", "pp", "ep"):
            with self.subTest(field=field):
                raw = _train_spec()
                raw["parallel"]["instances"][0][field] = 2  # type: ignore[index]
                with self.assertRaisesRegex(
                    UnsupportedFeatureError, field
                ):
                    from_data(ExperimentSpec, raw, path="spec")

        raw = _train_spec()
        raw["workload"]["train"]["global_batch"] = 5  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "micro_batch \* dp"):
            from_data(ExperimentSpec, raw, path="spec")

        raw = _train_spec()
        raw["workload"]["train"]["seq_len"] = 4097  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "max_position_embeddings"):
            from_data(ExperimentSpec, raw, path="spec")

        raw = _train_spec()
        raw["workload"]["train"]["seq_len"] = 33  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "local token count"):
            from_data(ExperimentSpec, raw, path="spec")

        raw = _train_spec()
        duplicate = dict(
            raw["parallel"]["instances"][0],  # type: ignore[index]
            id="P1",
        )
        raw["parallel"]["instances"].append(duplicate)  # type: ignore[index]
        with self.assertRaisesRegex(
            UnsupportedFeatureError, "exactly one TRAIN instance"
        ):
            from_data(ExperimentSpec, raw, path="spec")

    def test_workload_and_instance_roles_cannot_cross(self) -> None:
        raw = _train_spec()
        raw["parallel"]["instances"][0]["role"] = "prefill"  # type: ignore[index]
        raw["parallel"]["instances"][0]["dp"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "role=TRAIN"):
            from_data(ExperimentSpec, raw, path="spec")

        raw = valid_spec()
        raw["parallel"]["instances"][0]["role"] = "train"  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "cannot use role=TRAIN"):
            from_data(ExperimentSpec, raw, path="spec")

    def test_train_graph_expansion_is_an_explicit_n6_1_boundary(self) -> None:
        spec = from_data(ExperimentSpec, _train_spec(), path="spec")
        for producer in (
            lambda: build_ir0(spec),
            lambda: build_ir0_template_for_profile(
                spec,
                instance_ref="P0",
                exact_profile=object(),  # type: ignore[arg-type]
            ),
        ):
            with self.subTest(producer=producer), self.assertRaisesRegex(
                UnsupportedFeatureError,
                "N6.1 train graph expansion is not enabled",
            ):
                producer()


if __name__ == "__main__":
    unittest.main()
