from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema._validation_session import (
    builder_validation_session,
    mark_validation_complete,
    validation_seen,
)
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    _materialize_flexible_dense_train_forward,
    build_flexible_dense_train_plan,
    materialize_flexible_dense_train_forward,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
)
from llm.frontend.wafer_frontend.schema.flexible_dense_train import (
    FlexibleDenseTrainActionKind,
    FlexibleDenseTrainCapabilityStatus,
    FlexibleDenseTrainSpec,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware


def _spec(rows: int, columns: int) -> ExperimentSpec:
    hidden = 4 * columns
    raw = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "model": {
            "source": "analytic",
            "arch": "llama",
            "V": 8 * columns,
            "H": hidden,
            "I": 8 * columns,
            "NH": columns,
            "KVH": columns,
            "DH": 4,
            "rotary_dim": 4,
            "L": 2,
            "dtype": "fp16",
            "tie_word_embeddings": False,
            "rms_norm_epsilon": 1.0e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 64,
            "moe": None,
        },
        "hardware": {"ref": "synthetic_rect_mesh.json"},
        "workload": {
            "mode": "train",
            "infer": None,
            "train": {
                "global_batch": rows,
                "micro_batch": 1,
                "seq_len": columns,
                "backward": False,
                "optimizer": "none",
                "structure": {
                    "micro_batch_count": 1,
                    "pp_schedule": "gpipe",
                    "interleave_chunks": 1,
                    "recompute": "none",
                },
            },
        },
        "parallel": {
            "instances": [
                {
                    "id": "T0",
                    "role": "train",
                    "tp": columns,
                    "sp": columns > 1,
                    "replicas": 1,
                    "dp": rows,
                    "pp": 1,
                    "ep": 1,
                }
            ]
        },
        "placement": {"strategy": "compact", "groups": []},
        "policy": {
            "partition": "gemm_coll",
            "inter_die": "naive",
            "intra_die": "naive",
        },
        "backend": {
            "execution": "unified_stream",
            "ordinary_lowering": "json_coarse",
            "fused_lowering": "isa_region",
            "standalone_collective_lowering": "strict_actions",
            "reduction_contract": {
                "accumulate": "fp32",
                "rounding": "rne",
                "validation": "timing",
            },
            "transport": "strict",
            "static_link": True,
            "dynamic_region_dispatch": False,
        },
    }
    return from_data(ExperimentSpec, raw, path="flexible_dense_train.spec")


def _hardware(rows: int, columns: int) -> dict[str, object]:
    hardware = minimal_hardware(columns, rows)
    memory = hardware["memory"]
    assert isinstance(memory, dict)
    memory["sram_size"] = 1 << 20
    sram = memory["sram"]
    assert isinstance(sram, dict)
    sram["capacity_bytes"] = 1 << 20
    regions = sram["regions"]
    assert isinstance(regions, list)
    region = regions[0]
    assert isinstance(region, dict)
    region["size_bytes"] = 1 << 20
    return hardware


class FlexibleDenseTrainPlanTest(unittest.TestCase):
    def test_all_one_hundred_rectangles_build_deterministic_finite_steps(self) -> None:
        for rows in range(1, 11):
            for columns in range(1, 11):
                with self.subTest(rows=rows, columns=columns):
                    mesh = RectMeshSpec(rows, columns)
                    plan = build_flexible_dense_train_plan(
                        _spec(rows, columns), mesh
                    )
                    plan.validate()
                    self.assertEqual(len(plan.tp_groups), rows)
                    self.assertEqual(len(plan.dp_groups), columns)
                    self.assertEqual(
                        {item.state_ref for item in plan.parameter_templates},
                        {item.id for item in plan.forward_graph.persistent_states},
                    )
                    self.assertEqual(
                        len(plan.gradient_waves),
                        2 * len(plan.parameter_templates) * (rows - 1),
                    )
                    self.assertLessEqual(
                        max(
                            (
                                wave.max_sessions_per_rank
                                for wave in plan.gradient_waves
                            ),
                            default=0,
                        ),
                        3,
                    )
                    by_rank = {
                        rank: tuple(
                            action
                            for action in plan.rank_actions
                            if action.rank == rank
                        )
                        for rank in range(mesh.rank_count)
                    }
                    for actions in by_rank.values():
                        rank = actions[0].rank
                        row, column = divmod(rank, columns)
                        local_templates = tuple(
                            item
                            for item in plan.parameter_templates
                            if item.tp_shard_index == column
                        )
                        self.assertEqual(
                            len(actions),
                            2 * len(plan.forward_node_refs)
                            + len(local_templates)
                            * (
                                4
                                + 2
                                * sum(
                                    1
                                    for child in range(1, rows)
                                    if row in (child, (child - 1) // 2)
                                )
                            ),
                        )
                        self.assertEqual(
                            {
                                action.state_ref
                                for action in actions
                                if action.kind
                                is FlexibleDenseTrainActionKind.WEIGHT_GRADIENT
                            },
                            {item.state_ref for item in local_templates},
                        )
                        sgd_index = next(
                            index
                            for index, action in enumerate(actions)
                            if action.kind
                            is FlexibleDenseTrainActionKind.SGD_UPDATE
                        )
                        sync_indexes = tuple(
                            index
                            for index, action in enumerate(actions)
                            if action.kind
                            is FlexibleDenseTrainActionKind.GRADIENT_SYNC
                        )
                        self.assertTrue(
                            not sync_indexes or max(sync_indexes) < sgd_index
                        )
                        wgrad_indexes = tuple(
                            index
                            for index, action in enumerate(actions)
                            if action.kind
                            is FlexibleDenseTrainActionKind.WEIGHT_GRADIENT
                        )
                        self.assertLess(max(wgrad_indexes), sgd_index)
                        self.assertTrue(
                            not sync_indexes
                            or max(wgrad_indexes) < min(sync_indexes)
                        )
                    for template in plan.parameter_templates:
                        self.assertEqual(
                            template.owner_ranks,
                            tuple(
                                row * columns + template.tp_shard_index
                                for row in range(rows)
                            ),
                        )
                    self.assertIs(
                        plan.backward_lower_link_status,
                        FlexibleDenseTrainCapabilityStatus.OUT_OF_SCOPE,
                    )
                    self.assertFalse(plan.full_model_backward_materialized)

    def test_spec_round_trip_and_plan_repeatability_are_stable(self) -> None:
        source = _spec(2, 3)
        mesh = RectMeshSpec(2, 3)
        first = build_flexible_dense_train_plan(source, mesh)
        second = build_flexible_dense_train_plan(source, mesh)
        self.assertEqual(first, second)
        self.assertEqual(first.id, second.id)
        self.assertEqual(
            loads_dataclass(
                FlexibleDenseTrainSpec,
                canonical_json(first.spec),
            ),
            first.spec,
        )

    def test_optimizer_cannot_bypass_gradient_sync(self) -> None:
        plan = build_flexible_dense_train_plan(
            _spec(2, 3), RectMeshSpec(2, 3)
        )
        actions = list(plan.rank_actions)
        sgd_index = next(
            index
            for index, action in enumerate(actions)
            if action.rank == 0
            and action.kind is FlexibleDenseTrainActionKind.SGD_UPDATE
        )
        actions[sgd_index] = replace(
            actions[sgd_index],
            depends_on=(
                next(
                    action.id
                    for action in actions
                    if action.rank == 0
                    and action.kind
                    is FlexibleDenseTrainActionKind.WEIGHT_GRADIENT
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "all state"):
            replace(plan, rank_actions=tuple(actions)).validate()

    def test_parameter_ownership_and_coverage_fail_closed(self) -> None:
        plan = build_flexible_dense_train_plan(
            _spec(2, 3), RectMeshSpec(2, 3)
        )
        victim = plan.parameter_templates[0]
        with self.assertRaisesRegex(SchemaError, "all forward persistent states"):
            replace(
                plan,
                parameter_templates=plan.parameter_templates[1:],
            ).validate()
        with self.assertRaisesRegex(SchemaError, "all forward persistent states"):
            replace(
                plan,
                parameter_templates=(
                    replace(victim, owner_ranks=victim.owner_ranks[:-1]),
                    *plan.parameter_templates[1:],
                ),
            ).validate()

    def test_source_axis_mismatch_fails_before_graph_construction(self) -> None:
        source = _spec(2, 3)
        instance = source.parallel.instances[0]
        wrong = replace(
            source,
            parallel=replace(
                source.parallel,
                instances=(replace(instance, dp=1),),
            ),
            workload=replace(
                source.workload,
                train=replace(
                    source.workload.train,
                    global_batch=1,
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "DP=rows"):
            build_flexible_dense_train_plan(wrong, RectMeshSpec(2, 3))


class BuilderValidationSessionTest(unittest.TestCase):
    def test_cache_is_private_scoped_and_identity_exact(self) -> None:
        first = object()
        second = object()
        self.assertFalse(validation_seen(first, "test"))
        with builder_validation_session():
            mark_validation_complete(first, "test")
            self.assertTrue(validation_seen(first, "test"))
            self.assertFalse(validation_seen(second, "test"))
            self.assertFalse(validation_seen(first, "different-domain"))
        self.assertFalse(validation_seen(first, "test"))


class FlexibleDenseTrainForwardCarrierTest(unittest.TestCase):
    def test_one_by_one_reuses_production_forward_lower_link(self) -> None:
        hardware = _hardware(1, 1)
        carrier = materialize_flexible_dense_train_forward(
            _spec(1, 1),
            RectMeshSpec(1, 1),
            physical_fabric_from_data(hardware),
            hbm_address_spaces_from_data(hardware),
            producer_pass="flexible_dense_train_test",
        )
        carrier.validate()
        self.assertGreater(carrier.core_stream_count, 0)
        self.assertGreater(carrier.symbolic_record_count, 0)
        self.assertEqual(
            carrier.linked_forward.source.dp_degree,
            1,
        )

    def test_private_fast_path_matches_strict_and_public_rejects_forged_dag(self) -> None:
        hardware = _hardware(1, 1)
        inputs = (
            _spec(1, 1),
            RectMeshSpec(1, 1),
            physical_fabric_from_data(hardware),
            hbm_address_spaces_from_data(hardware),
        )
        fast = materialize_flexible_dense_train_forward(
            *inputs,
            producer_pass="flexible_dense_train_test",
        )
        strict = _materialize_flexible_dense_train_forward(
            *inputs,
            producer_pass="flexible_dense_train_test",
        )
        self.assertEqual(fast.id, strict.id)
        self.assertEqual(canonical_json(fast), canonical_json(strict))
        self.assertEqual(
            canonical_json(fast.linked_forward.manifest).encode("utf-8"),
            canonical_json(strict.linked_forward.manifest).encode("utf-8"),
        )
        replica = fast.linked_forward.source.replicas[0]
        fragment = replica.fragments[0]
        dag = replica.lowering_context.global_dag
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            fragment.validate_against(replace(dag, id="forged"))


if __name__ == "__main__":
    unittest.main()
