from __future__ import annotations

from dataclasses import replace
import unittest

from llm.test.frontend.integration.lite_train_cases import (
    build_s2_lite_source_case,
)

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.fusion_partition import (
    partition_train_forward,
)
from llm.frontend.wafer_frontend.passes.inter_die_plan import (
    plan_train_forward,
)
from llm.frontend.wafer_frontend.passes.intra_die_schedule import (
    schedule_train_forward,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    load_physical_fabric_and_hbm_address_spaces,
)
from llm.frontend.wafer_frontend.passes.placement import (
    place_train_forward_ir0,
)
from llm.frontend.wafer_frontend.passes.project_to_ir2 import (
    project_train_forward,
)
from llm.frontend.wafer_frontend.passes.train_global_action import (
    build_s2_lite_train_global_action,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1_SCHEMA_VERSION
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferOwnership,
    IntraDieSchedule,
    SemanticTaskKind,
)
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    IntraDieSchedulingContext,
    ProjectToIR2Context,
    TRAIN_PROJECTED_IR2_SCHEMA_VERSION,
    TRAIN_SCHEDULED_IR2_SCHEMA_VERSION,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.train_global_action import (
    S2_LITE_TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
    S2LiteTrainGlobalAction,
)


def _build_chain():
    source = build_s2_lite_source_case()
    fabric, spaces = load_physical_fabric_and_hbm_address_spaces(
        source.runtime_inputs.source_hardware_path,
        source.runtime_inputs.source_mapping_path,
    )
    placed = place_train_forward_ir0(
        source.logical.graph,
        PlacementContext.create(
            producer_pass="test_s2_lite_production_chain",
            fabric=fabric,
            placement=source.spec.placement,
            hbm_address_spaces=spaces,
        ),
    )
    partitioned = partition_train_forward(
        placed,
        FusionPartitionContext.create(
            producer_pass="test_s2_lite_production_chain"
        ),
    )
    registry = production_registry()
    planned = plan_train_forward(
        partitioned,
        InterDiePlanningContext.create(
            producer_pass="test_s2_lite_production_chain",
            fused_policy=registry.instantiate(
                RegistryKind.INTER_DIE, "naive"
            ).selection,
            standalone_policy=registry.instantiate(
                RegistryKind.STANDALONE_COLLECTIVE,
                "direct_all_gather",
            ).selection,
        ),
    )
    projected = project_train_forward(
        planned,
        ProjectToIR2Context.create(
            producer_pass="test_s2_lite_production_chain",
            state_transfers=(),
        ),
    )
    scheduled = schedule_train_forward(
        projected,
        IntraDieSchedulingContext.create(
            producer_pass="test_s2_lite_production_chain",
            policy=registry.instantiate(
                RegistryKind.INTRA_DIE, "naive"
            ).selection,
        ),
    )
    global_action = build_s2_lite_train_global_action(scheduled)
    return source, placed, partitioned, planned, projected, scheduled, global_action


def _rebuild_schedule(schedule: IntraDieSchedule, *, buffer_bindings):
    return IntraDieSchedule.create(
        producer_pass=schedule.producer_pass,
        dag_id=schedule.dag_id,
        die_id=schedule.die_id,
        placements=schedule.placements,
        buffer_bindings=buffer_bindings,
        task_buffer_uses=schedule.task_buffer_uses,
        task_state_uses=schedule.task_state_uses,
        flow_routes=schedule.flow_routes,
        runtime_bindings=schedule.runtime_bindings,
        core_orders=schedule.core_orders,
    )


class S2LiteTrainProductionChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chain = _build_chain()

    def test_exact_placement_n4_projection_schedule_and_global_counts(self) -> None:
        (
            _source,
            placed,
            partitioned,
            planned,
            projected,
            scheduled,
            global_action,
        ) = self.chain
        self.assertEqual(IR1_SCHEMA_VERSION, "wafer_frontend.ir1/v1alpha14")
        self.assertEqual(
            TRAIN_PROJECTED_IR2_SCHEMA_VERSION,
            "wafer_frontend.train_projected_ir2/v1alpha2",
        )
        self.assertEqual(
            TRAIN_SCHEDULED_IR2_SCHEMA_VERSION,
            "wafer_frontend.train_scheduled_ir2/v1alpha2",
        )
        graph = placed.replicas[0].graph
        manifest = graph.persistent_state_manifest
        assert manifest is not None
        self.assertEqual(
            (
                len(graph.nodes),
                len(manifest.declarations),
                len(graph.state_accesses),
                len(partitioned.replicas[0].fused_op_skeletons),
                len(planned.replicas[0].fusion_plans),
                len(planned.replicas[0].standalone_plans),
            ),
            (29, 15, 16, 0, 0, 0),
        )
        trainable = tuple(
            declaration
            for declaration in manifest.declarations
            if declaration.identity.kind is StateKind.TRAINABLE_PARAMETER
        )
        self.assertEqual(len(trainable), 1)
        self.assertIs(trainable[0].access, PersistentStateAccess.READ_WRITE)

        projection = projected.replicas[0].projection
        tasks = tuple(task for dag in projection.dags for task in dag.tasks)
        self.assertEqual(
            (
                len(tasks),
                sum(task.kind is SemanticTaskKind.COMP for task in tasks),
                sum(task.kind is SemanticTaskKind.DMA_IN for task in tasks),
                sum(task.kind is SemanticTaskKind.DMA_OUT for task in tasks),
                sum(len(dag.state_staging_values) for dag in projection.dags),
            ),
            (46, 29, 16, 1, 16),
        )
        schedules = scheduled.replicas[0].schedule_set.schedules
        self.assertEqual(
            (
                sum(len(item.placements) for item in schedules),
                sum(len(item.buffer_bindings) for item in schedules),
                sum(len(item.task_buffer_uses) for item in schedules),
                sum(len(item.task_state_uses) for item in schedules),
                sum(
                    binding.ownership is BufferOwnership.ALIASED
                    for item in schedules
                    for binding in item.buffer_bindings
                ),
            ),
            (46, 48, 99, 17, 1),
        )
        self.assertEqual(len(global_action.global_dags), 1)
        self.assertEqual(len(global_action.global_dags[0].actions), 46)
        with self.subTest("old-ir1-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    graph,
                    schema_version="wafer_frontend.ir1/v1alpha13",
                ).validate()
        with self.subTest("old-projected-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    projected,
                    schema_version="wafer_frontend.train_projected_ir2/v1alpha1",
                ).validate()
        with self.subTest("old-scheduled-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    scheduled,
                    schema_version="wafer_frontend.train_scheduled_ir2/v1alpha1",
                ).validate()

    def test_global_carrier_strict_roundtrip_stable_id_and_version(self) -> None:
        result = self.chain[-1]
        self.assertEqual(
            S2_LITE_TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_train_global_action/v1alpha1",
        )
        self.assertEqual(
            loads_dataclass(
                S2LiteTrainGlobalAction,
                canonical_json(result),
                path="s2_lite_global_action",
            ),
            result,
        )
        self.assertEqual(_build_chain()[-1].id, result.id)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                result,
                schema_version="wafer_frontend.s2_lite_train_global_action/v1alpha0",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(result, id="s2_lite_train_global_action_forged").validate()

    def test_alias_view_owner_and_state_permission_tamper_fail_closed(self) -> None:
        scheduled = self.chain[-2]
        replica = scheduled.replicas[0]
        schedule = next(
            item for item in replica.schedule_set.schedules if item.buffer_bindings
        )
        dag = next(
            item
            for item in replica.projected.projection.dags
            if item.id == schedule.dag_id
        )
        alias = next(
            binding
            for binding in schedule.buffer_bindings
            if binding.ownership is BufferOwnership.ALIASED
        )
        root = next(
            binding for binding in schedule.buffer_bindings if binding.id == alias.alias_of
        )
        other_root = next(
            binding
            for binding in schedule.buffer_bindings
            if binding.ownership is not BufferOwnership.ALIASED
            and binding.id != root.id
        )
        staging_ids = {value.id for value in dag.state_staging_values}
        other_state_root = next(
            binding
            for binding in schedule.buffer_bindings
            if binding.ownership is not BufferOwnership.ALIASED
            and binding.value_id in staging_ids
            and binding.id != root.id
        )
        cases = {
            "wrong-alias-root": replace(alias, alias_of=other_root.id),
            "cross-state-root": replace(alias, alias_of=other_state_root.id),
            "wrong-view": replace(
                alias,
                tensor_slice=replace(
                    alias.tensor_slice,
                    shape=(alias.tensor_slice.shape[0], alias.tensor_slice.shape[1] - 1),
                ),
            ),
            "wrong-owner": replace(alias, core_id=alias.core_id + 1),
        }
        for name, tampered_alias in cases.items():
            with self.subTest(name):
                bindings = tuple(
                    tampered_alias if item.id == alias.id else item
                    for item in schedule.buffer_bindings
                )
                tampered = _rebuild_schedule(
                    schedule, buffer_bindings=bindings
                )
                with self.assertRaises(SchemaError):
                    tampered.validate_against(
                        dag,
                        replica.projected.graph,
                    )

        manifest = replica.projected.graph.persistent_state_manifest
        assert manifest is not None
        trainable = next(
            declaration
            for declaration in manifest.declarations
            if declaration.identity.kind is StateKind.TRAINABLE_PARAMETER
        )
        with self.subTest("wrong-permission"):
            with self.assertRaises(SchemaError):
                replace(
                    trainable, access=PersistentStateAccess.READ_ONLY
                ).validate("trainable")
        with self.subTest("wrong-kind"):
            with self.assertRaises(SchemaError):
                replace(
                    trainable,
                    identity=replace(
                        trainable.identity,
                        kind=StateKind.PARAMETER,
                    ),
                ).validate("trainable")


if __name__ == "__main__":
    unittest.main()
