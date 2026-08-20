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
from llm.frontend.wafer_frontend.passes.lite_train_dp2 import (
    build_s2_lite_dp2_rooted_ar_global_action,
    build_s2_lite_dp2_rooted_ar_source,
)
from llm.frontend.wafer_frontend.passes import (
    build_s2_lite_dp2_rooted_ar_global_action as public_build_rooted_ar_global,
    build_s2_lite_dp2_rooted_ar_source as public_build_rooted_ar_source,
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
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.schema.ir0 import IR0
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind
from llm.frontend.wafer_frontend.schema.lite_train_dp2 import (
    RootedArStepKind,
    S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION,
    S2_LITE_DP2_ROOTED_AR_SOURCE_SCHEMA_VERSION,
    S2LiteDp2RootedArGlobalAction,
    S2LiteDp2RootedArSource,
    S2LiteRootedArContract,
    S2LiteRootedArScratchSlice,
    S2LiteRootedArStep,
    S2LiteSgdDependency,
)
from llm.frontend.wafer_frontend.schema import (
    RootedArStepKind as PublicRootedArStepKind,
    S2LiteDp2RootedArGlobalAction as PublicRootedArGlobalAction,
    S2LiteDp2RootedArSource as PublicRootedArSource,
    S2LiteRootedArContract as PublicRootedArContract,
    S2LiteRootedArScratchSlice as PublicRootedArScratchSlice,
    S2LiteRootedArStep as PublicRootedArStep,
    S2LiteSgdDependency as PublicSgdDependency,
)
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    IntraDieSchedulingContext,
    ProjectToIR2Context,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)


def _chain():
    base_case = build_s2_lite_source_case()
    source = build_s2_lite_dp2_rooted_ar_source(base_case.logical)
    fabric, spaces = load_physical_fabric_and_hbm_address_spaces(
        base_case.runtime_inputs.source_hardware_path,
        base_case.runtime_inputs.source_mapping_path,
    )
    placed = place_train_forward_ir0(
        source.graph,
        PlacementContext.create(
            producer_pass="test_lite_train_dp2",
            fabric=fabric,
            placement=base_case.spec.placement,
            hbm_address_spaces=spaces,
        ),
    )
    partitioned = partition_train_forward(
        placed,
        FusionPartitionContext.create(producer_pass="test_lite_train_dp2"),
    )
    registry = production_registry()
    planned = plan_train_forward(
        partitioned,
        InterDiePlanningContext.create(
            producer_pass="test_lite_train_dp2",
            fused_policy=registry.instantiate(
                RegistryKind.INTER_DIE, "naive"
            ).selection,
            standalone_policy=registry.instantiate(
                RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
            ).selection,
        ),
    )
    projected = project_train_forward(
        planned,
        ProjectToIR2Context.create(
            producer_pass="test_lite_train_dp2", state_transfers=()
        ),
    )
    scheduled = schedule_train_forward(
        projected,
        IntraDieSchedulingContext.create(
            producer_pass="test_lite_train_dp2",
            policy=registry.instantiate(
                RegistryKind.INTRA_DIE, "naive"
            ).selection,
        ),
    )
    global_action = build_s2_lite_dp2_rooted_ar_global_action(
        source, scheduled
    )
    return source, placed, partitioned, planned, projected, scheduled, global_action


class S2LiteDp2RootedArTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chain = _chain()

    def test_exact_two_die_counts_bytes_and_dependency_lineage(self) -> None:
        source, placed, _partitioned, _planned, projected, scheduled, result = (
            self.chain
        )
        self.assertEqual(
            (
                len(source.graph.nodes),
                len(source.graph.values),
                len(source.graph.edges),
                len(source.graph.persistent_states),
                len(source.graph.state_accesses),
            ),
            (29, 47, 34, 15, 16),
        )
        self.assertEqual(placed.dp_degree, 2)
        self.assertEqual(
            tuple(
                replica.graph.groups[0].placements[0].die_id
                for replica in placed.replicas
            ),
            (0, 1),
        )
        tasks = tuple(
            task
            for replica in projected.replicas
            for dag in replica.projection.dags
            for task in dag.tasks
        )
        self.assertEqual(
            (
                len(tasks),
                sum(task.kind is SemanticTaskKind.COMP for task in tasks),
                sum(task.kind is SemanticTaskKind.DMA_IN for task in tasks),
                sum(task.kind is SemanticTaskKind.DMA_OUT for task in tasks),
            ),
            (92, 58, 32, 2),
        )
        schedules = tuple(
            schedule
            for replica in scheduled.replicas
            for schedule in replica.schedule_set.schedules
        )
        self.assertEqual(
            (
                sum(len(item.placements) for item in schedules),
                sum(len(item.buffer_bindings) for item in schedules),
                sum(len(item.task_buffer_uses) for item in schedules),
                sum(len(item.task_state_uses) for item in schedules),
            ),
            (92, 96, 198, 34),
        )
        self.assertEqual(
            tuple(step.kind for step in result.ar_steps),
            (
                RootedArStepKind.UPLOAD,
                RootedArStepKind.ROOT_REDUCE,
                RootedArStepKind.DOWNLOAD,
            ),
        )
        self.assertEqual(
            (
                result.ar_contract.root_die_id,
                result.ar_contract.gradient_dtype.value,
                result.ar_contract.gradient_bytes,
                result.ar_contract.root_scratch_bytes,
                result.ar_contract.rank_input_offsets,
                result.ar_contract.input_stride_bytes,
                result.ar_contract.alignment_bytes,
                sum(len(dag.actions) for dag in result.local_dags),
                len(result.ar_steps),
            ),
            (0, "fp32", 2048, 4096, (0, 2048), 2048, 64, 92, 3),
        )
        upload, reduce, download = result.ar_steps
        self.assertEqual(reduce.deps[1], upload.id)
        self.assertEqual(download.deps, (reduce.id,))
        self.assertEqual(
            tuple(item.depends_on_step_ref for item in result.sgd_dependencies),
            (reduce.id, download.id),
        )

    def test_public_exports_are_exact(self) -> None:
        self.assertIs(public_build_rooted_ar_source, build_s2_lite_dp2_rooted_ar_source)
        self.assertIs(public_build_rooted_ar_global, build_s2_lite_dp2_rooted_ar_global_action)
        self.assertIs(PublicRootedArStepKind, RootedArStepKind)
        self.assertIs(PublicRootedArSource, S2LiteDp2RootedArSource)
        self.assertIs(PublicRootedArGlobalAction, S2LiteDp2RootedArGlobalAction)
        self.assertIs(PublicRootedArContract, S2LiteRootedArContract)
        self.assertIs(PublicRootedArScratchSlice, S2LiteRootedArScratchSlice)
        self.assertIs(PublicRootedArStep, S2LiteRootedArStep)
        self.assertIs(PublicSgdDependency, S2LiteSgdDependency)

    def test_strict_roundtrip_versions_and_tamper(self) -> None:
        source, *_middle, result = self.chain
        self.assertEqual(
            S2_LITE_DP2_ROOTED_AR_SOURCE_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_dp2_rooted_ar_source/v1alpha1",
        )
        self.assertEqual(
            S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_dp2_rooted_ar_global_action/v1alpha1",
        )
        self.assertEqual(
            loads_dataclass(
                S2LiteDp2RootedArSource,
                canonical_json(source),
                path="source",
            ),
            source,
        )
        self.assertEqual(
            loads_dataclass(
                S2LiteDp2RootedArGlobalAction,
                canonical_json(result),
                path="global_action",
            ),
            result,
        )
        with self.subTest("old-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    result,
                    schema_version=(
                        "wafer_frontend.s2_lite_dp2_rooted_ar_global_action/v1alpha0"
                    ),
                ).validate()
        with self.subTest("wrong-root"):
            with self.assertRaisesRegex(SchemaError, "source-exact"):
                replace(
                    result,
                    ar_contract=replace(result.ar_contract, root_die_id=1),
                ).validate()
        with self.subTest("missing-upload-dep"):
            reduce = result.ar_steps[1]
            with self.assertRaisesRegex(SchemaError, "source-exact"):
                replace(
                    result,
                    ar_steps=(
                        result.ar_steps[0],
                        replace(reduce, deps=reduce.deps[:1]),
                        result.ar_steps[2],
                    ),
                ).validate()
        with self.subTest("cross-replica-hbm"):
            first, second = result.local_dags
            first_state = next(
                action for action in first.actions if action.state_uses
            )
            second_index = next(
                index for index, action in enumerate(second.actions) if action.state_uses
            )
            second_action = second.actions[second_index]
            forged = replace(
                second_action,
                state_uses=first_state.state_uses,
            )
            tampered_second = replace(
                second,
                actions=(
                    *second.actions[:second_index],
                    forged,
                    *second.actions[second_index + 1 :],
                ),
            )
            with self.assertRaises(SchemaError):
                replace(result, local_dags=(first, tampered_second)).validate()

    def test_dp_geometry_gate_rejects_dp3_and_fake_producer(self) -> None:
        source = self.chain[0]

        def graph_with(*, dp: int, producer_pass: str) -> IR0:
            graph = source.graph
            instance = replace(
                graph.instances[0],
                parallel=replace(graph.instances[0].parallel, dp=dp),
            )
            return IR0.create(
                producer_pass=producer_pass,
                job=graph.job,
                instances=(instance,),
                nodes=graph.nodes,
                values=graph.values,
                edges=graph.edges,
                fusion_candidates=graph.fusion_candidates,
                profile=graph.profile,
                train=graph.train,
                instance_profiles=graph.instance_profiles,
                node_profiles=graph.node_profiles,
                pd_plan_id=graph.pd_plan_id,
                persistent_states=graph.persistent_states,
                state_accesses=graph.state_accesses,
            )

        with self.subTest("dp3"):
            with self.assertRaisesRegex(SchemaError, "DP=2"):
                DenseIR0Validator.validate(
                    graph_with(
                        dp=3,
                        producer_pass="s2_lite_dp2_rooted_ar_source",
                    )
                )
        with self.subTest("fake-producer"):
            with self.assertRaisesRegex(SchemaError, "DP=1"):
                DenseIR0Validator.validate(
                    graph_with(dp=2, producer_pass="logical_expand")
                )


if __name__ == "__main__":
    unittest.main()
