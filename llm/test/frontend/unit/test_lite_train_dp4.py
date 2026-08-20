from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from llm.test.frontend.integration.lite_train_cases import (
    build_s2_lite_source_case,
)

from llm.frontend.wafer_frontend import lowering as public_lowering
from llm.frontend.wafer_frontend import passes as public_passes
from llm.frontend.wafer_frontend import schema as public_schema
from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import lite_train_dp4_linker as dp4_linker_module
from llm.frontend.wafer_frontend.passes import lite_train_dp4 as dp4_pass_module
from llm.frontend.wafer_frontend.passes import lite_train_dp4_link_program as dp4_link_module
from llm.frontend.wafer_frontend.passes import lite_train_dp4_lower_program as dp4_lower_module
from llm.frontend.wafer_frontend.passes import lite_train_dp4_n6 as dp4_n6_pass_module
from llm.frontend.wafer_frontend.schema import lite_train_dp4 as dp4_schema_module
from llm.frontend.wafer_frontend.schema import lite_train_dp4_n6 as dp4_n6_schema_module
from llm.frontend.wafer_frontend.passes.fusion_partition import (
    partition_train_forward,
)
from llm.frontend.wafer_frontend.passes.inter_die_plan import (
    plan_train_forward,
)
from llm.frontend.wafer_frontend.passes.intra_die_schedule import (
    schedule_train_forward,
)
from llm.frontend.wafer_frontend.passes.lite_train_dp4 import (
    build_s2_lite_dp4_tree_ar_global_action,
    build_s2_lite_dp4_tree_ar_source,
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
from llm.frontend.wafer_frontend.schema.lite_train_dp4 import (
    S2_LITE_DP4_TREE_AR_CASE_ID,
    S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION,
    S2_LITE_DP4_TREE_AR_SOURCE_SCHEMA_VERSION,
    S2LiteDp4TreeArGlobalAction,
    S2LiteDp4TreeArSource,
    TreeArFlowKind,
    TreeArReduceKind,
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


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE_4DIE = _ROOT / "notes/frontend/examples/hardware_2x2.json"


def _chain():
    base_case = build_s2_lite_source_case()
    source = build_s2_lite_dp4_tree_ar_source(base_case.logical)
    fabric, spaces = load_physical_fabric_and_hbm_address_spaces(
        _HARDWARE_4DIE,
        base_case.runtime_inputs.source_mapping_path,
    )
    placed = place_train_forward_ir0(
        source.graph,
        PlacementContext.create(
            producer_pass="test_lite_train_dp4",
            fabric=fabric,
            placement=base_case.spec.placement,
            hbm_address_spaces=spaces,
        ),
    )
    partitioned = partition_train_forward(
        placed,
        FusionPartitionContext.create(producer_pass="test_lite_train_dp4"),
    )
    registry = production_registry()
    planned = plan_train_forward(
        partitioned,
        InterDiePlanningContext.create(
            producer_pass="test_lite_train_dp4",
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
            producer_pass="test_lite_train_dp4", state_transfers=()
        ),
    )
    scheduled = schedule_train_forward(
        projected,
        IntraDieSchedulingContext.create(
            producer_pass="test_lite_train_dp4",
            policy=registry.instantiate(
                RegistryKind.INTRA_DIE, "naive"
            ).selection,
        ),
    )
    global_action = build_s2_lite_dp4_tree_ar_global_action(
        source, scheduled
    )
    return (
        source,
        placed,
        partitioned,
        planned,
        projected,
        scheduled,
        global_action,
    )


def _graph_with(source: S2LiteDp4TreeArSource, *, dp: int, tp: int, producer: str) -> IR0:
    graph = source.graph
    instance = replace(
        graph.instances[0],
        parallel=replace(graph.instances[0].parallel, dp=dp, tp=tp),
    )
    return IR0.create(
        producer_pass=producer,
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


class S2LiteDp4TreeArTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chain = _chain()

    def test_public_exports_are_exact(self) -> None:
        schema_symbols = (
            "S2_LITE_DP4_TREE_AR_CASE_ID",
            "S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION",
            "S2_LITE_DP4_TREE_AR_SOURCE_SCHEMA_VERSION",
            "S2LiteDp4ScratchRoot",
            "S2LiteDp4SgdDependency",
            "S2LiteDp4TreeArContract",
            "S2LiteDp4TreeArFlow",
            "S2LiteDp4TreeArGlobalAction",
            "S2LiteDp4TreeArReduce",
            "S2LiteDp4TreeArSource",
            "TreeArFlowKind",
            "TreeArReduceKind",
        )
        n6_schema_symbols = (
            "S2_LITE_DP4_TREE_AR_LINKED_PROGRAM_SCHEMA_VERSION",
            "S2_LITE_DP4_TREE_AR_LOWERED_PROGRAM_SCHEMA_VERSION",
            "S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION",
            "Dp4TreeExecutableKind",
            "Dp4TreeExecutableUnit",
            "S2LiteDp4TreeArLinkedProgram",
            "S2LiteDp4TreeArLoweredProgram",
            "S2LiteDp4TreeArN6Intent",
        )
        for module, names in (
            (dp4_schema_module, schema_symbols),
            (dp4_n6_schema_module, n6_schema_symbols),
        ):
            for name in names:
                with self.subTest(namespace="schema", name=name):
                    self.assertIs(getattr(public_schema, name), getattr(module, name))
        pass_symbols = (
            (dp4_pass_module, "build_s2_lite_dp4_tree_ar_global_action"),
            (dp4_pass_module, "build_s2_lite_dp4_tree_ar_source"),
            (dp4_n6_pass_module, "build_s2_lite_dp4_tree_ar_n6_intent"),
            (dp4_lower_module, "lower_s2_lite_dp4_tree_ar"),
            (dp4_link_module, "link_s2_lite_dp4_tree_ar"),
        )
        for module, name in pass_symbols:
            with self.subTest(namespace="passes", name=name):
                self.assertIs(getattr(public_passes, name), getattr(module, name))
        self.assertIs(
            public_lowering.link_s2_lite_dp4_tree_ar_manifest,
            dp4_linker_module.link_s2_lite_dp4_tree_ar_manifest,
        )

    def test_exact_dp4_counts_tree_and_dependency_lineage(self) -> None:
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
        self.assertEqual(placed.dp_degree, 4)
        self.assertEqual(
            tuple(
                replica.graph.groups[0].placements[0].die_id
                for replica in placed.replicas
            ),
            (0, 1, 2, 3),
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
            (184, 116, 64, 4),
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
            (184, 192, 396, 68),
        )
        self.assertEqual(
            tuple(flow.kind for flow in result.tree_flows),
            (
                TreeArFlowKind.UPLOAD_1_TO_0,
                TreeArFlowKind.UPLOAD_3_TO_2,
                TreeArFlowKind.PARTIAL_2_TO_0,
                TreeArFlowKind.BROADCAST_0_TO_1,
                TreeArFlowKind.BROADCAST_0_TO_2,
                TreeArFlowKind.BROADCAST_2_TO_3,
            ),
        )
        self.assertEqual(
            tuple(reduce.kind for reduce in result.tree_reduces),
            (
                TreeArReduceKind.PAIR_01,
                TreeArReduceKind.PAIR_23,
                TreeArReduceKind.GLOBAL_AT_0,
            ),
        )
        self.assertEqual(
            (
                result.tree_contract.case_id,
                result.tree_contract.replica_die_ids,
                result.tree_contract.gradient_dtype.value,
                result.tree_contract.gradient_bytes,
                result.tree_contract.logical_flow_bytes,
                result.tree_contract.reduce_input_count,
                result.tree_contract.reduce_element_count,
                result.tree_contract.input_stride_bytes,
                result.tree_contract.reduce_source_span_bytes,
                result.tree_contract.reduce_destination_span_bytes,
                result.tree_contract.alignment_bytes,
                tuple(root.die_id for root in result.tree_contract.scratch_roots),
                tuple(root.span_bytes for root in result.tree_contract.scratch_roots),
                sum(len(dag.actions) for dag in result.local_dags),
                len(result.tree_flows),
                len(result.tree_reduces),
                len(result.sgd_dependencies),
            ),
            (
                S2_LITE_DP4_TREE_AR_CASE_ID,
                (0, 1, 2, 3),
                "fp32",
                2048,
                12288,
                2,
                512,
                2048,
                4096,
                2048,
                64,
                (0, 2),
                (4096, 4096),
                184,
                6,
                3,
                4,
            ),
        )
        upload10, upload32, partial20, broadcast01, broadcast02, broadcast23 = (
            result.tree_flows
        )
        reduce01, reduce23, global0 = result.tree_reduces
        self.assertEqual(reduce01.deps[1], upload10.id)
        self.assertEqual(reduce23.deps[1], upload32.id)
        self.assertEqual(partial20.deps, (reduce23.id,))
        self.assertEqual(global0.deps, (reduce01.id, partial20.id))
        self.assertEqual(broadcast01.deps, (global0.id,))
        self.assertEqual(broadcast02.deps, (global0.id,))
        self.assertEqual(broadcast23.deps, (broadcast02.id,))
        self.assertEqual(
            tuple(item.depends_on_step_ref for item in result.sgd_dependencies),
            (global0.id, broadcast01.id, broadcast02.id, broadcast23.id),
        )

    def test_strict_roundtrip_versions_and_ids(self) -> None:
        source, *_middle, result = self.chain
        self.assertEqual(
            S2_LITE_DP4_TREE_AR_SOURCE_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_dp4_tree_ar_source/v1alpha1",
        )
        self.assertEqual(
            S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_dp4_tree_ar_global_action/v1alpha1",
        )
        self.assertEqual(
            loads_dataclass(
                S2LiteDp4TreeArSource,
                canonical_json(source),
                path="source",
            ),
            source,
        )
        self.assertEqual(
            loads_dataclass(
                S2LiteDp4TreeArGlobalAction,
                canonical_json(result),
                path="global_action",
            ),
            result,
        )
        self.assertEqual(source.id, "s2_lite_dp4_tree_ar_source_eea0a80f0823f69e")
        self.assertEqual(
            result.id,
            "s2_lite_dp4_tree_ar_global_action_b1adc0ee038f3639",
        )

    def test_tree_and_replica_tamper_fail_closed(self) -> None:
        result = self.chain[-1]
        cases = {
            "old-version": replace(
                result,
                schema_version=(
                    "wafer_frontend.s2_lite_dp4_tree_ar_global_action/v1alpha0"
                ),
            ),
            "swapped-dies": replace(
                result,
                scheduled=replace(
                    result.scheduled,
                    replicas=(
                        result.scheduled.replicas[1],
                        result.scheduled.replicas[0],
                        *result.scheduled.replicas[2:],
                    ),
                ),
            ),
            "missing-tree-edge": replace(
                result, tree_flows=result.tree_flows[:-1]
            ),
            "changed-root": replace(
                result,
                tree_contract=replace(
                    result.tree_contract, replica_die_ids=(1, 0, 2, 3)
                ),
            ),
            "wrong-reduce-dep": replace(
                result,
                tree_reduces=(
                    *result.tree_reduces[:2],
                    replace(
                        result.tree_reduces[2],
                        deps=(result.tree_reduces[0].id,),
                    ),
                ),
            ),
            "duplicate-channel": replace(
                result,
                tree_flows=(
                    result.tree_flows[0],
                    replace(
                        result.tree_flows[1],
                        channel_ref=result.tree_flows[0].channel_ref,
                    ),
                    *result.tree_flows[2:],
                ),
            ),
            "missing-sgd-dependency": replace(
                result, sgd_dependencies=result.sgd_dependencies[:-1]
            ),
        }
        for name, tampered in cases.items():
            with self.subTest(name):
                with self.assertRaises(SchemaError):
                    tampered.validate()

    def test_geometry_gate_rejects_dp3_dp5_tp2_and_fake_producer(self) -> None:
        source = self.chain[0]
        cases = (
            ("dp3", 3, 1, "s2_lite_dp4_tree_ar_source", "DP=4"),
            ("dp5", 5, 1, "s2_lite_dp4_tree_ar_source", "DP=4"),
            ("tp2", 4, 2, "s2_lite_dp4_tree_ar_source", "TP=PP=EP=1"),
            ("fake-producer", 4, 1, "logical_expand", "DP=1"),
        )
        for name, dp, tp, producer, message in cases:
            with self.subTest(name):
                with self.assertRaisesRegex(SchemaError, message):
                    DenseIR0Validator.validate(
                        _graph_with(
                            source,
                            dp=dp,
                            tp=tp,
                            producer=producer,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
