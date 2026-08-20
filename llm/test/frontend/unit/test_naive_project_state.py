from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.action import (
    FusionPlan,
    StandaloneCollectivePlan,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, OpKind
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import (
    IR2ProjectionResult,
    IntraDieDAG,
    OriginKind,
    RegionLowering,
    SemanticTaskKind,
    StateIoOrigin,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateManifest,
)

from test_naive_inter_die import _partitioned_graph


def _ordinary_parameter_tp1() -> tuple[IR1, object, object]:
    source = _partitioned_graph(tp=1)
    node = next(item for item in source.nodes if item.kind is OpKind.GEMM)
    access = next(
        item for item in source.state_accesses if item.node_ref == node.id
    )
    manifest = source.persistent_state_manifest
    assert manifest is not None
    declaration = next(
        item for item in manifest.declarations if item.id == access.state_ref
    )
    value_ids = set(node.inputs + node.outputs)
    values = tuple(
        replace(
            value,
            producer=node.id if value.id in node.outputs else None,
            consumers=(node.id,) if value.id in node.inputs else (),
        )
        for value in source.values
        if value.id in value_ids
    )
    local_manifest = PersistentStateManifest.create(
        address_spaces=manifest.address_spaces,
        declarations=(declaration,),
        bindings=tuple(
            binding
            for binding in manifest.bindings
            if binding.state_ref == declaration.id
        ),
    )
    fields = source._semantic_key()
    fields.update(
        instances=tuple(
            replace(instance, node_ids=(node.id,))
            if instance.id == node.instance_id
            else replace(instance, node_ids=())
            for instance in source.instances
        ),
        nodes=(node,),
        values=values,
        edges=(),
        fusion_candidates=(),
        fused_op_skeletons=(),
        state_accesses=(access,),
        persistent_state_manifest=local_manifest,
    )
    graph = IR1.create(producer_pass="fusion_partition", **fields)
    graph.validate()
    return graph, access, declaration


def _ordinary_kv_tp1() -> tuple[IR1, tuple[object, ...], tuple[object, ...]]:
    source = _partitioned_graph(tp=1)
    node = next(item for item in source.nodes if item.kind is OpKind.ATTENTION)
    accesses = tuple(
        item for item in source.state_accesses if item.node_ref == node.id
    )
    manifest = source.persistent_state_manifest
    assert manifest is not None
    access_state_ids = {access.state_ref for access in accesses}
    declarations = tuple(
        item
        for item in manifest.declarations
        if item.id in access_state_ids
    )
    value_ids = set(node.inputs + node.outputs)
    values = tuple(
        replace(
            value,
            producer=node.id if value.id in node.outputs else None,
            consumers=(node.id,) if value.id in node.inputs else (),
        )
        for value in source.values
        if value.id in value_ids
    )
    local_manifest = PersistentStateManifest.create(
        address_spaces=manifest.address_spaces,
        declarations=declarations,
        bindings=tuple(
            binding
            for binding in manifest.bindings
            if binding.state_ref in access_state_ids
        ),
    )
    fields = source._semantic_key()
    fields.update(
        instances=tuple(
            replace(instance, node_ids=(node.id,))
            if instance.id == node.instance_id
            else replace(instance, node_ids=())
            for instance in source.instances
        ),
        nodes=(node,),
        values=values,
        edges=(),
        fusion_candidates=(),
        fused_op_skeletons=(),
        state_accesses=accesses,
        persistent_state_manifest=local_manifest,
    )
    graph = IR1.create(producer_pass="fusion_partition", **fields)
    graph.validate()
    return graph, accesses, declarations


def _complete_stateful_projection(
    tp: int,
) -> tuple[
    IR1,
    tuple[FusionPlan, ...],
    tuple[StandaloneCollectivePlan, ...],
    IR2ProjectionResult,
]:
    graph = _partitioned_graph(tp=tp)
    fusion_plans = tuple(
        NaiveInterDiePolicy().plan(graph, skeleton, graph.profile)
        for skeleton in graph.fused_op_skeletons
    )
    fused_members = {
        member_id
        for skeleton in graph.fused_op_skeletons
        for member_id in skeleton.member_node_ids
    }
    standalone_plans = tuple(
        DirectAllGatherPolicy().plan(graph, node, graph.profile)
        for node in graph.nodes
        if node.id not in fused_members
        and node.kind is OpKind.COLLECTIVE
        and getattr(node.workload, "collective", None)
        is CollectiveKind.ALL_GATHER
    )
    projection = NaiveProjectToIR2().run(
        graph,
        fusion_plans,
        standalone_plans,
        state_transfers=(),
    )
    return graph, fusion_plans, standalone_plans, projection


def _replace_dag(
    projection: IR2ProjectionResult,
    source_dag: IntraDieDAG,
    **updates: object,
) -> IR2ProjectionResult:
    dag_fields = source_dag._semantic_key()
    dag_fields.update(updates)
    replacement = IntraDieDAG.create(
        producer_pass=source_dag.producer_pass,
        **dag_fields,
    )
    projection_fields = projection._semantic_key()
    projection_fields["dags"] = tuple(
        replacement if dag.id == source_dag.id else dag
        for dag in projection.dags
    )
    return IR2ProjectionResult.create(
        producer_pass=projection.producer_pass,
        **projection_fields,
    )


class NaiveProjectStateTest(unittest.TestCase):
    def test_tp1_parameter_dma_feeds_ordinary_compute(self) -> None:
        graph, access, declaration = _ordinary_parameter_tp1()
        policy = NaiveProjectToIR2()
        projection = policy.run(graph, (), (), state_transfers=())
        self.assertEqual(
            policy.run(graph, (), (), state_transfers=()), projection
        )
        projection.validate_against(graph, (), ())
        dag = next(item for item in projection.dags if item.state_access_ids)
        self.assertEqual(dag.source_state_manifest_id, graph.persistent_state_manifest.id)
        self.assertEqual(dag.state_access_ids, (access.id,))
        self.assertEqual(len(dag.state_staging_values), 1)
        staging = dag.state_staging_values[0]
        self.assertEqual(staging.shape, declaration.shape)
        self.assertEqual(staging.state_ref, declaration.id)
        dma = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.DMA_IN
        )
        comp = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.COMP
        )
        self.assertEqual(dma.bytes, declaration.tensor_bytes)
        self.assertEqual(dma.write_values, (staging.id,))
        self.assertIn(staging.id, comp.read_values)
        self.assertIn(dma.id, comp.deps)
        self.assertNotIn(declaration.identity.tensor_ref, comp.read_values)
        self.assertIs(
            next(region for region in dag.regions if dma.id in region.task_ids).lowering,
            RegionLowering.STRICT_STATE_IO,
        )

    def test_parameter_rejects_forged_compute_substitution(self) -> None:
        graph, _access, declaration = _ordinary_parameter_tp1()
        projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
        dag = next(item for item in projection.dags if item.state_access_ids)
        comp = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.COMP
        )
        assert comp.compute is not None
        forged_inputs = tuple(
            replace(
                operand,
                value_id=(
                    declaration.identity.tensor_ref
                    if operand.value_id == dag.state_staging_values[0].id
                    else operand.value_id
                ),
            )
            for operand in comp.compute.inputs
        )
        forged = replace(
            comp,
            read_values=tuple(operand.value_id for operand in forged_inputs),
            compute=replace(comp.compute, inputs=forged_inputs),
        )
        bad = _replace_dag(
            projection,
            dag,
            tasks=tuple(
                forged if task.id == comp.id else task for task in dag.tasks
            ),
        )
        with self.assertRaises(SchemaError):
            bad.validate_against(graph, (), ())

    def test_state_task_reorder_is_rejected(self) -> None:
        graph, _access, _declaration = _ordinary_parameter_tp1()
        projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
        dag = next(item for item in projection.dags if item.state_access_ids)
        bad = _replace_dag(
            projection,
            dag,
            tasks=tuple(reversed(dag.tasks)),
        )
        with self.assertRaises(SchemaError):
            bad.validate_against(graph, (), ())

    def test_fake_state_io_origin_is_rejected(self) -> None:
        graph, _access, _declaration = _ordinary_parameter_tp1()
        projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
        dag = next(item for item in projection.dags if item.state_access_ids)
        dma = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.DMA_IN
        )
        forged_dma = replace(
            dma,
            origin_ref=StateIoOrigin(
                OriginKind.STATE_IO,
                "fake-access",
                dma.origin_ref.node_ref,
                dma.origin_ref.rank,
            ),
        )
        bad = _replace_dag(
            projection,
            dag,
            tasks=tuple(
                forged_dma if task.id == dma.id else task for task in dag.tasks
            ),
        )
        with self.assertRaises(SchemaError):
            bad.validate_against(graph, (), ())

    def test_tp1_kv_is_opaque_write_only_state_around_attention(self) -> None:
        graph, accesses, declarations = _ordinary_kv_tp1()
        original_attention = graph.nodes[0]
        projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
        projection.validate_against(graph, (), ())
        dag = next(item for item in projection.dags if item.state_access_ids)
        self.assertEqual(
            dag.state_access_ids,
            tuple(access.id for access in accesses),
        )
        self.assertEqual(len(dag.state_staging_values), 2)
        self.assertEqual(
            sum(
                task.bytes
                for task in dag.tasks
                if task.kind is SemanticTaskKind.DMA_IN
            ),
            0,
        )
        self.assertEqual(
            sum(
                task.bytes
                for task in dag.tasks
                if task.kind is SemanticTaskKind.DMA_OUT
            ),
            sum(declaration.tensor_bytes for declaration in declarations),
        )
        comp = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.COMP
        )
        self.assertEqual(comp.read_values, original_attention.inputs)
        self.assertEqual(
            tuple(operand.value_id for operand in comp.compute.inputs),
            original_attention.inputs,
        )
        for staging in dag.state_staging_values:
            self.assertNotIn(comp.id, staging.consumer_tasks)
            self.assertEqual(
                tuple(
                    task.kind
                    for task in dag.tasks
                    if task.id in staging.consumer_tasks
                ),
                (SemanticTaskKind.DMA_OUT,),
            )

    def test_tp2_rank1_parameter_uses_global_contained_dma_view(self) -> None:
        graph, fusion_plans, standalone_plans, projection = (
            _complete_stateful_projection(tp=2)
        )
        projection.validate_against(
            graph,
            fusion_plans,
            standalone_plans,
        )
        manifest = graph.persistent_state_manifest
        assert manifest is not None
        declarations = {
            declaration.id: declaration
            for declaration in manifest.declarations
        }
        values = {value.id: value for value in graph.values}
        accesses = {access.id: access for access in graph.state_accesses}
        observed = 0
        for dag in projection.dags:
            staging_by_access = {
                value.state_access_ref: value
                for value in dag.state_staging_values
            }
            for task in dag.tasks:
                if (
                    task.kind is not SemanticTaskKind.DMA_IN
                    or not isinstance(task.origin_ref, StateIoOrigin)
                    or task.origin_ref.rank != 1
                ):
                    continue
                access = accesses[task.origin_ref.state_access_ref]
                declaration = declarations[access.state_ref]
                tensor_ref = declaration.identity.tensor_ref
                if tensor_ref is None:
                    continue
                source = values[tensor_ref]
                staging = staging_by_access[access.id]
                assert task.tensor_slice is not None
                self.assertEqual(staging.shape, source.shape)
                self.assertEqual(
                    staging.logical_layout,
                    source.logical_layout,
                )
                self.assertEqual(task.tensor_slice.shape, declaration.shape)
                self.assertEqual(task.shape, declaration.shape)
                self.assertEqual(task.bytes, declaration.tensor_bytes)
                sharded = any(
                    axis is not None for axis in source.sharding.dim_map
                )
                self.assertEqual(any(task.tensor_slice.offset), sharded)
                self.assertTrue(
                    all(
                        task.tensor_slice.offset[axis]
                        + task.tensor_slice.shape[axis]
                        <= staging.shape[axis]
                        for axis in range(len(staging.shape))
                    )
                )
                observed += int(sharded)
        self.assertEqual(observed, 4)

    def test_tp2_rank1_parameter_rejects_forged_dma_offset(self) -> None:
        graph, fusion_plans, standalone_plans, projection = (
            _complete_stateful_projection(tp=2)
        )
        manifest = graph.persistent_state_manifest
        assert manifest is not None
        declarations = {
            declaration.id: declaration
            for declaration in manifest.declarations
        }
        accesses = {access.id: access for access in graph.state_accesses}
        witness = next(
            (dag, task)
            for dag in projection.dags
            for task in dag.tasks
            if task.kind is SemanticTaskKind.DMA_IN
            and isinstance(task.origin_ref, StateIoOrigin)
            and task.origin_ref.rank == 1
            and task.tensor_slice is not None
            and any(task.tensor_slice.offset)
            and declarations[
                accesses[task.origin_ref.state_access_ref].state_ref
            ].identity.tensor_ref
            is not None
        )
        dag, dma_task = witness
        assert dma_task.tensor_slice is not None
        forged_dma = replace(
            dma_task,
            tensor_slice=replace(
                dma_task.tensor_slice,
                offset=(0,) * len(dma_task.tensor_slice.offset),
            ),
        )
        bad = _replace_dag(
            projection,
            dag,
            tasks=tuple(
                forged_dma if task.id == dma_task.id else task
                for task in dag.tasks
            ),
        )
        with self.assertRaisesRegex(SchemaError, "canonical direction view"):
            bad.validate_against(
                graph,
                fusion_plans,
                standalone_plans,
            )

    def test_tp4_state_region_phase_order_is_exact_with_transit(self) -> None:
        graph, fusion_plans, standalone_plans, projection = (
            _complete_stateful_projection(tp=4)
        )
        projection.validate_against(
            graph,
            fusion_plans,
            standalone_plans,
        )

        witness = None
        for dag in projection.dags:
            task_index = {task.id: task for task in dag.tasks}
            for region_index, region in enumerate(dag.regions):
                if (
                    region_index == 0
                    or region.lowering is not RegionLowering.ISA_REGION
                    or not any(
                        task_index[task_id].kind
                        is SemanticTaskKind.TRANSIT
                        for task_id in region.task_ids
                    )
                ):
                    continue
                inbound = dag.regions[region_index - 1]
                if (
                    inbound.lowering is RegionLowering.STRICT_STATE_IO
                    and len(inbound.task_ids) == 1
                    and task_index[inbound.task_ids[0]].kind
                    is SemanticTaskKind.DMA_IN
                ):
                    witness = (dag, region_index)
                    break
            if witness is not None:
                break
        self.assertIsNotNone(witness)
        assert witness is not None
        dag, region_index = witness
        reordered_regions = list(dag.regions)
        reordered_regions[region_index - 1], reordered_regions[region_index] = (
            reordered_regions[region_index],
            reordered_regions[region_index - 1],
        )
        bad = _replace_dag(
            projection,
            dag,
            regions=tuple(reordered_regions),
        )
        with self.assertRaisesRegex(
            SchemaError,
            "regions must follow canonical",
        ):
            bad.validate_against(
                graph,
                fusion_plans,
                standalone_plans,
            )



if __name__ == "__main__":
    unittest.main()
