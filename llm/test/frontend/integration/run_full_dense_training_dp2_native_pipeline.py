"""Compile one real 2x2 TP2/DP2 two-step Dense TRAIN into one linked manifest.

Usage: PYTHONPATH=. python3 -m llm.test.frontend.integration.run_full_dense_training_dp2_native_pipeline --output /tmp/new-empty-result-root
A linked receipt alone is compilation evidence; it does not assert NpuSim execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_train_forward
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import schedule_train_forward
from llm.frontend.wafer_frontend.passes.load_fabric import hbm_address_spaces_from_data, physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_train_forward
from llm.frontend.wafer_frontend.passes.train_global_action import build_train_global_action
from llm.frontend.wafer_frontend.passes.train_link_program import link_train
from llm.frontend.wafer_frontend.passes.train_lower_program import lower_train
from llm.frontend.wafer_frontend.policies.registry import RegistryKind, production_registry
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext, InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.n5 import ProjectToIR2Context, IntraDieSchedulingContext
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragments
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec


@builder_validation_session()
def compile_dp2(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    producer = "dp2_production_n4"
    plan = build_flexible_dense_train_plan(_spec(2, 2), RectMeshSpec(2, 2))
    graph = build_full_dense_training_two_step_ir0(plan)
    if len(graph.persistent_states) != 30:
        raise RuntimeError("DP2 source must retain thirty TP shards before two physical replicas")
    print("SOURCE", len(graph.nodes), flush=True)
    hardware = _hardware(2, 2)
    placement = PlacementContext.create(
        producer_pass=producer,
        fabric=physical_fabric_from_data(hardware),
        placement=plan.source_experiment.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(hardware),
    )
    placed = place_train_forward_ir0(graph, placement)
    partitioned = partition_train_forward(
        placed, FusionPartitionContext.create(producer_pass=producer),
    )
    registry = production_registry()
    planning = InterDiePlanningContext.create(
        producer_pass=producer,
        fused_policy=registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather",
        ).selection,
    )
    planned = plan_train_forward(
        partitioned, planning,
        dense_dp2_plan=plan, dp2_placement_context=placement,
    )
    if len(planned.dp_gradient_routes.gradients) != 60:
        raise RuntimeError("DP2 requires sixty source-bound cross-replica gradients")
    projected = project_train_forward(
        planned,
        ProjectToIR2Context.create(producer_pass=producer, state_transfers=()),
    )
    if len(projected.dp_projected_tasks.tasks) != 480:
        raise RuntimeError("DP2 source/N5 gradient task coverage is incomplete")
    print("PROJECTED", len(projected.dp_projected_tasks.tasks), flush=True)
    scheduled = schedule_train_forward(
        projected,
        IntraDieSchedulingContext.create(
            producer_pass=producer,
            policy=registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection,
        ),
    )
    active_dies = {schedule.die_id for replica in scheduled.replicas
                   for schedule in replica.schedule_set.schedules
                   if schedule.placements}
    if active_dies != {0, 1, 2, 3}:
        raise RuntimeError("DP2 schedule must use four physical dies")
    print("SCHEDULED", len(scheduled.replicas), flush=True)
    global_actions = build_train_global_action(scheduled)
    print("GLOBAL_DAG", len(global_actions.replicas), flush=True)
    native = lower_train(global_actions)
    print("NATIVE", [len(replica.fragments) for replica in native.replicas], flush=True)
    linked = link_train(native)
    manifest = linked.manifest
    manifest_file = output / "full_dp2_two_step.linked.json"
    manifest_file.write_text(canonical_json(manifest), encoding="utf-8")
    print("LINKED", len(manifest.fragments), manifest_file, flush=True)
    counts = Counter(record.opcode for fragment in _leaf_fragments(manifest.fragments)
                     for stream in fragment.core_streams for record in stream.records)
    expected = {
        RecordOpcode.SGD_UPDATE: 120,
        RecordOpcode.CROSS_ENTROPY_FORWARD: 8,
        RecordOpcode.CROSS_ENTROPY_BACKWARD: 8,
    }
    if any(counts[opcode] != count for opcode, count in expected.items()) or (
        counts[RecordOpcode.DTE_SEND] < 120
        or counts[RecordOpcode.DTE_RECV] < 120
        or counts[RecordOpcode.LOCAL_REDUCE] < 60
    ):
        raise RuntimeError("DP2 true DP gradient DTE/SUM/SGD or CE native inventory incomplete")
    receipt = {
        "schema_version": "full_dense_dp2_native_compile/v1",
        "mesh": [2, 2], "tp_degree": 2, "dp_degree": 2,
        "steps": 2, "layers": 2,
        "source_nodes": len(graph.nodes),
        "source_trainable_state_shards": len(graph.persistent_states),
        "expected_physical_replica_state_shards": (
            len(graph.persistent_states) * len(native.replicas)
        ),
        "source_dp_gradient_routes": len(planned.dp_gradient_routes.gradients),
        "dp_gradient_tasks": len(projected.dp_projected_tasks.tasks),
        "physical_dies": sorted(active_dies),
        "lowered_replica_fragments": [len(replica.fragments) for replica in native.replicas],
        "linked_manifest_id": manifest.id,
        "linked_manifest_digest": canonical_digest(manifest),
        "record_counts": {opcode.name: count for opcode, count in
                          sorted(counts.items(), key=lambda item: item[0].name)},
        "native_executed": False,
    }
    (output / "compile_evidence.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    compile_dp2(arguments.output)
