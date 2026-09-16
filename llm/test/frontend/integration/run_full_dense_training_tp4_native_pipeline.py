"""Compile real two-step Dense TP4 from source IR0 through native link.

Run from the repository root with PYTHONPATH=. to reproduce every N4/N5/N6 gate.
A successful link is only compile evidence; NpuSim Fresh needs a separate runner.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sys

from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_runtime import verify_full_dense_two_step_linked_native
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_train_forward
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import schedule_train_forward
from llm.frontend.wafer_frontend.passes.load_fabric import hbm_address_spaces_from_data,physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_train_forward
from llm.frontend.wafer_frontend.passes.train_global_action import build_train_global_action
from llm.frontend.wafer_frontend.passes.train_link_program import link_train
from llm.frontend.wafer_frontend.passes.train_lower_program import lower_train
from llm.frontend.wafer_frontend.policies.registry import RegistryKind,production_registry
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext,InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.n5 import ProjectToIR2Context,IntraDieSchedulingContext
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.action import FusionActionKind
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, OpKind, OpPhase
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragments
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.test.frontend.unit.test_flexible_dense_train import _hardware,_spec
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, help='Persist exact linked manifest and compile evidence')
parser.add_argument('--npusim', type=Path)
parser.add_argument('--finalizer', type=Path)
parser.add_argument('--simulation', type=Path)
parser.add_argument('--timeout', type=int, default=900)
args=parser.parse_args()
if any(value is not None for value in (args.npusim, args.finalizer, args.simulation)):
    if args.output is None or any(value is None for value in (
        args.npusim, args.finalizer, args.simulation
    )):
        parser.error('double Fresh requires --output, --npusim, --finalizer, --simulation')
@builder_validation_session()
def run_pipeline(args: argparse.Namespace) -> None:
    p=build_flexible_dense_train_plan(_spec(1,4),RectMeshSpec(1,4))
    print('plan',flush=True)
    graph=build_full_dense_training_two_step_ir0(p)
    assert len(graph.nodes) == 382 and len(graph.state_accesses) == 320
    assert len(graph.persistent_states) == 60
    assert sum(node.kind is OpKind.OPTIMIZER_UPDATE for node in graph.nodes) == 120
    print('source',len(graph.nodes),flush=True)
    hardware=_hardware(1,4);producer='dense_tp4_native_probe'
    placed=place_train_forward_ir0(graph,PlacementContext.create(
        producer_pass=producer,fabric=physical_fabric_from_data(hardware),
        placement=p.source_experiment.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(hardware)))
    print('ir1',sum(len(x.graph.nodes) for x in placed.replicas),flush=True)
    partitioned=partition_train_forward(placed,FusionPartitionContext.create(producer_pass=producer))
    print('partition',flush=True)
    registry=production_registry()
    planned=plan_train_forward(partitioned,InterDiePlanningContext.create(
        producer_pass=producer,
        fused_policy=registry.instantiate(RegistryKind.INTER_DIE,'naive').selection,
        standalone_policy=registry.instantiate(RegistryKind.STANDALONE_COLLECTIVE,'direct_all_gather').selection))
    backward_rs = tuple(
        item for item in planned.replicas[0].standalone_plans
        if next(node for node in planned.replicas[0].graph.nodes
                if node.id == item.op_id).phase is OpPhase.DGRAD
        and next(node for node in planned.replicas[0].graph.nodes
                 if node.id == item.op_id).workload.collective
        is CollectiveKind.REDUCE_SCATTER
    )
    assert len(backward_rs) == 8
    for item in backward_rs:
        actual = Counter(action.kind for program in item.rank_programs
                         for action in program.actions)
        assert actual[FusionActionKind.LOCAL_COPY] == 4
        assert actual[FusionActionKind.SEND] == 12
        assert actual[FusionActionKind.RECV] == 12
        assert actual[FusionActionKind.WAIT] == 12
        assert actual[FusionActionKind.REDUCE] == 4
    print('plan_collective',flush=True)
    projected=project_train_forward(planned,ProjectToIR2Context.create(
        producer_pass=producer,state_transfers=()))
    print('projection',flush=True)
    scheduled=schedule_train_forward(projected,IntraDieSchedulingContext.create(
        producer_pass=producer,
        policy=registry.instantiate(RegistryKind.INTRA_DIE,'naive').selection))
    print('schedule',flush=True)
    dag=build_train_global_action(scheduled)
    print('action_dag',len(dag.replicas),flush=True)
    native=lower_train(dag)
    print('native',flush=True)
    linked=link_train(native)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_file = args.output / "full_tp4_two_step.linked.json"
        manifest_file.write_text(canonical_json(linked.manifest), encoding="utf-8")
        (args.output / "compile_command.json").write_text(
            json.dumps({"cwd": str(Path.cwd()), "argv": [sys.executable, *sys.argv],
                        "mesh": [1, 4], "program_invocations": 0},
                       sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"linked artifact: {manifest_file.resolve()}", flush=True)
    native_opcodes = Counter(record.opcode for fragment in _leaf_fragments(linked.manifest.fragments)
                             for stream in fragment.core_streams for record in stream.records)
    assert native_opcodes[RecordOpcode.DTE_SEND] >= 96
    assert native_opcodes[RecordOpcode.DTE_RECV] >= 96
    assert native_opcodes[RecordOpcode.LOCAL_REDUCE] >= 32
    assert native_opcodes[RecordOpcode.SGD_UPDATE] == 120
    print('linked',len(linked.manifest.fragments),flush=True)

    if args.output is not None:
        receipt = {
            "schema_version": "dense_full_tp4_native_compile/v1",
            "mesh": [1, 4], "steps": 2, "layers": 2,
            "source_nodes": len(graph.nodes),
            "state_accesses": len(graph.state_accesses),
            "parameter_state_shards": len(graph.persistent_states),
            "backward_reduce_scatter_plans": len(backward_rs),
            "native_opcode_counts": {opcode.name: count for opcode, count
                                     in sorted(native_opcodes.items(), key=lambda item: item[0].name)},
            "linked_manifest_id": linked.manifest.id,
            "linked_manifest_digest": canonical_digest(linked.manifest),
            "native_executed": False,
        }
        (args.output / "compile_evidence.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"linked artifact: {manifest_file.resolve()}", flush=True)
    physical = verify_full_dense_two_step_linked_native(linked, p)
    assert len(physical.physical_dag.state_version_edges) == 60
    assert len(physical.requirements.paths) == 120
    assert len(physical.loss_gradient_seed_abi_by_step) == 8
    assert len(physical.physical_dag.transport_edges) >= 96
    if args.output is not None:
        (args.output / "full_tp4_two_step.physical_dag.json").write_text(
            canonical_json(physical.physical_dag), encoding="utf-8")
        receipt.update({
            "physical_gradient_gate": "PASS",
            "physical_actions": len(physical.physical_dag.actions),
            "state_version_edges": len(physical.physical_dag.state_version_edges),
            "transport_edges": len(physical.physical_dag.transport_edges),
            "ce_seed_buffer_abi_by_step_rank": {
                f"step{step}.rank{rank}": abi for (step, rank), abi
                in sorted(physical.loss_gradient_seed_abi_by_step.items())
            },
        })
        (args.output / "compile_evidence.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    from .tp4_inverse_collective_tamper import (
        require_tp4_inverse_collective_tamper_rejections,
    )
    require_tp4_inverse_collective_tamper_rejections(linked, p, physical)
    if args.output is not None:
        receipt["inverse_collective_missing_dte_or_sum_rejected"] = True
        (args.output / "compile_evidence.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if args.npusim is not None:
        from .run_full_dense_training_tp4_native_fresh import run_linked_tp4_fresh
        result = run_linked_tp4_fresh(
            linked, physical, npusim=args.npusim, finalizer=args.finalizer,
            simulation=args.simulation, output=args.output, timeout=args.timeout,
        )
        print(f"source-backed TP4 native double Fresh PASS {result}", flush=True)
    else:
        print("source-backed per-rank TP4 native compile and physical gradient closure completed; NpuSim execution not yet verified", flush=True)


if __name__ == '__main__':
    run_pipeline(args)
