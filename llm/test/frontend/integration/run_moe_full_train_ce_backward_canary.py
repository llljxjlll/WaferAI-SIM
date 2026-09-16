"""Native EP1 MoE two-layer forward plus one genuine seeded CE backward edge.

This is a partial backward canary; it cannot satisfy full training or SGD.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import struct

from llm.frontend.wafer_frontend.lowering.context import LoweringContext
from llm.frontend.wafer_frontend.lowering.linker import NaiveManifestLinker
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.global_action import build_global_action_dag
from llm.frontend.wafer_frontend.passes.lower_program import (
    _lower_fragments, _resolve_dependencies,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ce_backward_ir0 import (
    append_moe_full_train_ce_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_head_backward_ir0 import (
    append_moe_full_train_head_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_ir1_source import (
    build_moe_ep_placed_ir1_candidate,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_route_program_io import (
    bind_full_moe_route_state_program_io,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_route_table_source import (
    build_moe_full_train_route_table_source,
)
from llm.frontend.wafer_frontend.passes.placement import _physical_node
from llm.frontend.wafer_frontend.passes.program_io import (
    MoeFullTrainForwardLinkedSource, _resolved_abis_prevalidated,
    _resolved_state_abis, build_timing_program_io,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import NaiveProjectToIR2
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership, StateUseAccess
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.test.frontend.integration.run_moe_full_train_two_step_forward_canary import (
    _run, _sha,
)
from llm.test.frontend.integration.flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from llm.test.frontend.integration.run_moe_full_model_sequence_runtime_canary import (
    _bind_native_hardware_to_fabric,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--head-backward", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    finalizer, resolver, npusim = (args.finalizer.resolve(),
                                   args.resolver.resolve(), args.npusim.resolve())
    Fixture.setUpClass()
    receipts = []
    native_context = None
    for step in (0, 1):
        phase, sequence, placement, physical = (
            build_single_die_moe_train_physical_source(Fixture, step=step)
        )
        native_context = physical
        source = append_moe_full_train_ce_backward_ir0(phase)
        if args.head_backward:
            source = append_moe_full_train_head_backward_ir0(source)
        base = build_moe_ep_placed_ir1_candidate(
            phase, original_dense=Fixture.dense, sequence=sequence,
            placement=placement, context=physical,
            dense_manifest=Fixture.manifest,
        ).physical_ir1
        reverse = source.nodes[len(phase.graph.nodes):]
        instance = replace(base.instances[0], node_ids=(
            *base.instances[0].node_ids, *(node.id for node in reverse),
        ))
        ir1 = IR1.create(
            producer_pass="placement", source_ir0_id=source.id,
            profile=source.profile, fabric=physical.fabric,
            instances=(instance,), groups=base.groups,
            nodes=(*base.nodes, *(
                _physical_node(node, base.groups[0].id) for node in reverse)),
            values=source.values, edges=source.edges,
            fusion_candidates=source.fusion_candidates,
            state_accesses=source.state_accesses,
            persistent_state_manifest=base.persistent_state_manifest,
            instance_profiles=source.instance_profiles,
            node_profiles=source.node_profiles, pd_plan_id=source.pd_plan_id,
        )
        ir1.validate()
        graph = partition_ir1(ir1)
        projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
        schedules = NaiveIntraDiePolicy().schedule(projection, graph)
        dag = build_global_action_dag(graph, projection, schedules)
        context = LoweringContext(graph, (), (), projection, schedules, dag)
        leaves = _lower_fragments(
            context, _resolve_dependencies(None, None, None, None, None),
        )
        if len(leaves) != (55 if args.head_backward else 52):
            raise RuntimeError("reverse path physical leaf count drifted")
        manifest = NaiveManifestLinker().link(context, leaves)
        manifest.validate_against(
            context.ir1, context.fusion_plans, context.standalone_plans,
            context.projection, context.schedule_set, context.global_dag,
            manifest.fragments,
        )
        opcodes = [record.opcode for fragment in manifest.fragments
                   for stream in fragment.core_streams for record in stream.records]
        if (opcodes.count(RecordOpcode.CROSS_ENTROPY_BACKWARD) != 1
                or opcodes.count(RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING) != int(args.head_backward)
                or opcodes.count(RecordOpcode.GEMM_DX_TIMING) != int(args.head_backward)):
            raise RuntimeError("linked program lacks exact CE/LM-head reverse records")
        linked = output / f"step{step}.linked.json"
        artifact = output / f"step{step}.npup"
        sidecar = output / f"step{step}.program_io.json"
        linked.write_text(canonical_json(manifest))
        _run([str(finalizer), "--input", str(linked), "--output",
              str(artifact), "--report", str(output / f"step{step}.finalizer.json")],
             cwd=output, log=output / f"step{step}.finalizer.log")
        carrier = MoeFullTrainForwardLinkedSource(manifest, context)
        carrier.validate()
        routes = build_moe_full_train_route_table_source(phase, sequence)
        route_by_state = {phase.route_state_refs[seed.layer]: seed.payload
                          for seed in routes.seeds}
        state_seeds = {}
        for item in _resolved_state_abis(carrier):
            if item.first_access is not StateUseAccess.READ:
                continue
            abi = item.abi
            if abi.kind is StateKind.MOE_STATIC_ROUTE:
                state_seeds[abi.state_ref] = route_by_state[abi.state_ref]
            elif abi.dtype is DType.FP16 and abi.size_bytes % 2 == 0:
                state_seeds[abi.state_ref] = struct.pack("<e", 0.0625) * (
                    abi.size_bytes // 2
                )
            else:
                raise RuntimeError(f"unseeded physical state: {abi.state_ref}")
        gradient = next(value for value in source.values
                        if value.id == f"{source.instances[0].id}.loss_gradient")
        dloss = [item.abi for item in _resolved_abis_prevalidated(carrier)
                 if item.abi.value_id == gradient.id
                 and item.abi.ownership is BufferOwnership.BORROWED]
        if len(dloss) != 1 or dloss[0].size_bytes != gradient.shape[0] * 4:
            raise RuntimeError("independent CE dLoss lacks one physical seed ABI")
        seed = struct.pack("<f", 1.0 / gradient.shape[0]) * gradient.shape[0]
        base_io = build_timing_program_io(
            carrier, _sha(artifact), state_seed_overrides=state_seeds,
            sram_seed_overrides={dloss[0].id: seed},
        )
        contract = bind_full_moe_route_state_program_io(
            manifest, base_io, routes, phase, sequence, placement,
            original_dense=Fixture.dense, dense_manifest=Fixture.manifest,
            context=physical,
        )
        sidecar.write_text(canonical_json(contract))
        _run([str(resolver), "--resolve", str(linked), str(artifact),
              str(sidecar)], cwd=output,
             log=output / f"step{step}.resolver.log")
        receipts.append(dict(
            step=step, ir0=source.id, linked=manifest.id,
            artifact_sha256=_sha(artifact), program_io_sha256=_sha(sidecar),
            leaves=len(leaves), records=sum(len(stream.records)
                for stream in manifest.core_streams),
            ce_backward_records=opcodes.count(RecordOpcode.CROSS_ENTROPY_BACKWARD),
            head_wgrad_records=opcodes.count(RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING),
            head_dx_records=opcodes.count(RecordOpcode.GEMM_DX_TIMING),
            dloss_seed_abi=dloss[0].id,
        ))
    assert native_context is not None
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    _bind_native_hardware_to_fabric(hardware, native_context.fabric)
    hardware["memory"]["sram_size"] = 131072
    hardware["memory"]["sram"]["capacity_bytes"] = 131072
    hardware["memory"]["sram"]["regions"] = [dict(
        name="sram", base_bytes=0, size_bytes=131072, allocator="block",
        spillable=False, access=["compute", "dte", "lsu", "legacy", "noc_rx"],
    )]
    space = native_context.hbm_address_spaces[0]
    for stack in hardware["memory_system"]["hbm_stacks"]:
        stack["capacity_bytes"] = space.size_bytes
    hardware["memory_system"]["address_policy"]["home_ranges"] = [dict(
        die_id=0, base=space.base_address, size_bytes=space.size_bytes,
    )]
    hardware["memory_system"]["address_policy"]["stack_interleave_bytes"] = space.size_bytes
    hardware_path = output / "hardware.json"
    hardware_path.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")))
    mapping = output / "mapping.spec"
    mapping.write_text("0:0\n")
    simulation = Path(__file__).resolve().parents[3] / "test/program/p5_behavioral_simulation.json"
    for step in (0, 1):
        log = output / f"step{step}.npusim.log"
        _run([str(npusim), "--program-one-shot", "--program",
              str(output / f"step{step}.npup"), "--linked-manifest",
              str(output / f"step{step}.linked.json"), "--program-io",
              str(output / f"step{step}.program_io.json"),
              "--hardware-config", str(hardware_path),
              "--simulation-config", str(simulation),
              "--mapping-config", str(mapping), "--trace-window", "1000000"],
             cwd=npusim.parent, log=log)
        content = log.read_text()
        if (content.count("[PROGRAM_IO] phase=verify") != 1
                or content.count("[PROGRAM_MEMORY] core=0") != 1
                or "[CREDIT] data_balanced=1 ctrl_balanced=1" not in content
                or "[DRAIN] d2d_link_residual=0" not in content):
            raise RuntimeError(f"step{step} native CE backward audit failed")
        receipts[step]["npusim_log_sha256"] = _sha(log)
    repo = Path(__file__).resolve().parents[4]
    source_files = (
        "llm/frontend/wafer_frontend/passes/moe_full_train_ce_backward_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_head_backward_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_forward_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_ep_ir1_source.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_ep_placement.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_route_program_io.py",
        "llm/frontend/wafer_frontend/passes/program_io.py",
        "llm/test/frontend/integration/run_moe_full_train_ce_backward_canary.py",
        "llm/unittest/npusim.cpp",
    )
    source_sha256 = {name: _sha(repo / name) for name in source_files}
    (output / "receipt.json").write_text(json.dumps(dict(
        source_file_sha256=source_sha256,
        status=("head_backward_physical_partial" if args.head_backward
                else "ce_backward_physical_partial"),
        full_training_gate="closed", steps=receipts,
        finalizer_sha256=_sha(finalizer), resolver_sha256=_sha(resolver),
        npusim_sha256=_sha(npusim), hardware_sha256=_sha(hardware_path),
        simulation_sha256=_sha(simulation),
    ), sort_keys=True, indent=2))
    print(json.dumps(receipts, sort_keys=True))


if __name__ == "__main__":
    main()
