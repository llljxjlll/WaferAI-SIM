"""Native EP1 MoE two-layer forward and source-bound partial reverse canary.

Timing-only reverse primitives do not establish numerical gradient production or
full training. Every mode keeps the full-training gate closed.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

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
from llm.frontend.wafer_frontend.passes.moe_full_train_shared_reverse_ir0 import (
    append_moe_full_train_shared_reverse_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_combine_backward_ir0 import (
    append_moe_full_train_combine_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_wgrad_ir0 import (
    append_moe_full_train_router_wgrad_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_sgd_ir0 import (
    append_moe_full_train_router_sgd_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_expert_backward_ir0 import (
    append_moe_full_train_expert_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_dx_ir0 import (
    append_moe_full_train_router_dx_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_input_gradient_ir0 import (
    append_moe_full_train_input_gradient_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_backbone_ir0 import (
    append_moe_full_train_layer1_backbone_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_parameter_sgd_ir0 import (
    append_moe_full_train_layer1_parameter_sgd_ir0,
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
from llm.frontend.wafer_frontend.schema.ir0 import StateAccessMode
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership, StateUseAccess
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind, PersistentStateAccess
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
    parser.add_argument("--shared-reverse", action="store_true")
    parser.add_argument("--combine-backward", action="store_true")
    parser.add_argument("--router-wgrad", action="store_true")
    parser.add_argument("--router-sgd", action="store_true")
    parser.add_argument("--expert-backward", action="store_true")
    parser.add_argument("--router-dx", action="store_true")
    parser.add_argument("--input-gradient", action="store_true")
    parser.add_argument("--layer1-backbone", action="store_true")
    parser.add_argument("--layer1-parameter-sgd", action="store_true")
    args = parser.parse_args()
    layer1_backbone_mode = args.layer1_backbone or args.layer1_parameter_sgd
    input_gradient_mode = args.input_gradient or layer1_backbone_mode
    router_dx_mode = args.router_dx or input_gradient_mode
    expert_mode = args.expert_backward or router_dx_mode
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    finalizer, resolver, npusim = (args.finalizer.resolve(),
                                   args.resolver.resolve(), args.npusim.resolve())
    repo_root = Path(__file__).resolve().parents[4]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source_tree_clean_at_entry = not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root, text=True).strip()
    def imported_python_hashes() -> dict[str, str]:
        result = {}
        for module in tuple(sys.modules.values()):
            source_path = getattr(module, "__file__", None)
            if not source_path or not source_path.endswith(".py"):
                continue
            path = Path(source_path).resolve()
            if path.is_relative_to(repo_root) and path.is_file():
                result[str(path.relative_to(repo_root))] = _sha(path)
        return dict(sorted(result.items()))
    imported_at_entry = imported_python_hashes()
    tool_hashes_at_entry = {"finalizer": _sha(finalizer),
                            "resolver": _sha(resolver), "npusim": _sha(npusim)}
    Fixture.setUpClass()
    receipts = []
    native_context = None
    for step in (0, 1):
        phase, sequence, placement, physical = (
            build_single_die_moe_train_physical_source(Fixture, step=step)
        )
        native_context = physical
        source = append_moe_full_train_ce_backward_ir0(phase)
        if args.head_backward or args.shared_reverse or args.combine_backward or args.router_wgrad or args.router_sgd or expert_mode:
            source = append_moe_full_train_head_backward_ir0(source)
        if args.shared_reverse or args.combine_backward or args.router_wgrad or args.router_sgd or expert_mode:
            source = append_moe_full_train_shared_reverse_ir0(source)
        if args.combine_backward or args.router_wgrad or args.router_sgd or expert_mode:
            source = append_moe_full_train_combine_backward_ir0(source)
        if args.router_wgrad or args.router_sgd or expert_mode:
            source = append_moe_full_train_router_wgrad_ir0(source)
        if args.router_sgd:
            source = append_moe_full_train_router_sgd_ir0(source, sequence)
        if expert_mode:
            source = append_moe_full_train_expert_backward_ir0(source)
        if router_dx_mode:
            source = append_moe_full_train_router_dx_ir0(source)
        if input_gradient_mode:
            source = append_moe_full_train_input_gradient_ir0(source)
        if layer1_backbone_mode:
            source = append_moe_full_train_layer1_backbone_ir0(source)
        if args.layer1_parameter_sgd:
            source = append_moe_full_train_layer1_parameter_sgd_ir0(
                source, sequence)
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
        if len(leaves) != (79 if args.layer1_parameter_sgd else 67 if layer1_backbone_mode else 64 if input_gradient_mode else 63 if router_dx_mode else 61 if expert_mode else 63 if args.router_sgd else 60 if args.router_wgrad else
                           59 if args.combine_backward else
                           58 if args.shared_reverse else
                           55 if args.head_backward else 52):
            raise RuntimeError(f"reverse path physical leaf count drifted: {len(leaves)}")
        manifest = NaiveManifestLinker().link(context, leaves)
        manifest.validate_against(
            context.ir1, context.fusion_plans, context.standalone_plans,
            context.projection, context.schedule_set, context.global_dag,
            manifest.fragments,
        )
        opcodes = [record.opcode for fragment in manifest.fragments
                   for stream in fragment.core_streams for record in stream.records]
        if (opcodes.count(RecordOpcode.CROSS_ENTROPY_BACKWARD) != 1
                or opcodes.count(RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING) != (5 if expert_mode else 2 if args.router_wgrad or args.router_sgd else int(args.head_backward or args.shared_reverse or args.combine_backward))
                or opcodes.count(RecordOpcode.GEMM_DX_TIMING) != (5 if router_dx_mode else 4 if expert_mode else int(args.head_backward or args.shared_reverse or args.combine_backward or args.router_wgrad or args.router_sgd))
                or opcodes.count(RecordOpcode.NORM_GAMMA_WGRAD_TIMING) != (2 if layer1_backbone_mode else int(args.shared_reverse or args.combine_backward or args.router_wgrad or args.router_sgd or expert_mode))
                or opcodes.count(RecordOpcode.RMSNORM_BACKWARD_TIMING) != (2 if layer1_backbone_mode else int(args.shared_reverse or args.combine_backward or args.router_wgrad or args.router_sgd or expert_mode))
                or opcodes.count(RecordOpcode.RESIDUAL_BACKWARD_TIMING) != int(args.shared_reverse or args.combine_backward or args.router_wgrad or args.router_sgd or expert_mode)
                or opcodes.count(RecordOpcode.MOE_SCORE_WEIGHT_BACKWARD) != int(args.combine_backward or args.router_wgrad or args.router_sgd or expert_mode)
                or opcodes.count(RecordOpcode.SGD_UPDATE) != (4 if args.layer1_parameter_sgd else int(args.router_sgd))
                or opcodes.count(RecordOpcode.SWIGLU_BACKWARD_TIMING) != int(expert_mode)
                or opcodes.count(RecordOpcode.LOCAL_REDUCE) != int(expert_mode)):
            raise RuntimeError(f"linked program reverse opcode counts: {[(item.name, opcodes.count(item)) for item in set(opcodes)]}")
        linked = output / f"step{step}.linked.json"
        artifact = output / f"step{step}.npup"
        sidecar = output / f"step{step}.program_io.json"
        linked.write_text(canonical_json(manifest))
        _run([str(finalizer), "--input", str(linked), "--output",
              str(artifact), "--report", str(output / f"step{step}.finalizer.json")],
             cwd=output, log=output / f"step{step}.finalizer.log")
        carrier = MoeFullTrainForwardLinkedSource(manifest, context)
        carrier.validate()
        state_write = None
        layer1_state_writes = ()
        if args.layer1_parameter_sgd:
            update_nodes = source.nodes[-4:]
            state_refs = tuple(next(access.state_ref for access in source.state_accesses
                                    if access.node_ref == node.id
                                    and access.mode is StateAccessMode.READ_WRITE)
                               for node in update_nodes)
            matches = tuple(item for item in _resolved_state_abis(carrier)
                            if item.abi.state_ref in state_refs)
            if (len(state_refs) != 4 or len(set(state_refs)) != 4
                    or len(matches) != 4
                    or {item.abi.state_ref for item in matches} != set(state_refs)
                    or sorted(item.abi.size_bytes for item in matches) != [8, 64, 64, 64]
                    or any(item.abi.access is not PersistentStateAccess.READ_WRITE
                           or item.first_access is not StateUseAccess.READ
                           or sum(access is StateUseAccess.WRITE
                                  for _index, access in item.uses) != 1
                           for item in matches)
                    or opcodes.count(RecordOpcode.LSU_STORE) != 4):
                raise RuntimeError("four MoE parameter SGD states lack exact physical writes")
            layer1_state_writes = matches
        if args.router_sgd:
            gate_state_ref = source.nodes[-2].workload.source_parameter_state_ref
            matches = [item for item in _resolved_state_abis(carrier)
                       if item.abi.state_ref == gate_state_ref]
            if (len(matches) != 1
                    or matches[0].abi.access is not PersistentStateAccess.READ_WRITE
                    or matches[0].first_access is not StateUseAccess.READ
                    or sum(access is StateUseAccess.WRITE
                           for _index, access in matches[0].uses) != 1
                    or matches[0].abi.size_bytes != 8
                    or opcodes.count(RecordOpcode.LSU_STORE) != 1):
                raise RuntimeError("router SGD lacks exact real gate HBM write")
            state_write = matches[0]
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
        expert_probe_count = 0
        expert_probe_nonzero_expected_bytes = 0
        if expert_mode:
            sidecar_payload = json.loads(sidecar.read_text())
            blobs = {item["id"]: base64.b64decode(item["bytes_base64"])
                     for item in sidecar_payload["blobs"]}
            expert_probes = tuple(item for item in sidecar_payload["output_probes"]
                                  if item["target"]["value_id"].startswith(
                                      "backward::T0.layer1.moe.expert0."))
            if len(expert_probes) != (0 if args.layer1_parameter_sgd else
                                      3 if input_gradient_mode else 4):
                raise RuntimeError("expert reverse lacks exact physical output probes")
            if input_gradient_mode:
                terminal_ids = (
                    ("backward::T0.layer1.residual1.merge.input_gradient",
                     next(node.outputs[0] for node in source.nodes
                          if node.id.startswith("backward::T0.layer1.norm2::")))
                    if layer1_backbone_mode else
                    ("backward::T0.layer1.moe.input_sum.norm2_gradient",)
                )
                for ref in terminal_ids:
                    matches = tuple(item for item in sidecar_payload["output_probes"]
                                    if item["target"]["value_id"] == ref)
                    if len(matches) != 1:
                        raise RuntimeError(f"MoE shared gradient lacks one physical output probe: {ref}")
            expert_probe_count = len(expert_probes)
            expert_probe_nonzero_expected_bytes = sum(
                byte != 0 for item in expert_probes
                for byte in blobs[item["blob_ref"]])
        _run([str(resolver), "--resolve", str(linked), str(artifact),
              str(sidecar)], cwd=output,
             log=output / f"step{step}.resolver.log")
        receipts.append(dict(
            step=step, ir0=source.id, linked=manifest.id,
            artifact_sha256=_sha(artifact), program_io_sha256=_sha(sidecar),
            leaves=len(leaves), records=sum(len(stream.records)
                for stream in manifest.core_streams),
            ce_backward_records=opcodes.count(RecordOpcode.CROSS_ENTROPY_BACKWARD),
            wgrad_records=opcodes.count(RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING),
            head_dx_records=opcodes.count(RecordOpcode.GEMM_DX_TIMING),
            final_norm_dx_records=opcodes.count(RecordOpcode.RMSNORM_BACKWARD_TIMING),
            dcombined_records=opcodes.count(RecordOpcode.RESIDUAL_BACKWARD_TIMING),
            combine_backward_records=opcodes.count(RecordOpcode.MOE_SCORE_WEIGHT_BACKWARD),
            sgd_records=opcodes.count(RecordOpcode.SGD_UPDATE),
            expert_gradient_probes=expert_probe_count,
            expert_gradient_nonzero_expected_bytes=expert_probe_nonzero_expected_bytes,
            numeric_gradient_witness=False,
            gate_hbm_write_state_ref=(state_write.abi.state_ref
                                      if state_write else None),
            gate_hbm_write_bytes=(state_write.abi.size_bytes
                                  if state_write else 0),
            layer1_sgd_state_refs=sorted(item.abi.state_ref
                                         for item in layer1_state_writes),
            layer1_sgd_write_bytes=sum(item.abi.size_bytes
                                       for item in layer1_state_writes),
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
        if (args.layer1_parameter_sgd
                and (content.count("[TRAIN_SGD]") != 4
                     or "lsu_hbm_write_bytes=200" not in content)):
            raise RuntimeError(f"step{step} four MoE SGD/state writes did not execute")
        if (args.router_sgd
                and (content.count("[TRAIN_SGD]") != 1
                     or "lsu_hbm_write_bytes=8" not in content)):
            raise RuntimeError(f"step{step} router SGD/state write did not execute")
        if (content.count("[PROGRAM_IO] phase=verify") != 1
                or content.count("[PROGRAM_MEMORY] core=0") != 1
                or "[CREDIT] data_balanced=1 ctrl_balanced=1" not in content
                or "[DRAIN] d2d_link_residual=0" not in content):
            raise RuntimeError(f"step{step} native CE backward audit failed")
        receipts[step]["npusim_log_sha256"] = _sha(log)
    sequence_log_sha256 = None
    if args.router_sgd:
        sequence_log = output / "router_sgd_partial_sequence.npusim.log"
        _run([
            str(npusim),
            "--program-sequence", ",".join(str(output / f"step{step}.npup")
                                            for step in (0, 1)),
            "--linked-manifest-sequence", ",".join(
                str(output / f"step{step}.linked.json") for step in (0, 1)),
            "--program-io-sequence", ",".join(
                str(output / f"step{step}.program_io.json") for step in (0, 1)),
            "--moe-router-sgd-partial-sequence",
            "--hardware-config", str(hardware_path),
            "--simulation-config", str(simulation),
            "--mapping-config", str(mapping), "--trace-window", "1000000",
        ], cwd=npusim.parent, log=sequence_log)
        content = sequence_log.read_text()
        required = (
            "[MOE_ROUTER_SGD_PARTIAL_STATE] version=0 bytes=952",
            "[MOE_ROUTER_SGD_PARTIAL_STATE] version=1 bytes=952",
            "[MOE_ROUTER_SGD_PARTIAL_STATE] version=2 bytes=952",
            "[MOE_ROUTER_SGD_PARTIAL_INPUT] index=1 prior_store_completed=1 same_hbm_state=1",
            "[MOE_ROUTER_SGD_PARTIAL_SEQUENCE_STEP] index=0 input_version=0 output_version=1 trainable_states=19 route_states=2 records=254 sgd=1 store=1",
            "[MOE_ROUTER_SGD_PARTIAL_SEQUENCE_STEP] index=1 input_version=1 output_version=2 trainable_states=19 route_states=2 records=254 sgd=1 store=1",
            "[DENSE_SEQUENCE_DRAIN] segments=2 one_shot=1",
            "lsu_hbm_write_bytes=16",
        )
        if (any(content.count(marker) != 1 for marker in required)
                or content.count("[TRAIN_SGD]") != 2
                or "[CREDIT] data_balanced=1 ctrl_balanced=1" not in content
                or "[DRAIN] d2d_link_residual=0" not in content):
            raise RuntimeError("MoE router SGD partial native sequence audit failed")
        sequence_log_sha256 = _sha(sequence_log)
    layer1_sequence_log_sha256 = None
    if args.layer1_parameter_sgd:
        sequence_log = output / "layer1_sgd_partial_sequence.npusim.log"
        _run([
            str(npusim),
            "--program-sequence", ",".join(str(output / f"step{step}.npup")
                                           for step in (0, 1)),
            "--linked-manifest-sequence", ",".join(
                str(output / f"step{step}.linked.json") for step in (0, 1)),
            "--program-io-sequence", ",".join(
                str(output / f"step{step}.program_io.json") for step in (0, 1)),
            "--moe-layer1-sgd-partial-sequence",
            "--hardware-config", str(hardware_path),
            "--simulation-config", str(simulation),
            "--mapping-config", str(mapping), "--trace-window", "1000000",
        ], cwd=npusim.parent, log=sequence_log)
        content = sequence_log.read_text()
        required = (
            "[MOE_LAYER1_SGD_PARTIAL_STATE] version=0 bytes=952",
            "[MOE_LAYER1_SGD_PARTIAL_STATE] version=1 bytes=952",
            "[MOE_LAYER1_SGD_PARTIAL_STATE] version=2 bytes=952",
            "[MOE_LAYER1_SGD_PARTIAL_INPUT] index=1 prior_store_completed=1 same_hbm_state=1",
            "[MOE_LAYER1_SGD_PARTIAL_SEQUENCE_STEP] index=0 input_version=0 output_version=1 trainable_states=19 route_states=2 records=340 sgd=4 store=4",
            "[MOE_LAYER1_SGD_PARTIAL_SEQUENCE_STEP] index=1 input_version=1 output_version=2 trainable_states=19 route_states=2 records=340 sgd=4 store=4",
            "[DENSE_SEQUENCE_PROGRAM_IO] index=0 probes=5 pass=1",
            "[DENSE_SEQUENCE_PROGRAM_IO] index=1 probes=5 pass=1",
            "[DENSE_SEQUENCE_DRAIN] segments=2 one_shot=1",
            "lsu_hbm_read_bytes=2896 lsu_hbm_write_bytes=400",
        )
        state_versions = re.findall(
            r"\[MOE_LAYER1_SGD_PARTIAL_STATE\] version=([012]) bytes=952 "
            r"digest=([0-9a-f]{64}) content_changed=0 functional=0 "
            r"full_training=0 pass=1", content)
        steps = re.findall(
            r"\[MOE_LAYER1_SGD_PARTIAL_SEQUENCE_STEP\] index=([01]) .*?"
            r"state_digest_before=([0-9a-f]{64}) "
            r"state_digest_after=([0-9a-f]{64}) "
            r"full_training=0 functional=0 pass=1", content)
        input_match = re.findall(
            r"\[MOE_LAYER1_SGD_PARTIAL_INPUT\] index=1 "
            r"prior_store_completed=1 same_hbm_state=1 "
            r"digest=([0-9a-f]{64}) pass=1", content)
        if (any(content.count(marker) != 1 for marker in required)
                or content.count("[TRAIN_SGD]") != 8
                or tuple(version for version, _digest in state_versions)
                   != ("0", "1", "2")
                or tuple(index for index, _before, _after in steps)
                   != ("0", "1")
                or len(input_match) != 1
                or steps[0][1:] != (state_versions[0][1], state_versions[1][1])
                or steps[1][1:] != (state_versions[1][1], state_versions[2][1])
                or input_match[0] != state_versions[1][1]
                or "[CREDIT] data_balanced=1 ctrl_balanced=1" not in content
                or "[DRAIN] d2d_link_residual=0" not in content):
            raise RuntimeError("MoE layer1 four-SGD partial native sequence audit failed")
        layer1_sequence_log_sha256 = _sha(sequence_log)
    repo = repo_root
    source_files = (
        "llm/frontend/wafer_frontend/passes/moe_full_train_ce_backward_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_head_backward_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_shared_reverse_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_combine_backward_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_router_wgrad_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_router_sgd_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_expert_backward_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_router_dx_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_input_gradient_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_layer1_backbone_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_layer1_parameter_sgd_ir0.py",
        "llm/frontend/wafer_frontend/schema/moe_expert_backward_workload.py",
        "llm/frontend/wafer_frontend/schema/moe_expert_backward_record_check.py",
        "llm/frontend/wafer_frontend/schema/moe_expert_scratch.py",
        "llm/frontend/wafer_frontend/lowering/moe_full_train_expert_backward.py",
        "llm/frontend/wafer_frontend/lowering/lifecycle.py",
        "llm/frontend/wafer_frontend/schema/moe_combine_backward_workload.py",
        "llm/frontend/wafer_frontend/schema/ir0.py",
        "llm/frontend/wafer_frontend/schema/ir1.py",
        "llm/frontend/wafer_frontend/schema/ir2.py",
        "llm/frontend/wafer_frontend/schema/action.py",
        "llm/frontend/wafer_frontend/schema/artifact_manifest.py",
        "llm/frontend/wafer_frontend/lowering/moe_full_train_combine_backward.py",
        "llm/frontend/wafer_frontend/lowering/linker.py",
        "llm/frontend/wafer_frontend/passes/lower_program.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_forward_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_ep_ir1_source.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_ep_placement.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_route_program_io.py",
        "llm/frontend/wafer_frontend/passes/program_io.py",
        "llm/src/frontend/program_finalizer.cpp",
        "llm/test/frontend/integration/run_moe_full_train_ce_backward_canary.py",
        "llm/unittest/npusim.cpp",
    )
    source_sha256 = {name: _sha(repo / name) for name in source_files}
    imported_at_exit = imported_python_hashes()
    if any(imported_at_exit.get(name) != digest
           for name, digest in imported_at_entry.items()):
        raise RuntimeError("imported Python source drifted during native run")
    tool_hashes_at_exit = {"finalizer": _sha(finalizer),
                           "resolver": _sha(resolver), "npusim": _sha(npusim)}
    if tool_hashes_at_exit != tool_hashes_at_entry:
        raise RuntimeError("native tool bytes drifted during run")
    (output / "receipt.json").write_text(json.dumps(dict(
        source_file_sha256=source_sha256,
        source_commit=source_commit,
        source_tree_clean_at_entry=source_tree_clean_at_entry,
        imported_python_sha256_at_entry=imported_at_entry,
        imported_python_sha256_at_exit=imported_at_exit,
        native_tool_sha256=tool_hashes_at_entry,
        runner_cwd=str(Path.cwd().resolve()),
        native_runtime_cwd=str(npusim.parent),
        status=("layer1_parameter_sgd_physical_partial" if args.layer1_parameter_sgd else
                "layer1_backbone_physical_partial" if layer1_backbone_mode else
                "input_gradient_physical_partial" if input_gradient_mode else
                "router_dx_physical_partial" if router_dx_mode else
                "expert_backward_physical_partial" if expert_mode else
                "router_sgd_physical_partial" if args.router_sgd else
                "router_wgrad_physical_partial" if args.router_wgrad else
                "combine_backward_physical_partial" if args.combine_backward else
                "shared_dcombined_physical_partial" if args.shared_reverse else
                "head_backward_physical_partial" if args.head_backward else
                "ce_backward_physical_partial"),
        full_training_gate="closed", steps=receipts,
        router_sgd_partial_sequence_log_sha256=sequence_log_sha256,
        layer1_sgd_partial_sequence_log_sha256=layer1_sequence_log_sha256,
        finalizer_sha256=_sha(finalizer), resolver_sha256=_sha(resolver),
        npusim_sha256=_sha(npusim), hardware_sha256=_sha(hardware_path),
        simulation_sha256=_sha(simulation),
    ), sort_keys=True, indent=2))
    print(json.dumps(receipts, sort_keys=True))


if __name__ == "__main__":
    main()
