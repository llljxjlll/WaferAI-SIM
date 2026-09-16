"""Run a rectangular MoE Prefill -> Decode -> Decode full model in one instance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import (
    compile_moe_full_model_inference_sequence,
)
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION, ProgramSymbolKind, RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.flexible_moe import MoeRectActionKind, MoeRectFlowStage
from llm.frontend.wafer_frontend.schema.global_action import LogicalCoreRef
from llm.frontend.wafer_frontend.schema.artifact_manifest import SemanticOperandId
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramBlob,
    ProgramHbmTarget,
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramIoTargetKind,
    ProgramOutputCapture,
    ProgramOutputComparison,
    ProgramOutputProbe,
    ProgramSramInitialization,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_moe_compile_sequence import _manifest
from llm.test.frontend.unit.test_moe_full_model_compile_sequence import _legacy_template
from llm.frontend.wafer_frontend.passes.load_fabric import (
    SIMULATOR_PACKET_PAYLOAD_BYTES, physical_fabric_from_data,
)

from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware
from .run_dense_sequence_runtime_canary import (
    _bind_native_hardware_to_fabric, _run,
)


_ROOT = Path(__file__).resolve().parents[4]
_LINKER_SCHEMA = "wafer_frontend.moe_full_model_region_linker/v1alpha1"


def _executable_core_bindings(sequence, rank_count: int) -> dict[int, int]:
    """Return one common rank-to-native-core binding for all three segments."""
    expected_ranks = set(range(rank_count))
    common = None
    for index, segment in enumerate(sequence.segments):
        bindings = segment.executable_manifest.core_bindings
        current = {
            item.logical_core.die_id: item.runtime_core_id
            for item in bindings
            if item.logical_core.local_core_id == 0
        }
        if (len(current) != len(bindings) or set(current) != expected_ranks
                or len(set(current.values())) != rank_count):
            raise RuntimeError(
                f"segment {index} executable core bindings do not cover the mesh exactly"
            )
        if common is None:
            common = current
        elif current != common:
            raise RuntimeError("full-model segment executable core bindings disagree")
    if common is None:
        raise RuntimeError("full-model sequence has no executable segments")
    return common


def _flow_link_expectations(flows, rows: int, columns: int):
    """Accumulate exact X-first native packet and handshake hops for every flow."""
    rank_count = rows * columns
    links: dict[tuple[int, int, str], list[int]] = {}
    for flow in flows:
        source = flow.source_rank
        destination = flow.destination_rank
        if (not 0 <= source < rank_count or not 0 <= destination < rank_count
                or source == destination or flow.logical_bytes <= 0):
            raise RuntimeError(f"invalid remote MoE flow {flow.id}")
        packets = (
            flow.logical_bytes + SIMULATOR_PACKET_PAYLOAD_BYTES - 1
        ) // SIMULATOR_PACKET_PAYLOAD_BYTES
        current = source
        source_x, source_y = source % columns, source // columns
        destination_x, destination_y = (
            destination % columns, destination // columns
        )
        while source_x != destination_x:
            step = 1 if source_x < destination_x else -1
            next_rank = current + step
            direction = "E" if step > 0 else "W"
            counts = links.setdefault((current, next_rank, direction), [0, 0])
            counts[0] += 1
            counts[1] += packets
            current = next_rank
            source_x += step
        while source_y != destination_y:
            step = 1 if source_y < destination_y else -1
            next_rank = current + step * columns
            direction = "N" if step > 0 else "S"
            counts = links.setdefault((current, next_rank, direction), [0, 0])
            counts[0] += 1
            counts[1] += packets
            current = next_rank
            source_y += step
        if current != destination:
            raise RuntimeError(f"failed to route expected MoE flow {flow.id}")
    return {key: tuple(value) for key, value in links.items()}


def _audit_native_runtime(
    stdout: str,
    *,
    sequence,
    expected_flows,
    expert_records_by_core: dict[int, tuple[int, int]],
    core_bindings: dict[int, int],
    rows: int,
    columns: int,
) -> tuple[tuple[str, str, str], ...]:
    """Fail closed on the full sequence, native cores, and routed D2D traffic."""
    segment_markers = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)", stdout
    )
    probes = re.findall(
        r"\[DENSE_SEQUENCE_PROGRAM_IO\] index=(\d+) probes=(\d+) pass=(\d+)",
        stdout,
    )
    drains = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)", stdout
    )
    kv = tuple(re.findall(
        r"\[DENSE_SEQUENCE_KV\] index=(\d+) bytes=(\d+) digest=([0-9a-f]{64}) pass=1",
        stdout,
    ))
    if (segment_markers != [("0", "0"), ("1", "0"), ("2", "1")]
            or probes != [("0", "1", "1"), ("1", "1", "1"), ("2", "1", "1")]
            or drains != [("3", "1")]
            or tuple((index, int(size)) for index, size, _ in kv)
            != (("0", 128), ("1", 192), ("2", 256))):
        raise RuntimeError("full-model sequence/KV closure failed")

    links_expected = _flow_link_expectations(expected_flows, rows, columns)
    expected_request_hops = sum(counts[0] for counts in links_expected.values())
    expected_packet_hops = sum(counts[1] for counts in links_expected.values())
    d2d_data = re.findall(
        r"\[D2D_DATA\] in_pkts=(\d+) out_pkts=(\d+)", stdout
    )
    d2d_type = re.findall(
        r"\[D2D_TYPE\] request_in=(\d+) request_out=(\d+) "
        r"ack_in=(\d+) ack_out=(\d+) data_in=(\d+) data_out=(\d+)",
        stdout,
    )
    expected_type = tuple(map(str, (
        expected_request_hops, expected_request_hops,
        2 * expected_request_hops, 2 * expected_request_hops,
        expected_packet_hops, expected_packet_hops,
    )))
    if (d2d_data != [(str(expected_packet_hops), str(expected_packet_hops))]
            or d2d_type != [expected_type]):
        raise RuntimeError(
            "native aggregate D2D traffic differs from all expected MoE flow hops"
        )
    link_pattern = re.compile(
        r"\[D2D_LINK\] idx=(\d+) die(\d+)->die(\d+) dir=([EWNS]) "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)\."
    )
    links_actual = {}
    for match in link_pattern.finditer(stdout):
        _, source, destination, direction, *raw_counts = match.groups()
        key = (int(source), int(destination), direction)
        if key in links_actual:
            raise RuntimeError(f"duplicate native D2D link evidence: {key}")
        links_actual[key] = tuple(map(int, raw_counts))
    expected_rows = {
        key: (requests, requests, 2 * requests, 2 * requests, packets, packets)
        for key, (requests, packets) in links_expected.items()
    }
    if links_actual != expected_rows:
        raise RuntimeError(
            f"native directed D2D flow closure differs: "
            f"expected={expected_rows!r} actual={links_actual!r}"
        )

    expected_cores = set(core_bindings.values())
    memory_pattern = re.compile(
        r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+) lsu_completed=(\d+) "
        r"[^\n]*lsu_residual=(\d+) dte_residual=(\d+)"
    )
    memory_rows = tuple(tuple(map(int, match.groups()))
                        for match in memory_pattern.finditer(stdout))
    if (len(memory_rows) != len(expected_cores)
            or {row[0] for row in memory_rows} != expected_cores
            or any(issued <= 0 or completed != issued or lsu or dte
                   for _, issued, completed, lsu, dte in memory_rows)):
        raise RuntimeError(
            f"PROGRAM_MEMORY does not close every executable core: {memory_rows!r}"
        )
    p2p = tuple(
        (int(core), int(residual))
        for core, residual in re.findall(
            r"\[P5 P2P DRAIN\] core=(\d+) residual=(\d+)", stdout
        )
    )
    if (len(p2p) != len(expected_cores) or {core for core, _ in p2p} != expected_cores
            or any(residual for _, residual in p2p)):
        raise RuntimeError(f"P2P drain does not close every executable core: {p2p!r}")

    for core, (expected_matmuls, expected_swiglus) in expert_records_by_core.items():
        matmuls = len(re.findall(
            rf"Core {core} start compute primitive Matmul_f\.", stdout
        ))
        swiglus = len(re.findall(
            rf"Core {core} start compute primitive swiglu_forward\.", stdout
        ))
        if matmuls != expected_matmuls or swiglus != expected_swiglus:
            raise RuntimeError(
                f"remote expert core {core} compute differs: "
                f"expected={(expected_matmuls, expected_swiglus)} "
                f"actual={(matmuls, swiglus)}"
            )

    bridge_transfers = 2 * sum(
        len(segment.moe_unit_refs) for segment in sequence.segments
    )
    bridge_core = core_bindings[0]
    bridge_stats = re.findall(
        rf"\[DTE_STATS\] core={bridge_core} issued=(\d+) completed=(\d+) "
        r".*pending=(\d+) active=(\d+) inflight=(\d+)", stdout
    )
    drain_markers = (
        "[P5 P2P TIMING DRAIN] residual=0",
        "[DRAIN] router_residual=0",
        "[DRAIN] d2d_link_residual=0",
        "[CREDIT] data_balanced=1 ctrl_balanced=1",
    )
    expected_bridge = tuple(map(str, (
        bridge_transfers, bridge_transfers, 0, 0, 0,
    )))
    if (bridge_stats != [expected_bridge]
            or any(stdout.count(marker) != 1 for marker in drain_markers)
            or stdout.count("[SIM_RESULT]") != 1
            or "[PROTO_WAIT]" in stdout or "[D2D_BEHA]" in stdout):
        raise RuntimeError("full-model native completion/drain closure failed")
    return kv


def prove_full_model_dataflow(segment, units) -> None:
    """Fail when physical dispatch, expert, combine, or Dense bridges drift."""
    manifest = segment.executable_manifest
    buffers = {item.id: item for fragment in manifest.fragments for item in fragment.buffer_abi}
    fragments = {item.id: item for item in manifest.fragments}
    bindings = {
        (item.fragment_id, item.logical_core, item.fragment_record_index, item.operand_id): item
        for item in manifest.address_operand_bindings
    }
    relocations = {
        (fragment.id, stream.logical_core, relocation.record_index, relocation.operand_id): relocation
        for fragment in manifest.fragments for stream in fragment.core_streams
        for relocation in stream.address_relocations
    }
    records = {}
    for core in manifest.core_streams:
        for ref in core.records:
            stream = next(item for item in fragments[ref.fragment_id].core_streams if item.logical_core == core.logical_core)
            record = stream.records[ref.fragment_record_index]
            records.setdefault((ref.source_global_action_id, core.logical_core, record.opcode), []).append((ref, record))

    def endpoint(action_id, core, opcode, operand_id):
        choices = records.get((action_id, core, opcode), ())
        if len(choices) != 1:
            raise RuntimeError(f"MoE dataflow lacks exact action record: {action_id} {opcode.name}")
        ref, record = choices[0]
        binding = bindings.get((ref.fragment_id, core, ref.fragment_record_index, operand_id))
        if binding is None or len(binding.buffer_abi_ids) != 1:
            raise RuntimeError(f"MoE dataflow lacks exact SRAM endpoint: {action_id} {operand_id.name}")
        return buffers[binding.buffer_abi_ids[0]], record

    source = manifest.source_global_dag_id
    for unit in units:
        layer = unit.layer
        actions = {
            action.id: stable_artifact_id(
                "moe_full_model_action",
                {"source": source, "layer": layer, "action": action.id},
                schema_version=_LINKER_SCHEMA,
            ) for action in unit.plan.actions
        }
        for flow in unit.plan.flows:
            if flow.stage not in (MoeRectFlowStage.DISPATCH, MoeRectFlowStage.COMBINE):
                continue
            send = next(item for item in unit.plan.actions if item.kind is MoeRectActionKind.SEND and item.flow_ref == flow.id)
            recv = next(item for item in unit.plan.actions if item.kind is MoeRectActionKind.RECV and item.flow_ref == flow.id)
            source_abi, source_record = endpoint(actions[send.id], LogicalCoreRef(flow.source_rank, 0), RecordOpcode.DTE_SEND, SemanticOperandId.SOURCE_ADDRESS)
            target_abi, target_record = endpoint(actions[recv.id], LogicalCoreRef(flow.destination_rank, 0), RecordOpcode.DTE_RECV, SemanticOperandId.DESTINATION_ADDRESS)
            expected_source = "activation" if flow.stage is MoeRectFlowStage.DISPATCH else "output"
            expected_target = "activation" if flow.stage is MoeRectFlowStage.DISPATCH else "output"
            for abi, rank, suffix in ((source_abi, flow.source_rank, expected_source), (target_abi, flow.destination_rank, expected_target)):
                if abi.value_id != f"moe_full_model.layer{layer}.flexible_moe.value.rank{rank}.{suffix}":
                    raise RuntimeError(f"MoE {flow.stage.value} SRAM endpoint disagrees with expert dataflow")
            if min(source_abi.size_bytes, target_abi.size_bytes) < flow.logical_bytes or any(
                next(item.literal_value for item in record.operands if item.name == "length_bytes") != flow.logical_bytes
                for record in (source_record, target_record)
            ):
                raise RuntimeError(f"MoE {flow.stage.value} physical payload differs from P2")
        for action in unit.plan.actions:
            if action.kind is MoeRectActionKind.GATE:
                core = LogicalCoreRef(action.rank, 0)
                if not action.assignment_refs:
                    if records.get((actions[action.id], core, RecordOpcode.MATMUL)):
                        raise RuntimeError("zero-work gate emitted phantom MATMUL")
                    continue
                activation, gate_record = endpoint(actions[action.id], core, RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_INPUT_ADDRESS)
                gate_weight, _ = endpoint(actions[action.id], core, RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_DATA_ADDRESS)
                params = next(item.literal_value for item in gate_record.operands if item.name == "parameters")
                if (activation.value_id != f"moe_full_model.layer{layer}.flexible_moe.value.rank{action.rank}.activation"
                        or not gate_weight.value_id.startswith(f"moe_full_model.layer{layer}.flexible_moe.value.rank{action.rank}.state.")
                        or tuple(params[1:]) != (len(action.assignment_refs), unit.spec.hidden_size, unit.spec.expert_count)
                        or action.flops != 2 * params[1] * params[2] * params[3]
                        or gate_weight.size_bytes < 2 * params[2] * params[3]):
                    raise RuntimeError("physical router GATE violates P2 operation count and H×E weight footprint")
            if action.kind is MoeRectActionKind.WEIGHTED_COMBINE and action.assignment_refs:
                combined, _ = endpoint(actions[action.id], LogicalCoreRef(action.rank, 0), RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS)
                if combined.value_id != f"moe_full_model.layer{layer}.flexible_moe.value.rank{action.rank}.output":
                    raise RuntimeError("physical weighted combine does not read returned expert workspace")
            if action.kind is not MoeRectActionKind.EXPERT_FORWARD:
                continue
            core = LogicalCoreRef(action.rank, 0)
            m, h, intermediate = len(action.assignment_refs), unit.spec.hidden_size, unit.spec.intermediate_size
            projection_ids = tuple(stable_artifact_id(
                "moe_full_model_action",
                {"source": source, "layer": layer, "action": stable_artifact_id(
                    "flexible_moe_expert_projection_action",
                    {"plan": unit.plan.id, "expert": action.id, "stage": stage},
                    schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
                )},
                schema_version=_LINKER_SCHEMA,
            ) for stage in ("up", "swiglu", "down"))
            matmuls = tuple(record for physical_action_id in (
                actions[action.id], projection_ids[0], projection_ids[2],
            ) for record in records.get((physical_action_id, core, RecordOpcode.MATMUL), ()))
            swiglus = records.get((projection_ids[1], core, RecordOpcode.SWIGLU), ())
            if m == 0:
                if action.flops != 0 or matmuls or swiglus:
                    raise RuntimeError("empty expert rank emitted phantom expert compute")
                continue
            if len(matmuls) != 3 or len(swiglus) != 1:
                raise RuntimeError("MoE expert lacks exact gate/up/down MATMUL and SwiGLU records")
            matrix_bytes = 2 * h * intermediate
            projection_bytes = 2 * m * intermediate
            expert_flops = 0
            def physical_operand(ref, operand_id):
                key = (ref.fragment_id, core, ref.fragment_record_index, operand_id)
                binding = bindings.get(key)
                relocation = relocations.get(key)
                if binding is None or relocation is None or len(binding.buffer_abi_ids) != 1:
                    raise RuntimeError("MoE expert projection has no exact physical SRAM view")
                return buffers[binding.buffer_abi_ids[0]], binding.tensor_slices[0], relocation.addend
            for index, (ref, record) in enumerate(matmuls):
                params = tuple(next(item.literal_value for item in record.operands if item.name == "parameters"))
                expected = (1, m, h, intermediate) if index < 2 else (1, m, intermediate, h)
                inbound, _, _ = physical_operand(ref, SemanticOperandId.COMPUTE_INPUT_ADDRESS)
                weight, weight_view, weight_addend = physical_operand(ref, SemanticOperandId.COMPUTE_DATA_ADDRESS)
                outbound, output_view, output_addend = physical_operand(ref, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
                source_suffix = "activation" if index < 2 else "expert_activated"
                destination_suffix = "expert_gate_up" if index < 2 else "output"
                weight_offset = index * matrix_bytes
                destination_offset = projection_bytes if index == 1 else 0
                destination_bytes = projection_bytes if index < 2 else 2 * m * h
                expected_prefix = f"moe_full_model.layer{layer}.flexible_moe.value.rank{action.rank}."
                if (params != expected or inbound.value_id != expected_prefix + source_suffix
                        or outbound.value_id != expected_prefix + destination_suffix
                        or not weight.value_id.startswith(expected_prefix + "state.")
                        or weight.size_bytes != 3 * matrix_bytes
                        or weight_addend != weight_offset
                        or weight_view.offset != (weight_offset // 2,)
                        or weight_view.shape != (matrix_bytes // 2,)
                        or output_addend != destination_offset
                        or output_view.offset != (destination_offset // 2,)
                        or output_view.shape != (destination_bytes // 2,)):
                    raise RuntimeError("MoE expert gate/up/down operation or 192B weight view disagrees with model")
                expert_flops += 2 * params[0] * params[1] * params[2] * params[3]
            ref, swiglu = swiglus[0]
            concat, concat_view, _ = physical_operand(ref, SemanticOperandId.COMPUTE_INPUT_ADDRESS)
            activated, activated_view, _ = physical_operand(ref, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
            swiglu_size = tuple(next(item.literal_value for item in swiglu.operands if item.name == "parameters"))
            if (concat.value_id != f"moe_full_model.layer{layer}.flexible_moe.value.rank{action.rank}.expert_gate_up"
                    or activated.value_id != f"moe_full_model.layer{layer}.flexible_moe.value.rank{action.rank}.expert_activated"
                    or concat_view.shape != (2 * m * intermediate,)
                    or activated_view.shape != (m * intermediate,)
                    or swiglu_size != (m * intermediate,)
                    or expert_flops != action.flops):
                raise RuntimeError("MoE expert SwiGLU staging or complete physical FLOPs differ from P2")
        for role, dense_suffix, moe_suffix in (
            ("input", ".norm2_out", "activation"),
            ("output", ".down_out", "output"),
        ):
            bridge = stable_artifact_id(
                "moe_full_model_bridge_action",
                {"source": source, "layer": layer, "role": role},
                schema_version=_LINKER_SCHEMA,
            )
            a, record = endpoint(bridge, LogicalCoreRef(0, 0), RecordOpcode.DTE_ISSUE, SemanticOperandId.SOURCE_ADDRESS)
            b, _ = endpoint(bridge, LogicalCoreRef(0, 0), RecordOpcode.DTE_ISSUE, SemanticOperandId.DESTINATION_ADDRESS)
            dense = a if role == "input" else b
            moe = b if role == "input" else a
            if not dense.value_id.endswith(f".layer{layer}{dense_suffix}") or moe.value_id != f"moe_full_model.layer{layer}.flexible_moe.value.rank0.{moe_suffix}":
                raise RuntimeError(f"Dense↔MoE bridge missing exact layer{layer} {role} boundary")
            if next(item.literal_value for item in record.operands if item.name == "size_bytes") != min(a.size_bytes, b.size_bytes):
                raise RuntimeError(f"Dense↔MoE bridge layer{layer} {role} length is not exact")


def _dense_base_io(profile, sha256: str) -> ProgramIoContract:
    abis = {
        abi.hbm_binding_ref: abi
        for fragment in profile.manifest.fragments
        for abi in fragment.state_abi
    }
    first_access: dict[str, StateUseAccess] = {}
    for action in profile.lowering_context.global_dag.actions:
        for use in action.state_uses:
            first_access.setdefault(use.hbm_binding_ref, use.access)
    state_seeds = {
        abis[binding].state_ref: bytes(abis[binding].size_bytes)
        for binding, access in first_access.items()
        if access is StateUseAccess.READ
    }
    return build_timing_program_io(
        profile, sha256, state_seed_overrides=state_seeds,
    )


def build_full_model_program_io(segment, artifact_sha256: str) -> ProgramIoContract:
    """Seed all borrowed inputs and readable state with a timing-only sidecar."""
    manifest = segment.executable_manifest
    base = _dense_base_io(segment.shared_spine_profile, artifact_sha256)
    definitions = {
        item.symbol.id: (index, item)
        for index, item in enumerate(manifest.program_symbol_definitions)
    }
    state_abis = {
        item.id: item
        for fragment in manifest.fragments for item in fragment.state_abi
    }
    fragments = {fragment.id: fragment for fragment in manifest.fragments}
    readable_state_ids = set()
    for binding in manifest.state_operand_bindings:
        stream = next(
            stream for stream in fragments[binding.fragment_id].core_streams
            if stream.logical_core == binding.logical_core
        )
        if stream.records[binding.fragment_record_index].opcode is RecordOpcode.LSU_LOAD:
            readable_state_ids.add(binding.state_abi_id)
    buffer_abis = {
        item.id: item
        for fragment in manifest.fragments for item in fragment.buffer_abi
    }
    runtime_cores = {
        item.logical_core: item.runtime_core_id
        for item in manifest.core_bindings
    }
    labels = {
        item.symbol.source_ref: (item.symbol.id, index, item)
        for index, item in enumerate(manifest.program_symbol_definitions)
        if item.symbol.kind is ProgramSymbolKind.SRAM_LABEL
    }
    hbms = {
        item.symbol.source_ref: (item.symbol.id, index, item)
        for index, item in enumerate(manifest.program_symbol_definitions)
        if item.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
    }
    blobs = {item.id: item for item in base.blobs}
    initializations = []
    probes = []
    for entry in base.initializations:
        target = entry.target
        if target.buffer_abi_id not in buffer_abis if isinstance(target, ProgramSramTarget) else target.state_abi_id not in state_abis:
            continue
        definition_index, definition = definitions[target.program_symbol_ref]
        initializations.append(type(entry).create(
            target=type(target)(
                **{
                    **{field: getattr(target, field) for field in target.__dataclass_fields__},
                    "finalized_symbol_index": definition_index,
                    "expected_symbol_name": definition.name,
                }
            ),
            offset_bytes=entry.offset_bytes,
            length_bytes=entry.length_bytes,
            blob_ref=entry.blob_ref,
            purpose=entry.purpose,
        ))
    for entry in base.output_probes:
        target = entry.target
        if target.buffer_abi_id not in buffer_abis if isinstance(target, ProgramSramTarget) else target.state_abi_id not in state_abis:
            continue
        definition_index, definition = definitions[target.program_symbol_ref]
        probes.append(ProgramOutputProbe.create(
            target=type(target)(
                **{
                    **{field: getattr(target, field) for field in target.__dataclass_fields__},
                    "finalized_symbol_index": definition_index,
                    "expected_symbol_name": definition.name,
                }
            ),
            offset_bytes=entry.offset_bytes,
            length_bytes=entry.length_bytes,
            blob_ref=entry.blob_ref,
            comparison=entry.comparison,
            capture=entry.capture,
        ))
    initialized_states = {
        entry.target.state_abi_id
        for entry in initializations if isinstance(entry.target, ProgramHbmTarget)
    }
    initialized_buffers = {
        entry.target.buffer_abi_id
        for entry in initializations if isinstance(entry.target, ProgramSramTarget)
    }
    blob_by_size = {item.length_bytes: item for item in blobs.values()}
    def zero_blob(size: int) -> ProgramBlob:
        if size not in blob_by_size:
            item = ProgramBlob.create(bytes(size))
            blob_by_size[size] = item
            blobs[item.id] = item
        return blob_by_size[size]
    for abi in state_abis.values():
        if abi.id in initialized_states or abi.id not in readable_state_ids:
            continue
        symbol_id, index, definition = hbms[abi.hbm_binding_ref]
        target = ProgramHbmTarget(
            ProgramIoTargetKind.HBM, symbol_id, index, definition.name,
            abi.id, abi.state_ref, abi.hbm_binding_ref,
        )
        blob = zero_blob(abi.size_bytes)
        initializations.append(ProgramSramInitialization.create(
            target=target, offset_bytes=0, length_bytes=abi.size_bytes,
            blob_ref=blob.id, purpose=ProgramIoPurpose.STATE,
        ))
    # Timing primitives charge real expert MAC and SwiGLU work but do not
    # materialize FP16 output bytes.  Bootstrap all three expert staging
    # buffers only as TIMING_PARTIAL to make downstream SRAM reads valid;
    # this cannot demonstrate functional gate/up/down values.
    for abi in buffer_abis.values():
        if not (
            abi.ownership.value == "owned"
            and abi.value_id.startswith("moe_full_model.layer")
            and abi.value_id.endswith((".output", ".expert_gate_up", ".expert_activated"))
        ):
            continue
        symbol_id, index, definition = labels[abi.storage_id]
        target = ProgramSramTarget(
            ProgramIoTargetKind.SRAM, runtime_cores[abi.logical_core],
            symbol_id, index, definition.name,
            abi.id, abi.storage_id, abi.value_id, abi.tensor_slice,
            abi.dtype, abi.layout,
        )
        initializations.append(ProgramSramInitialization.create(
            target=target, offset_bytes=0, length_bytes=abi.size_bytes,
            blob_ref=zero_blob(abi.size_bytes).id,
            purpose=ProgramIoPurpose.TIMING_PARTIAL,
        ))
    for abi in buffer_abis.values():
        if abi.ownership.value != "borrowed" or abi.id in initialized_buffers:
            continue
        label = labels.get(abi.storage_id)
        if label is None:
            raise RuntimeError(f"borrowed activation lacks SRAM label: {abi.id}")
        symbol_id, index, definition = label
        target = ProgramSramTarget(
            ProgramIoTargetKind.SRAM, runtime_cores[abi.logical_core],
            symbol_id, index, definition.name,
            abi.id, abi.storage_id, abi.value_id, abi.tensor_slice,
            abi.dtype, abi.layout,
        )
        blob = zero_blob(abi.size_bytes)
        initializations.append(ProgramSramInitialization.create(
            target=target, offset_bytes=0, length_bytes=abi.size_bytes,
            blob_ref=blob.id, purpose=ProgramIoPurpose.ACTIVATION,
        ))
    contract = ProgramIoContract.create(
        producer_pass="build_moe_full_model_program_io",
        mode=ProgramIoMode.TIMING,
        source_manifest=manifest,
        program_artifact_sha256=artifact_sha256,
        blobs=tuple(blobs.values()),
        initializations=tuple(initializations),
        output_probes=tuple(probes),
    )
    contract.validate_against(manifest)
    return contract


def run(args: argparse.Namespace) -> None:
    rows, columns = (int(item) for item in args.mesh_size.split("x"))
    rank_count = rows * columns
    materialization = _manifest(
        WorkloadFamily.MOE_INFERENCE, rows=rows, columns=columns,
    )
    fabric = physical_fabric_from_data(
        minimal_hardware(columns, rows, sram_bytes=65536)
    )
    spaces = valid_hbm_address_spaces(fabric)
    sequence = compile_moe_full_model_inference_sequence(
        materialization, _legacy_template(), fabric,
        hbm_address_spaces=spaces,
    )
    core_bindings = _executable_core_bindings(sequence, rank_count)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    artifacts = []
    sidecars = []
    artifact_digests = []
    units_by_id = {unit.id: unit for unit in sequence.moe_blocks.units}
    expected_flows = []
    expert_records = {
        core_bindings[rank]: [0, 0] for rank in range(1, rank_count)
    }
    for index, segment in enumerate(sequence.segments):
        units = tuple(units_by_id[ref] for ref in segment.moe_unit_refs)
        prove_full_model_dataflow(segment, units)
        expected_flows.extend(
            flow for unit in units for flow in unit.plan.flows
            if (flow.stage in (MoeRectFlowStage.DISPATCH, MoeRectFlowStage.COMBINE)
                and flow.source_rank != flow.destination_rank)
        )
        for unit in units:
            for action in unit.plan.actions:
                if (action.kind is MoeRectActionKind.EXPERT_FORWARD
                        and action.rank != 0 and action.assignment_refs):
                    counts = expert_records[core_bindings[action.rank]]
                    counts[0] += 3
                    counts[1] += 1
        path = output / f"segment_{index}.linked.json"
        artifact = output / f"segment_{index}.npup"
        report = output / f"segment_{index}.finalizer.json"
        path.write_text(
            canonical_json(segment.executable_manifest), encoding="utf-8"
        )
        _run((
            str(args.finalizer.resolve()), "--input", str(path),
            "--output", str(artifact), "--report", str(report),
        ), cwd=output, timeout=120)
        summary = json.loads(report.read_text(encoding="utf-8"))
        sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if (summary.get("artifact_sha256") != sha
                or summary.get("linked_manifest_id")
                != segment.executable_manifest.id
                or summary.get("linked_manifest_digest")
                != segment.executable_manifest_digest):
            raise RuntimeError(f"segment {index} finalizer closure failed")
        contract = build_full_model_program_io(segment, sha)
        if (len(contract.output_probes) != 1
                or contract.output_probes[0].target.value_id != "P0.logits"):
            raise RuntimeError(
                "each MoE full-model segment must physically probe LM logits"
            )
        sidecar = output / f"segment_{index}.program_io.json"
        sidecar.write_text(canonical_json(contract), encoding="utf-8")
        manifests.append(path)
        artifacts.append(artifact)
        sidecars.append(sidecar)
        artifact_digests.append(sha)

    hardware = json.loads(
        specialize_p5_large_release_hardware(rows, columns)
    )
    _bind_native_hardware_to_fabric(hardware, fabric)
    hardware["memory"]["sram_size"] = 131072
    hardware["memory"]["sram"]["capacity_bytes"] = 131072
    access = ["compute", "dte", "lsu", "legacy", "noc_rx"]
    hardware["memory"]["sram"]["regions"] = [
        {
            "name": name,
            "base_bytes": base,
            "size_bytes": size,
            "allocator": "block",
            "spillable": name == "input",
            "access": access,
        }
        for name, base, size in (
            ("sram", 0, 4096),
            ("input", 4096, 36864),
            ("comm", 40960, 36864),
        )
    ]
    spaces_by_die = {space.die_id: space for space in spaces}
    for stack in hardware["memory_system"]["hbm_stacks"]:
        stack["capacity_bytes"] = spaces_by_die[
            stack["compute_die_id"]
        ].size_bytes
    hardware["memory_system"]["address_policy"]["home_ranges"] = [
        {
            "die_id": item.die_id,
            "base": item.base_address,
            "size_bytes": item.size_bytes,
        }
        for item in spaces
    ]
    hardware["memory_system"]["address_policy"][
        "stack_interleave_bytes"
    ] = spaces[0].size_bytes
    hardware_path = output / "hardware.json"
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    mapping_path = output / "mapping.spec"
    mapping_path.write_text("0:0\n", encoding="utf-8")
    stdout = _run((
        str(args.npusim.resolve()),
        "--program-sequence", ",".join(map(str, artifacts)),
        "--linked-manifest-sequence", ",".join(map(str, manifests)),
        "--program-io-sequence", ",".join(map(str, sidecars)),
        "--hardware-config", str(hardware_path),
        "--simulation-config", str(args.simulation.resolve()),
        "--mapping-config", str(mapping_path),
        "--trace-window", "1000000",
    ), cwd=output, timeout=args.timeout)
    (output / "npusim.stdout.txt").write_text(stdout, encoding="utf-8")
    kv = _audit_native_runtime(
        stdout,
        sequence=sequence,
        expected_flows=tuple(expected_flows),
        expert_records_by_core={
            core: tuple(counts) for core, counts in expert_records.items()
        },
        core_bindings=core_bindings,
        rows=rows,
        columns=columns,
    )
    print(
        f"MoE full-model runtime canary PASS mesh={args.mesh_size} "
        f"ep={rank_count} layers=2 sequence={sequence.digest} "
        f"artifacts={','.join(artifact_digests)} "
        f"kv={','.join(item[2] for item in kv)}"
    )

def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-debug-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=build / "moe-full-model-runtime-canary")
    parser.add_argument(
        "--mesh-size",
        choices=tuple(
            f"{rows}x{columns}"
            for rows in range(1, 11)
            for columns in range(1, 11)
        ),
        default="1x2",
        help="physical MoE EP mesh within the 1..10 release envelope",
    )
    parser.add_argument("--finalizer", type=Path, default=build / "npusim_program_finalizer")
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument("--simulation", type=Path, default=_ROOT / "llm/test/program/p5_behavioral_simulation.json")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    for name in ("finalizer", "npusim", "simulation"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name} must name an existing file")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


if __name__ == "__main__":
    run(_parse_args())
