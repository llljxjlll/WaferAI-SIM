"""Source-bound one-core public 0x29-0x2C ProgramIO -> NpuSim canary."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION, AddressOperandBinding, AddressRelocation,
    BufferABI, CommandFragment, CoreFragmentStream, LinkedCoreStream,
    LinkedProgramManifest, LinkedRecordRef, ManifestInputKind, ProgramSymbol,
    ProgramSymbolDefinition, ProgramSymbolKind, RecordOpcode, RecordOperand,
    RelocatableRecord, SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.common import DType, stable_artifact_id
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership, TensorSlice
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramBlob, ProgramIoContract, ProgramIoMode, ProgramIoPurpose,
    ProgramIoTargetKind, ProgramOutputCapture, ProgramOutputComparison,
    ProgramOutputProbe, ProgramSramInitialization, ProgramSramTarget,
    _entry_order,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest, canonical_json, from_data,
)
from llm.test.frontend.integration.flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)

_ROOT = Path(__file__).resolve().parents[4]
_SIM = _ROOT / "llm/test/program/p5_behavioral_simulation.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_PRODUCER = "public_dense_backward_scoped_fragment"
_SOURCE_FILES = (
    _ROOT / "llm/frontend/wafer_frontend/schema/artifact_manifest.py",
    _ROOT / "llm/src/frontend/program_finalizer.cpp",
    _ROOT / "llm/src/isa/record_codec.cpp",
    _ROOT / "llm/src/isa/record_lowering.cpp",
    _ROOT / "llm/src/prims/comp_prims/backward_timing_prims.cpp",
    _ROOT / "llm/src/prims/comp_prims/dense_rope_residual_backward_physical_work.cpp",
    Path(__file__).resolve(),
)


@dataclass(frozen=True, slots=True)
class ScopedDenseBackwardSource:
    rms: tuple[int, ...] = (2, 4, 1, 0)
    attention: tuple[int, ...] = (2, 2, 1, 2, 1, 1, 3)
    rope: tuple[int, ...] = (
        2, 2, 2, 1, 2, 1, 1, 2, 2, 8, 300186932,
    )
    residual: tuple[int, ...] = (2, 2, 1, 4)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack_half(values) -> bytes:
    return b"".join(struct.pack("<e", float(value)) for value in values)


def _buffer(template: BufferABI, *, name: str, offset: int, size: int,
            dtype: DType, shape: tuple[int, ...],
            ownership: BufferOwnership) -> BufferABI:
    value_id = f"dense_backward.{name}"
    semantic = {
        "schedule_id": template.schedule_id,
        "binding_id": f"abs_{name}",
        "value_id": value_id,
        "logical_core": template.logical_core,
        "tensor_slice": TensorSlice(value_id, (0,) * len(shape), shape),
        "region_ref": template.region_ref,
        "region_offset_bytes": offset,
        "size_bytes": size,
        "alignment_bytes": 64,
        "banks": template.banks,
        "storage_id": f"dense_backward_{name}",
        "alias_of": None,
        "lifetime_start": 0,
        "lifetime_end_exclusive": 4,
        "dtype": dtype,
        "layout": "row_major",
        "ownership": ownership,
    }
    return BufferABI(
        stable_artifact_id(
            "buffer_abi", semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        **semantic,
    )


def _alloc(action: str, name: str, offset: int, size: int) -> RelocatableRecord:
    return RelocatableRecord(action, RecordOpcode.SRAM_ALLOC_AT, (
        RecordOperand.address(
            "region_name", SemanticOperandId.REGION_NAME, "p_region"),
        RecordOperand.address(
            "label_symbol", SemanticOperandId.LABEL_SYMBOL,
            f"p_label_{name}"),
        RecordOperand.literal("region_offset_bytes", offset),
        RecordOperand.literal("size_bytes", size),
        RecordOperand.literal("alignment_bytes", 64),
        RecordOperand.literal("lifetime", 0),
        RecordOperand.literal("spillable", False),
    ))


def _free(action: str, name: str) -> RelocatableRecord:
    return RelocatableRecord(action, RecordOpcode.SRAM_FREE, (
        RecordOperand.address(
            "symbol", SemanticOperandId.SYMBOL, f"p_label_{name}"),
    ))


def _bind(action: str, inputs: tuple[str, ...],
          output: str) -> RelocatableRecord:
    operands = [RecordOperand.literal("input_count", len(inputs))]
    for index in range(16):
        operands.append(
            RecordOperand.address(
                f"input_label_{index}",
                SemanticOperandId(SemanticOperandId.SRAM_BIND_INPUT_0 + index),
                f"p_label_{inputs[index]}",
            )
            if index < len(inputs)
            else RecordOperand.literal(f"input_label_{index}", 0)
        )
    operands.append(RecordOperand.address(
        "output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
        f"p_label_{output}",
    ))
    return RelocatableRecord(action, RecordOpcode.SRAM_BIND, tuple(operands))


def _compute(action: str, opcode: RecordOpcode, names: tuple[str, ...],
             values: tuple[int, ...], addresses: tuple[tuple[str,
             SemanticOperandId, str], ...]) -> RelocatableRecord:
    operands = [RecordOperand.literal(name, value)
                for name, value in zip(names[:len(values)], values)]
    for name, operand_id, symbol in addresses:
        operands.append(RecordOperand.address(name, operand_id, symbol))
    operands.extend(
        RecordOperand.literal(name, value)
        for name, value in zip(names[len(values) + len(addresses):],
                               values[len(values):])
    )
    return RelocatableRecord(action, opcode, tuple(operands))


def _record(action: str, opcode: RecordOpcode,
            dtype_names: tuple[str, ...], address_names: tuple[str, ...],
            address_ids: tuple[SemanticOperandId, ...],
            buffer_names: tuple[str, ...], parameter_names: tuple[str, ...],
            parameters: tuple[int, ...], dtypes: tuple[int, ...]) -> RelocatableRecord:
    operands = [
        *(RecordOperand.literal(name, value)
          for name, value in zip(dtype_names, dtypes)),
        *(RecordOperand.address(name, operand_id, f"p_abs_{buffer}")
          for name, operand_id, buffer in
          zip(address_names, address_ids, buffer_names)),
        *(RecordOperand.literal(name, value)
          for name, value in zip(parameter_names, parameters)),
    ]
    return RelocatableRecord(action, opcode, tuple(operands))


def _relocation(index: int, operand: SemanticOperandId, symbol: str,
                kind: ProgramSymbolKind) -> AddressRelocation:
    return AddressRelocation(index, operand, kind, symbol, 0)


def _binding(fragment: str, core, index: int, operand: SemanticOperandId,
             abi: BufferABI) -> AddressOperandBinding:
    return AddressOperandBinding(
        fragment, core, index, operand, (abi.id,), (abi.tensor_slice,),
    )


def build_manifest(emitter: Path,
                   source: ScopedDenseBackwardSource) -> LinkedProgramManifest:
    emitted = subprocess.run(
        [str(emitter), "--emit-gemm-wgrad-manifest"],
        check=True, capture_output=True, text=True,
    )
    base = from_data(LinkedProgramManifest, json.loads(emitted.stdout))
    base.validate("dense_backward.fixture_base")
    core_binding = base.core_bindings[0]
    core = core_binding.logical_core
    template = next(abi for abi in base.fragments[0].buffer_abi
                    if abi.logical_core == core)
    borrowed = BufferOwnership.BORROWED
    owned = BufferOwnership.OWNED
    specs = {
        "rms_forward": (0x100, 16, DType.FP16, (2, 4), borrowed),
        "rms_upstream": (0x140, 16, DType.FP16, (2, 4), borrowed),
        "rms_output": (0x180, 16, DType.FP16, (2, 4), owned),
        "attention_forward": (0x200, 32, DType.FP16, (2, 4, 2), borrowed),
        "attention_upstream": (0x240, 16, DType.FP16, (2, 2, 2), borrowed),
        "attention_output": (0x280, 32, DType.FP16, (2, 4, 2), owned),
        "rope_positions": (0x300, 8, DType.INT32, (2,), borrowed),
        "rope_upstream": (0x340, 32, DType.FP16, (2, 4, 2), borrowed),
        "rope_output": (0x380, 32, DType.FP16, (2, 4, 2), owned),
        "residual_forward": (0x400, 16, DType.FP16, (2, 4), borrowed),
        "residual_upstream": (0x440, 16, DType.FP16, (2, 4), borrowed),
        "residual_left": (0x480, 16, DType.FP16, (2, 4), owned),
        "residual_right": (0x4C0, 16, DType.FP16, (2, 4), owned),
    }
    buffers = {
        name: _buffer(
            template, name=name, offset=offset, size=size, dtype=dtype,
            shape=shape, ownership=ownership,
        )
        for name, (offset, size, dtype, shape, ownership) in specs.items()
    }
    s = SemanticOperandId
    actions = (
        ("dense_backward.rmsnorm", RecordOpcode.RMSNORM_BACKWARD_TIMING,
         ("rms_forward", "rms_upstream"), ("rms_output",),
         ("input_datatype", "upstream_datatype", "output_datatype"),
         ("input_address", "upstream_address", "output_address"),
         (s.COMPUTE_INPUT_ADDRESS, s.COMPUTE_DATA_ADDRESS,
          s.COMPUTE_OUTPUT_ADDRESS),
         ("rows", "hidden_size", "tp_degree", "mode"), source.rms, (1, 1, 1)),
        ("dense_backward.attention", RecordOpcode.ATTENTION_BACKWARD_TIMING,
         ("attention_forward", "attention_upstream"), ("attention_output",),
         ("input_datatype", "upstream_datatype", "output_datatype"),
         ("input_address", "upstream_address", "output_address"),
         (s.COMPUTE_INPUT_ADDRESS, s.COMPUTE_DATA_ADDRESS,
          s.COMPUTE_OUTPUT_ADDRESS),
         ("tokens", "rank_heads", "rank_kv_heads", "head_dim", "tp_degree",
          "sequences", "pairs"), source.attention, (1, 1, 1)),
        ("dense_backward.rope", RecordOpcode.ROPE_BACKWARD_TIMING,
         ("rope_positions", "rope_upstream"), ("rope_output",),
         ("position_datatype", "upstream_datatype", "output_datatype"),
         ("position_address", "upstream_address", "output_address"),
         (s.COMPUTE_INPUT_ADDRESS, s.COMPUTE_DATA_ADDRESS,
          s.COMPUTE_OUTPUT_ADDRESS),
         ("logical_tokens", "rank_tokens", "logical_query_heads",
          "logical_kv_heads", "rank_query_heads", "rank_kv_heads",
          "tp_degree", "head_dim", "rotary_dim",
          "max_position_embeddings", "position_trace_tag"),
         source.rope, (2, 1, 1)),
        ("dense_backward.residual", RecordOpcode.RESIDUAL_BACKWARD_TIMING,
         ("residual_forward", "residual_upstream"),
         ("residual_left", "residual_right"),
         ("forward_datatype", "upstream_datatype", "left_output_datatype",
          "right_output_datatype"),
         ("forward_address", "upstream_address", "left_output_address",
          "right_output_address"),
         (s.COMPUTE_INPUT_ADDRESS, s.COMPUTE_DATA_ADDRESS,
          s.COMPUTE_OUTPUT_ADDRESS, s.COMPUTE_AUX_ADDRESS),
         ("logical_rows", "rank_rows", "tp_degree", "hidden_size"),
         source.residual, (1, 1, 1, 1)),
    )
    records = []
    record_bindings: list[tuple[int, SemanticOperandId, str,
                                ProgramSymbolKind, str]] = []
    for (action, opcode, inputs, outputs, dtype_names, address_names,
         address_ids, parameter_names, parameters, dtypes) in actions:
        names = inputs + outputs
        for name in names:
            index = len(records)
            records.append(_alloc(action, name, specs[name][0], specs[name][1]))
            record_bindings.extend((
                (index, s.REGION_NAME, "p_region",
                 ProgramSymbolKind.SRAM_REGION, name),
                (index, s.LABEL_SYMBOL, f"p_label_{name}",
                 ProgramSymbolKind.SRAM_LABEL, name),
            ))
        bind_index = len(records)
        records.append(_bind(action, inputs, outputs[0]))
        for slot, name in enumerate(inputs):
            record_bindings.append((
                bind_index, SemanticOperandId(
                    SemanticOperandId.SRAM_BIND_INPUT_0 + slot),
                f"p_label_{name}", ProgramSymbolKind.SRAM_LABEL, name))
        record_bindings.append((
            bind_index, s.SRAM_BIND_OUTPUT, f"p_label_{outputs[0]}",
            ProgramSymbolKind.SRAM_LABEL, outputs[0]))
        compute_index = len(records)
        records.append(_record(
            action, opcode, dtype_names, address_names, address_ids,
            inputs + outputs, parameter_names, parameters, dtypes))
        for operand_id, name in zip(address_ids, inputs + outputs):
            record_bindings.append((
                compute_index, operand_id, f"p_abs_{name}",
                ProgramSymbolKind.ABSOLUTE_ADDRESS, name))
        for name in inputs:
            index = len(records)
            records.append(_free(action, name))
            record_bindings.append((
                index, s.SYMBOL, f"p_label_{name}",
                ProgramSymbolKind.SRAM_LABEL, name))

    region = next(definition for definition in base.program_symbol_definitions
                  if definition.symbol.id == "p_region")
    symbols = [region.symbol]
    definitions = [replace(region, logical_cores=(core,))]
    for name, abi in buffers.items():
        absolute = ProgramSymbol(
            f"p_abs_{name}", ProgramSymbolKind.ABSOLUTE_ADDRESS, f"abs_{name}")
        label = ProgramSymbol(
            f"p_label_{name}", ProgramSymbolKind.SRAM_LABEL, abi.storage_id)
        symbols.extend((absolute, label))
        definitions.extend((
            ProgramSymbolDefinition(
                absolute, f"dense_backward.abs.{name}",
                abi.region_offset_bytes, abi.size_bytes, (core,)),
            ProgramSymbolDefinition(
                label, f"dense_backward.label.{name}", 0, 0, (core,)),
        ))
    relocations = tuple(sorted((
        _relocation(index, operand, symbol, kind)
        for index, operand, symbol, kind, _ in record_bindings
    ), key=lambda item: (item.record_index, int(item.operand_id))))
    stream = CoreFragmentStream(core, tuple(records), (), relocations)
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER,
        source_global_dag_id=base.source_global_dag_id,
        kind=base.fragments[0].kind,
        claimed_action_ids=tuple(sorted(action[0] for action in actions)),
        core_streams=(stream,), runtime_symbols=(),
        program_symbols=tuple(sorted(symbols, key=lambda item: item.id)),
        buffer_abi=tuple(sorted(buffers.values(), key=lambda item: item.id)),
        state_abi=(),
    )
    fragment.validate("dense_backward.fragment")
    witnesses = tuple(sorted((
        _binding(fragment.id, core, index, operand, buffers[name])
        for index, operand, _, _, name in record_bindings
    ), key=lambda item: (
        item.logical_core.die_id, item.logical_core.local_core_id,
        item.fragment_id, item.fragment_record_index, int(item.operand_id))))
    linked_stream = LinkedCoreStream(
        core, core_binding.runtime_core_id,
        tuple(LinkedRecordRef(
            fragment.id, index, record.source_global_action_id)
            for index, record in enumerate(records)),
    )
    envelope = replace(
        base.envelope, active_cores=(core,),
        start_events=tuple(event for event in base.envelope.start_events
                           if event.target_core == core),
        terminal_cores=(core,), expected_ack_cores=(core,),
        expected_done_cores=(core,),
    )
    runtime_definitions = tuple(
        definition for definition in base.runtime_symbol_definitions
        if core in definition.logical_cores
    )
    runtime_definitions = (replace(
        runtime_definitions[0],
        symbol=replace(runtime_definitions[0].symbol,
                       source_ref=actions[0][0]),
    ),)
    interface = replace(
        base.fragment_interfaces[0], fragment_id=fragment.id,
        program_exports=tuple(sorted(symbol.id for symbol in symbols)),
    )
    digests = tuple(sorted((
        replace(item, artifact_id=fragment.id,
                schema_version=fragment.schema_version,
                digest=canonical_digest(fragment))
        if item.kind is ManifestInputKind.COMMAND_FRAGMENT else item
        for item in base.input_digests
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    manifest = LinkedProgramManifest.create(
        producer_pass=_PRODUCER, capabilities=base.capabilities,
        source_ir1_id=base.source_ir1_id,
        source_projection_id=base.source_projection_id,
        source_schedule_set_id=base.source_schedule_set_id,
        source_global_dag_id=base.source_global_dag_id,
        input_digests=digests, fragments=(fragment,),
        fragment_interfaces=(interface,), core_bindings=(core_binding,),
        core_streams=(linked_stream,),
        runtime_symbol_definitions=runtime_definitions,
        program_symbol_definitions=tuple(
            sorted(definitions, key=lambda item: item.symbol.id)),
        address_operand_bindings=witnesses,
        state_operand_bindings=(), core_groups=(), envelope=envelope,
    )
    manifest.validate("dense_backward.manifest")
    return manifest


def _payloads() -> dict[str, bytes]:
    return {
        "rms_forward": _pack_half(range(1, 9)),
        "rms_upstream": _pack_half(range(9, 17)),
        "rms_output": bytes(16),
        "attention_forward": _pack_half(range(1, 17)),
        "attention_upstream": _pack_half(range(17, 25)),
        "attention_output": bytes(32),
        "rope_positions": struct.pack("<II", 0, 1),
        "rope_upstream": _pack_half(range(25, 41)),
        "rope_output": bytes(32),
        "residual_forward": _pack_half(range(41, 49)),
        "residual_upstream": _pack_half(range(49, 57)),
        "residual_left": bytes(16),
        "residual_right": bytes(16),
    }


def build_program_io(
    manifest: LinkedProgramManifest, artifact_sha256: str,
    *, payload_override: dict[str, bytes] | None = None,
) -> ProgramIoContract:
    manifest.validate("dense_backward.program_io_source")
    payloads = _payloads()
    if payload_override:
        for name, payload in payload_override.items():
            if name not in payloads or payload != payloads[name]:
                raise SchemaError(
                    "runtime Dense backward payload disagrees with scoped source",
                    path=f"dense_backward.program_io.{name}",
                )
    definitions = {
        definition.symbol.id: (index, definition)
        for index, definition in enumerate(manifest.program_symbol_definitions)
    }
    core_ids = {binding.logical_core: binding.runtime_core_id
                for binding in manifest.core_bindings}
    fragment = manifest.fragments[0]
    stream = fragment.core_streams[0]
    roots = {abi.binding_id.removeprefix("abs_"): abi
             for abi in fragment.buffer_abi
             if abi.logical_core == stream.logical_core}
    outputs = {"rms_output", "attention_output", "rope_output",
               "residual_left", "residual_right"}
    blobs = {}
    initializations = []
    probes = []
    for name, payload in payloads.items():
        abi = roots.get(name)
        if (abi is None or len(payload) != abi.size_bytes or
                (name in outputs) !=
                (abi.ownership is BufferOwnership.OWNED)):
            raise SchemaError(
                "Dense backward payload/BufferABI inventory differs",
                path=f"dense_backward.program_io.{name}",
            )
        matches = [
            (symbol_id, index, definition)
            for symbol_id, (index, definition) in definitions.items()
            if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
            and definition.symbol.source_ref == abi.storage_id
        ]
        if len(matches) != 1:
            raise SchemaError(
                "Dense backward buffer has no unique SRAM label",
                path=f"dense_backward.program_io.{name}",
            )
        symbol_id, symbol_index, definition = matches[0]
        target = ProgramSramTarget(
            ProgramIoTargetKind.SRAM, core_ids[stream.logical_core],
            symbol_id, symbol_index, definition.name, abi.id, abi.storage_id,
            abi.value_id, abi.tensor_slice, abi.dtype, abi.layout,
        )
        blob = ProgramBlob.create(payload)
        blobs[blob.id] = blob
        initializations.append(ProgramSramInitialization.create(
            target=target, offset_bytes=0, length_bytes=abi.size_bytes,
            blob_ref=blob.id,
            purpose=(ProgramIoPurpose.TIMING_PARTIAL
                     if name in outputs else ProgramIoPurpose.ACTIVATION),
        ))
        if name in outputs:
            probes.append(ProgramOutputProbe.create(
                target=target, offset_bytes=0, length_bytes=abi.size_bytes,
                blob_ref=blob.id,
                comparison=ProgramOutputComparison.EXACT_BYTES,
                capture=ProgramOutputCapture.AFTER_PROGRAM,
            ))
    if set(roots) != set(payloads) or len(initializations) != 13 or len(probes) != 5:
        raise SchemaError(
            "Dense backward ProgramIO requires 13 initializations and five probes",
            path="dense_backward.program_io.inventory",
        )
    contract = ProgramIoContract.create(
        producer_pass="public_dense_backward_scoped_program_io",
        mode=ProgramIoMode.TIMING, source_manifest=manifest,
        program_artifact_sha256=artifact_sha256,
        blobs=tuple(sorted(blobs.values(), key=lambda item: item.id)),
        initializations=tuple(sorted(initializations, key=_entry_order)),
        output_probes=tuple(sorted(probes, key=_entry_order)),
    )
    contract.validate_against(manifest)
    return contract


def run(*, emitter: Path, finalizer: Path, resolver: Path, npusim: Path,
        output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    source_sha = {str(path.relative_to(_ROOT)): _sha(path)
                  for path in _SOURCE_FILES}
    tools = {"emitter": emitter, "finalizer": finalizer,
             "resolver": resolver, "npusim": npusim}
    tool_sha = {name: _sha(path) for name, path in tools.items()}
    source = ScopedDenseBackwardSource()
    manifest = build_manifest(emitter, source)
    linked = output / "dense_backward.linked.json"
    linked.write_text(canonical_json(manifest), encoding="utf-8")
    artifact = output / "dense_backward.npup"
    report = output / "dense_backward.finalizer.json"
    finalized = subprocess.run([
        str(finalizer), "--input", str(linked), "--output", str(artifact),
        "--report", str(report),
    ], cwd=_ROOT / "llm", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       text=True, check=False, timeout=120)
    (output / "dense_backward.finalizer.stdout.txt").write_text(
        finalized.stdout, encoding="utf-8")
    if finalized.returncode:
        raise RuntimeError(f"0x29-0x2C finalizer failed: {finalized.stdout}")
    artifact_sha = _sha(artifact)
    contract = build_program_io(manifest, artifact_sha)
    try:
        build_program_io(
            manifest, artifact_sha,
            payload_override={"rms_forward": bytes(16)})
    except SchemaError as error:
        source_negative = "disagrees with scoped source" in str(error)
    else:
        source_negative = False
    if not source_negative:
        raise RuntimeError("zeroed source operand bypassed source binding")
    sidecar = output / "dense_backward.program_io.json"
    sidecar.write_text(canonical_json(contract), encoding="utf-8")
    resolved = subprocess.run([
        str(resolver), "--resolve", str(linked), str(artifact), str(sidecar),
    ], cwd=_ROOT / "llm", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       text=True, check=False, timeout=120)
    (output / "dense_backward.resolver.stdout.txt").write_text(
        resolved.stdout, encoding="utf-8")
    if resolved.returncode:
        raise RuntimeError(
            f"0x29-0x2C ProgramIO resolver failed: {resolved.stdout}")
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 65536
    hardware["memory"]["sram"]["capacity_bytes"] = 65536
    hardware["memory"]["sram"]["regions"] = [{
        "name": "sram_main", "base_bytes": 0, "size_bytes": 4096,
        "allocator": "block", "spillable": False,
        "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
    }]
    hw = output / "hardware.json"
    hw.write_text(json.dumps(
        hardware, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    runtime = subprocess.run([
        str(npusim), "--program", str(artifact),
        "--linked-manifest", str(linked), "--program-io", str(sidecar),
        "--hardware-config", str(hw), "--simulation-config", str(_SIM),
        "--mapping-config", str(_MAPPING), "--trace-window", "1000000",
    ], cwd=_ROOT / "llm", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       text=True, check=False, timeout=600)
    transcript = output / "dense_backward.npusim.stdout.txt"
    transcript.write_text(runtime.stdout, encoding="utf-8")
    stages = re.findall(
        r"\[DENSE_BACKWARD_PUBLIC\] stage=(rmsnorm|attention|rope|residual)"
        r".* pass=(\d+)", runtime.stdout)
    phases = re.findall(
        r"\[PROGRAM_IO\] phase=(resolved|applied|verify) mode=timing "
        r"initializations=(\d+) probes=(\d+) "
        r"checksum=([0-9a-f]+) pass=(\d+)", runtime.stdout)
    probes = re.findall(
        r"\[PROGRAM_IO_PROBE\].* address=(\d+) bytes=(\d+)"
        r".* valid=(\d+) exact=(\d+) pass=(\d+)", runtime.stdout)
    makespan = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)",
                          runtime.stdout)
    expected_probes = {
        ("384", "16", "1", "1", "1"),
        ("640", "32", "1", "1", "1"),
        ("896", "32", "1", "1", "1"),
        ("1152", "16", "1", "1", "1"),
        ("1216", "16", "1", "1", "1"),
    }
    passed = (
        runtime.returncode == 0
        and stages == [("rmsnorm", "1"), ("attention", "1"),
                       ("rope", "1"), ("residual", "1")]
        and [phase[0] for phase in phases] ==
            ["resolved", "applied", "verify"]
        and all((phase[1], phase[2], phase[4]) == ("13", "5", "1")
                for phase in phases)
        and set(probes) == expected_probes
        and len(makespan) == 1 and int(makespan[0]) > 0
        and "[DRAIN] router_residual=0" in runtime.stdout
        and "[DRAIN] d2d_link_residual=0" in runtime.stdout
    )
    if source_sha != {str(path.relative_to(_ROOT)): _sha(path)
                      for path in _SOURCE_FILES}:
        raise RuntimeError("Dense backward source drifted during scoped run")
    if tool_sha != {name: _sha(path) for name, path in tools.items()}:
        raise RuntimeError("Dense backward native tool drifted during run")
    evidence = {
        "scope": "one core public 0x29-0x2C timing primitive chain",
        "manifest_id": manifest.id, "manifest_sha256": _sha(linked),
        "artifact_sha256": artifact_sha,
        "program_io_id": contract.id, "program_io_sha256": _sha(sidecar),
        "source_sha256": source_sha, "tool_sha256": tool_sha,
        "finalizer_exit": finalized.returncode,
        "resolver_exit": resolved.returncode,
        "npusim_exit": runtime.returncode,
        "npusim_stdout_sha256": _sha(transcript),
        "program_io_phases": phases, "primitive_stages": stages,
        "output_probes": probes,
        "makespan_cycles": int(makespan[0]) if makespan else None,
        "negative_zeroed_source_rejected": source_negative,
        "full_dense_training_complete": False,
        "scoped_runtime_status": "PASS" if passed else "FAIL",
    }
    (output / "dense_backward.evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2), encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emitter", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = run(
        emitter=args.emitter.resolve(), finalizer=args.finalizer.resolve(),
        resolver=args.resolver.resolve(), npusim=args.npusim.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps({
        "scope": evidence["scope"],
        "scoped_runtime_status": evidence["scoped_runtime_status"],
        "full_dense_training_complete": evidence["full_dense_training_complete"],
        "artifact_sha256": evidence["artifact_sha256"],
        "program_io_sha256": evidence["program_io_sha256"],
    }, sort_keys=True))
    return 0 if evidence["scoped_runtime_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
