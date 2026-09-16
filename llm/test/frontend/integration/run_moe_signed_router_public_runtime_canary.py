"""Source-bound public 0x27/0x28 ProgramIO -> NpuSim scoped canary.

This executes one signed EP2 router tile. It does not claim a full-model
shared dCombined producer or M1 MoE TRAIN completion.
"""
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
from llm.frontend.wafer_frontend.passes.moe_full_train_router_native_protocol import (
    MoeRouterNativeSourcePair, build_moe_router_native_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_return_protocol import (
    build_moe_router_signed_return_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_score_source import (
    build_moe_trainable_signed_router_requirements,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION, AddressOperandBinding, AddressRelocation,
    BufferABI, CommandFragment, CoreFragmentStream, LinkedCoreStream,
    LinkedProgramManifest, LinkedRecordRef, ManifestInputKind, ProgramSymbol,
    ProgramSymbolDefinition, ProgramSymbolKind, RecordOpcode, RecordOperand,
    RelocatableRecord, SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.common import DType, stable_artifact_id
from llm.frontend.wafer_frontend.schema.flexible_moe import (
    MoeRectSignedTop1TrainSource,
)
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
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)

_ROOT = Path(__file__).resolve().parents[4]
_SIM = _ROOT / "llm/test/program/p5_behavioral_simulation.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_PRODUCER = "public_moe_signed_router_scoped_fragment"
_SOURCE_FILES = (
    _ROOT / "llm/frontend/wafer_frontend/schema/artifact_manifest.py",
    _ROOT / "llm/frontend/wafer_frontend/schema/flexible_moe.py",
    _ROOT / "llm/frontend/wafer_frontend/schema/moe_router_signed_weight_workload.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/moe_signed_router_train_source_plan.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/moe_full_train_router_native_protocol.py",
    Path(__file__).resolve(),
)


@dataclass(frozen=True, slots=True)
class ScopedSource:
    signed: MoeRectSignedTop1TrainSource
    pair: MoeRouterNativeSourcePair
    native_protocol_id: str

    @property
    def shared_dcombined_is_bound(self) -> bool:
        return not (
            self.signed.shared_dcombined_producer_ref.startswith("REQUIRED.")
            or self.signed.shared_full_model_manifest_ref.startswith("REQUIRED.")
        )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack_half(values) -> bytes:
    return b"".join(struct.pack("<e", value) for value in values)


def build_scoped_source() -> ScopedSource:
    Fixture.setUpClass()
    sequence = Fixture.sequence
    score = build_moe_trainable_signed_router_requirements(sequence)
    returned = build_moe_router_signed_return_protocol(score, sequence)
    native = build_moe_router_native_protocol(score, returned, sequence)
    pair = native.pairs[0]
    unit = next(item for item in sequence.units
                if (item.step, item.layer) == (pair.step, pair.layer))
    signed = MoeRectSignedTop1TrainSource.create(
        unit.spec,
        dynamic_case_ref=score.dynamic_score_case_ref,
        shared_dcombined_producer_ref=(
            "REQUIRED.real_shared_spine.backward.dcombined."
            f"step{pair.step}.layer{pair.layer}"
        ),
        shared_full_model_manifest_ref=(
            "REQUIRED.real_shared_dense_global_training_timeline"
        ),
    )
    pair.weighted_forward.validate("scoped_source.forward")
    pair.score_backward.validate("scoped_source.backward")
    route_words = tuple(
        field for route in pair.weighted_forward.routes
        for field in (
            route.token_index, route.source_rank, route.selected_expert,
            route.expert_home_rank, route.expert_slot_index,
        )
    )
    if (
        signed.route_words != route_words
        or signed.route_bytes != 80
        or (pair.weighted_forward.rank_rows,
            pair.weighted_forward.hidden_size,
            pair.weighted_forward.expert_count) != (4, 4, 2)
        or pair.score_backward.source_weighted_forward_ref
        != pair.weighted_forward.id
    ):
        raise SchemaError(
            "scoped 0x27/0x28 geometry is not the exact signed source pair",
            path="moe_signed_router_canary.source",
        )
    return ScopedSource(signed, pair, native.id)


def _buffer(template: BufferABI, *, name: str, offset: int, size: int,
            dtype: DType, shape: tuple[int, ...],
            ownership: BufferOwnership) -> BufferABI:
    value_id = f"moe_router.{name}"
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
        "storage_id": f"moe_router_{name}",
        "alias_of": None,
        "lifetime_start": 0,
        "lifetime_end_exclusive": 2,
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


def _alloc(action: str, name: str, offset: int,
           size: int) -> RelocatableRecord:
    return RelocatableRecord(
        action, RecordOpcode.SRAM_ALLOC_AT, (
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
        ),
    )


def _free(action: str, name: str) -> RelocatableRecord:
    return RelocatableRecord(
        action, RecordOpcode.SRAM_FREE, (
            RecordOperand.address(
                "symbol", SemanticOperandId.SYMBOL, f"p_label_{name}"),
        ),
    )


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


def _forward(source: ScopedSource, action: str) -> RelocatableRecord:
    work = source.pair.weighted_forward
    s = SemanticOperandId
    return RelocatableRecord(
        action, RecordOpcode.MOE_SCORE_WEIGHTED_FORWARD, (
            RecordOperand.literal("route_datatype", 2),
            RecordOperand.literal("score_datatype", 1),
            RecordOperand.literal("expert_datatype", 1),
            RecordOperand.literal("combined_datatype", 1),
            RecordOperand.address("route_address",
                                  s.COMPUTE_ROUTE_TABLE_ADDRESS,
                                  "p_abs_route"),
            RecordOperand.address("score_address", s.COMPUTE_INPUT_ADDRESS,
                                  "p_abs_score"),
            RecordOperand.address("return_address", s.COMPUTE_DATA_ADDRESS,
                                  "p_abs_returns"),
            RecordOperand.address("combined_address",
                                  s.COMPUTE_OUTPUT_ADDRESS,
                                  "p_abs_combined"),
            RecordOperand.literal("rank_rows", work.rank_rows),
            RecordOperand.literal("hidden_size", work.hidden_size),
            RecordOperand.literal("expert_count", work.expert_count),
            RecordOperand.literal("route_bytes", source.signed.route_bytes),
        ),
    )


def _backward(source: ScopedSource, action: str) -> RelocatableRecord:
    work = source.pair.score_backward
    s = SemanticOperandId
    return RelocatableRecord(
        action, RecordOpcode.MOE_SCORE_WEIGHT_BACKWARD, (
            RecordOperand.literal("route_datatype", 2),
            RecordOperand.literal("score_datatype", 1),
            RecordOperand.literal("expert_datatype", 1),
            RecordOperand.literal("upstream_datatype", 1),
            RecordOperand.literal("dscore_datatype", 1),
            RecordOperand.literal("dexpert_datatype", 1),
            RecordOperand.address("route_address",
                                  s.COMPUTE_ROUTE_TABLE_ADDRESS,
                                  "p_abs_route"),
            RecordOperand.address("score_address", s.COMPUTE_INPUT_ADDRESS,
                                  "p_abs_score"),
            RecordOperand.address("return_address", s.COMPUTE_DATA_ADDRESS,
                                  "p_abs_returns"),
            RecordOperand.address("dcombined_address",
                                  s.COMPUTE_OUTPUT_ADDRESS,
                                  "p_abs_dcombined"),
            RecordOperand.address("dscore_address", s.COMPUTE_AUX_ADDRESS,
                                  "p_abs_dscore"),
            RecordOperand.address("dexpert_address",
                                  s.COMPUTE_ROUTER_DEXPERT_ADDRESS,
                                  "p_abs_dexpert"),
            RecordOperand.literal("rank_rows", work.rank_rows),
            RecordOperand.literal("hidden_size", work.hidden_size),
            RecordOperand.literal("expert_count", work.expert_count),
            RecordOperand.literal("route_bytes", source.signed.route_bytes),
        ),
    )


def _symbols_and_definitions(base: LinkedProgramManifest, buffers):
    core = base.core_bindings[0].logical_core
    region = next(definition for definition in base.program_symbol_definitions
                  if definition.symbol.id == "p_region")
    symbols = [region.symbol]
    definitions = [replace(region, logical_cores=(core,))]
    for name, abi in buffers.items():
        absolute = ProgramSymbol(
            f"p_abs_{name}", ProgramSymbolKind.ABSOLUTE_ADDRESS,
            f"abs_{name}")
        label = ProgramSymbol(
            f"p_label_{name}", ProgramSymbolKind.SRAM_LABEL,
            abi.storage_id)
        symbols.extend((absolute, label))
        definitions.extend((
            ProgramSymbolDefinition(
                absolute, f"moe_router.abs.{name}", abi.region_offset_bytes,
                abi.size_bytes, (core,)),
            ProgramSymbolDefinition(
                label, f"moe_router.label.{name}", 0, 0, (core,)),
        ))
    return (
        tuple(sorted(symbols, key=lambda item: item.id)),
        tuple(sorted(definitions, key=lambda item: item.symbol.id)),
    )


def _relocation(index: int, operand: SemanticOperandId, symbol: str,
                kind: ProgramSymbolKind) -> AddressRelocation:
    return AddressRelocation(index, operand, kind, symbol, 0)


def _binding(fragment: str, core, index: int, operand: SemanticOperandId,
             abi: BufferABI) -> AddressOperandBinding:
    return AddressOperandBinding(
        fragment, core, index, operand, (abi.id,), (abi.tensor_slice,),
    )


def build_manifest(emitter: Path, source: ScopedSource) -> LinkedProgramManifest:
    emitted = subprocess.run(
        [str(emitter), "--emit-gemm-wgrad-manifest"],
        check=True, capture_output=True, text=True,
    )
    base = from_data(LinkedProgramManifest, json.loads(emitted.stdout))
    base.validate("moe_router.fixture_base")
    core_binding = base.core_bindings[0]
    core = core_binding.logical_core
    template = next(abi for abi in base.fragments[0].buffer_abi
                    if abi.logical_core == core)
    specs = {
        "route": (0x100, 80, DType.INT32, (4, 5),
                  BufferOwnership.BORROWED),
        "score": (0x180, 16, DType.FP16, (4, 2),
                  BufferOwnership.BORROWED),
        "returns": (0x1C0, 32, DType.FP16, (4, 4),
                    BufferOwnership.BORROWED),
        "combined": (0x200, 32, DType.FP16, (4, 4),
                     BufferOwnership.OWNED),
        "dcombined": (0x240, 32, DType.FP16, (4, 4),
                      BufferOwnership.BORROWED),
        "dscore": (0x280, 16, DType.FP16, (4, 2),
                   BufferOwnership.OWNED),
        "dexpert": (0x2C0, 32, DType.FP16, (4, 4),
                    BufferOwnership.OWNED),
    }
    buffers = {
        name: _buffer(
            template, name=name, offset=offset, size=size, dtype=dtype,
            shape=shape, ownership=ownership,
        )
        for name, (offset, size, dtype, shape, ownership) in specs.items()
    }
    forward_action = source.pair.weighted_forward.id
    backward_action = source.pair.score_backward.id
    records = (
        *(_alloc(forward_action, name, specs[name][0], specs[name][1])
          for name in ("route", "score", "returns", "combined")),
        _bind(forward_action, ("route", "score", "returns"), "combined"),
        _forward(source, forward_action),
        *(_alloc(backward_action, name, specs[name][0], specs[name][1])
          for name in ("dcombined", "dscore", "dexpert")),
        _bind(backward_action,
              ("route", "score", "returns", "dcombined"), "dscore"),
        _backward(source, backward_action),
        *(_free(backward_action, name)
          for name in ("route", "score", "returns", "dcombined")),
    )
    relocations = []
    witnesses = []

    def add(index, operand, symbol, kind, name):
        relocations.append(_relocation(index, operand, symbol, kind))
        witnesses.append(
            _binding("pending", core, index, operand, buffers[name]))

    for index, name in enumerate(("route", "score", "returns", "combined")):
        add(index, SemanticOperandId.REGION_NAME, "p_region",
            ProgramSymbolKind.SRAM_REGION, name)
        add(index, SemanticOperandId.LABEL_SYMBOL, f"p_label_{name}",
            ProgramSymbolKind.SRAM_LABEL, name)

    bind_forward = 4
    for slot, name in enumerate(("route", "score", "returns")):
        operand = SemanticOperandId(
            SemanticOperandId.SRAM_BIND_INPUT_0 + slot)
        add(bind_forward, operand, f"p_label_{name}",
            ProgramSymbolKind.SRAM_LABEL, name)
    add(bind_forward, SemanticOperandId.SRAM_BIND_OUTPUT,
        "p_label_combined", ProgramSymbolKind.SRAM_LABEL, "combined")

    forward_index = 5
    for operand, name in (
        (SemanticOperandId.COMPUTE_ROUTE_TABLE_ADDRESS, "route"),
        (SemanticOperandId.COMPUTE_INPUT_ADDRESS, "score"),
        (SemanticOperandId.COMPUTE_DATA_ADDRESS, "returns"),
        (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, "combined"),
    ):
        add(forward_index, operand, f"p_abs_{name}",
            ProgramSymbolKind.ABSOLUTE_ADDRESS, name)

    for index, name in enumerate(("dcombined", "dscore", "dexpert"), start=6):
        add(index, SemanticOperandId.REGION_NAME, "p_region",
            ProgramSymbolKind.SRAM_REGION, name)
        add(index, SemanticOperandId.LABEL_SYMBOL, f"p_label_{name}",
            ProgramSymbolKind.SRAM_LABEL, name)

    bind_backward = 9
    for slot, name in enumerate(("route", "score", "returns", "dcombined")):
        operand = SemanticOperandId(
            SemanticOperandId.SRAM_BIND_INPUT_0 + slot)
        add(bind_backward, operand, f"p_label_{name}",
            ProgramSymbolKind.SRAM_LABEL, name)
    add(bind_backward, SemanticOperandId.SRAM_BIND_OUTPUT,
        "p_label_dscore", ProgramSymbolKind.SRAM_LABEL, "dscore")

    backward_index = 10
    for operand, name in (
        (SemanticOperandId.COMPUTE_ROUTE_TABLE_ADDRESS, "route"),
        (SemanticOperandId.COMPUTE_INPUT_ADDRESS, "score"),
        (SemanticOperandId.COMPUTE_DATA_ADDRESS, "returns"),
        (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, "dcombined"),
        (SemanticOperandId.COMPUTE_AUX_ADDRESS, "dscore"),
        (SemanticOperandId.COMPUTE_ROUTER_DEXPERT_ADDRESS, "dexpert"),
    ):
        add(backward_index, operand, f"p_abs_{name}",
            ProgramSymbolKind.ABSOLUTE_ADDRESS, name)

    for index, name in enumerate(
            ("route", "score", "returns", "dcombined"), start=11):
        add(index, SemanticOperandId.SYMBOL, f"p_label_{name}",
            ProgramSymbolKind.SRAM_LABEL, name)

    symbols, definitions = _symbols_and_definitions(base, buffers)
    stream = CoreFragmentStream(
        core, records, (),
        tuple(sorted(relocations, key=lambda item:
                     (item.record_index, int(item.operand_id)))),
    )
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER,
        source_global_dag_id=base.source_global_dag_id,
        kind=base.fragments[0].kind,
        claimed_action_ids=tuple(sorted((forward_action, backward_action))),
        core_streams=(stream,), runtime_symbols=(),
        program_symbols=symbols,
        buffer_abi=tuple(sorted(buffers.values(), key=lambda item: item.id)),
        state_abi=(),
    )
    fragment.validate("moe_router.fragment")
    witnesses = tuple(sorted((
        replace(item, fragment_id=fragment.id) for item in witnesses
    ), key=lambda item: (
        item.logical_core.die_id, item.logical_core.local_core_id,
        item.fragment_id, item.fragment_record_index, int(item.operand_id),
    )))
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
    if (
        len(runtime_definitions) != 1 or len(envelope.start_events) != 1
        or runtime_definitions[0].symbol.id
           != envelope.start_events[0].tag_symbol_ref
    ):
        raise SchemaError(
            "scoped manifest requires one exact inherited START_TAG",
            path="moe_router.manifest.runtime_symbols",
        )
    runtime_definitions = (replace(
        runtime_definitions[0],
        symbol=replace(runtime_definitions[0].symbol,
                       source_ref=forward_action),
    ),)
    interface = replace(
        base.fragment_interfaces[0], fragment_id=fragment.id,
        program_exports=tuple(symbol.id for symbol in symbols),
    )
    digests = tuple(sorted((
        replace(item, artifact_id=fragment.id,
                schema_version=fragment.schema_version,
                digest=canonical_digest(fragment))
        if item.kind is ManifestInputKind.COMMAND_FRAGMENT else item
        for item in base.input_digests
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    manifest = LinkedProgramManifest.create(
        producer_pass=_PRODUCER,
        capabilities=base.capabilities,
        source_ir1_id=base.source_ir1_id,
        source_projection_id=base.source_projection_id,
        source_schedule_set_id=base.source_schedule_set_id,
        source_global_dag_id=base.source_global_dag_id,
        input_digests=digests, fragments=(fragment,),
        fragment_interfaces=(interface,), core_bindings=(core_binding,),
        core_streams=(linked_stream,),
        runtime_symbol_definitions=runtime_definitions,
        program_symbol_definitions=definitions,
        address_operand_bindings=witnesses,
        state_operand_bindings=(), core_groups=(), envelope=envelope,
    )
    manifest.validate("moe_router.manifest")
    return manifest


def _expected_payloads(source: ScopedSource) -> dict[str, bytes]:
    scores = (2.0, -8.0, 7.0, 3.0, 5.0, -9.0, 11.0, 4.0)
    returned = tuple(float(value) for value in range(1, 17))
    upstream = (1.0,) * 16
    work = source.pair.weighted_forward
    group_rows = {
        (group.expert_home_rank, group.expert_index):
        group.offset_bytes // (2 * work.hidden_size)
        for group in work.expert_groups
    }
    combined = [0.0] * 16
    dscore = [0.0] * 8
    dexpert = [0.0] * 16
    for row, item in enumerate(work.routes):
        expert_row = group_rows[
            (item.expert_home_rank, item.selected_expert)]
        expert_row += item.expert_slot_index
        score = scores[row * work.expert_count + item.selected_expert]
        dot = 0.0
        for column in range(work.hidden_size):
            value = returned[expert_row * work.hidden_size + column]
            combined[row * work.hidden_size + column] = score * value
            dot += upstream[row * work.hidden_size + column] * value
            dexpert[expert_row * work.hidden_size + column] = (
                score * upstream[row * work.hidden_size + column])
        dscore[row * work.expert_count + item.selected_expert] = dot
    return {
        "route": source.signed.route_blob,
        "score": _pack_half(scores),
        "returns": _pack_half(returned),
        "combined": _pack_half(combined),
        "dcombined": _pack_half(upstream),
        "dscore": _pack_half(dscore),
        "dexpert": _pack_half(dexpert),
    }


def build_program_io(
    manifest: LinkedProgramManifest, artifact_sha256: str,
    source: ScopedSource, *, route_override: bytes | None = None,
) -> ProgramIoContract:
    manifest.validate("moe_router.program_io_source")
    if manifest.producer_pass != _PRODUCER:
        raise SchemaError("router ProgramIO requires exact scoped producer",
                          path="moe_router.program_io.producer")
    payloads = _expected_payloads(source)
    if route_override is not None:
        if type(route_override) is not bytes or route_override != payloads["route"]:
            raise SchemaError(
                "runtime route blob disagrees with signed router source",
                path="moe_router.program_io.route",
            )
        payloads["route"] = route_override
    definitions = {
        definition.symbol.id: (index, definition)
        for index, definition in enumerate(
            manifest.program_symbol_definitions)
    }
    core_ids = {
        binding.logical_core: binding.runtime_core_id
        for binding in manifest.core_bindings
    }
    fragment = manifest.fragments[0]
    stream = fragment.core_streams[0]
    roots = {
        abi.binding_id.removeprefix("abs_"): abi
        for abi in fragment.buffer_abi
        if abi.logical_core == stream.logical_core
    }
    outputs = {"combined", "dscore", "dexpert"}
    blobs = {}
    initializations = []
    probes = []
    for name, payload in payloads.items():
        abi = roots.get(name)
        if (
            abi is None or len(payload) != abi.size_bytes
            or (name == "route" and abi.dtype is not DType.INT32)
            or (name != "route" and abi.dtype is not DType.FP16)
            or (name in outputs) !=
               (abi.ownership is BufferOwnership.OWNED)
            or (name not in outputs) !=
               (abi.ownership is BufferOwnership.BORROWED)
        ):
            raise SchemaError(
                "router ProgramIO payload/BufferABI inventory differs",
                path=f"moe_router.program_io.{name}",
            )
        matches = [
            (symbol_id, index, definition)
            for symbol_id, (index, definition) in definitions.items()
            if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
            and definition.symbol.source_ref == abi.storage_id
        ]
        if len(matches) != 1:
            raise SchemaError(
                "router buffer has no unique SRAM_ALLOC_AT label",
                path=f"moe_router.program_io.{name}",
            )
        symbol_id, symbol_index, definition = matches[0]
        target = ProgramSramTarget(
            ProgramIoTargetKind.SRAM, core_ids[stream.logical_core],
            symbol_id, symbol_index, definition.name, abi.id, abi.storage_id,
            abi.value_id, abi.tensor_slice, abi.dtype, abi.layout,
        )
        expected_blob = ProgramBlob.create(payload)
        initialization_blob = ProgramBlob.create(
            bytes(abi.size_bytes) if name in outputs else payload
        )
        blobs[expected_blob.id] = expected_blob
        blobs[initialization_blob.id] = initialization_blob
        initializations.append(ProgramSramInitialization.create(
            target=target, offset_bytes=0, length_bytes=abi.size_bytes,
            blob_ref=initialization_blob.id,
            purpose=(ProgramIoPurpose.TIMING_PARTIAL
                     if name in outputs else ProgramIoPurpose.ACTIVATION),
        ))
        if name in outputs:
            probes.append(ProgramOutputProbe.create(
                target=target, offset_bytes=0, length_bytes=abi.size_bytes,
                blob_ref=expected_blob.id,
                comparison=ProgramOutputComparison.EXACT_BYTES,
                capture=ProgramOutputCapture.AFTER_PROGRAM,
            ))
    if (
        set(roots) != set(payloads)
        or len(initializations) != 7 or len(probes) != 3
        or hashlib.sha256(payloads["route"]).hexdigest()
           != source.signed.route_sha256
    ):
        raise SchemaError(
            "router ProgramIO must initialize seven exact spans and probe all three outputs",
            path="moe_router.program_io.inventory",
        )
    contract = ProgramIoContract.create(
        producer_pass="public_moe_signed_router_scoped_program_io",
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
    source_sha = {
        str(path.relative_to(_ROOT)): _sha(path) for path in _SOURCE_FILES
    }
    tools = {
        "emitter": emitter, "finalizer": finalizer,
        "resolver": resolver, "npusim": npusim,
    }
    tool_sha = {name: _sha(path) for name, path in tools.items()}
    source = build_scoped_source()
    manifest = build_manifest(emitter, source)
    linked = output / "moe_signed_router.linked.json"
    linked.write_text(canonical_json(manifest), encoding="utf-8")
    artifact = output / "moe_signed_router.npup"
    report = output / "moe_signed_router.finalizer.json"
    finalized = subprocess.run([
        str(finalizer), "--input", str(linked), "--output", str(artifact),
        "--report", str(report),
    ], cwd=_ROOT / "llm", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       text=True, check=False, timeout=120)
    (output / "moe_signed_router.finalizer.stdout.txt").write_text(
        finalized.stdout, encoding="utf-8")
    if finalized.returncode:
        raise RuntimeError(
            f"0x27/0x28 finalizer failed: {finalized.stdout}")
    artifact_sha = _sha(artifact)
    contract = build_program_io(manifest, artifact_sha, source)
    forged_route = bytearray(source.signed.route_blob)
    forged_route[8] ^= 1
    try:
        build_program_io(
            manifest, artifact_sha, source,
            route_override=bytes(forged_route),
        )
    except SchemaError as error:
        route_negative = "disagrees with signed router source" in str(error)
    else:
        route_negative = False
    if not route_negative:
        raise RuntimeError("forged equal-length route bypassed source binding")
    sidecar = output / "moe_signed_router.program_io.json"
    sidecar.write_text(canonical_json(contract), encoding="utf-8")
    resolved = subprocess.run([
        str(resolver), "--resolve", str(linked), str(artifact), str(sidecar),
    ], cwd=_ROOT / "llm", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       text=True, check=False, timeout=120)
    (output / "moe_signed_router.resolver.stdout.txt").write_text(
        resolved.stdout, encoding="utf-8")
    if resolved.returncode:
        raise RuntimeError(
            f"0x27/0x28 ProgramIO resolver failed: {resolved.stdout}")
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
    transcript = output / "moe_signed_router.npusim.stdout.txt"
    transcript.write_text(runtime.stdout, encoding="utf-8")
    stages = re.findall(
        r"\[MOE_SIGNED_ROUTER\] stage=(forward|backward) rows=(\d+) "
        r"hidden=(\d+) experts=(\d+) route_read_bytes=(\d+) "
        r"fp16_read_bytes=(\d+) (?:combined_write_bytes=(\d+)|"
        r"dscore_write_bytes=(\d+) dexpert_write_bytes=(\d+)) pass=(\d+)",
        runtime.stdout,
    )
    phases = re.findall(
        r"\[PROGRAM_IO\] phase=(resolved|applied|verify) mode=timing "
        r"initializations=(\d+) probes=(\d+) "
        r"checksum=([0-9a-f]+) pass=(\d+)",
        runtime.stdout,
    )
    probes = re.findall(
        r"\[PROGRAM_IO_PROBE\].* address=(\d+) bytes=(\d+)"
        r".* valid=(\d+) exact=(\d+) pass=(\d+)",
        runtime.stdout,
    )
    makespan = re.findall(
        r"\[SIM_RESULT\] makespan_cycles=(\d+)", runtime.stdout)
    expected_stages = [
        ("forward", "4", "4", "2", "80", "48", "32", "", "", "1"),
        ("backward", "4", "4", "2", "80", "80", "", "16", "32", "1"),
    ]
    passed = (
        runtime.returncode == 0
        and stages == expected_stages
        and [phase[0] for phase in phases]
            == ["resolved", "applied", "verify"]
        and all((phase[1], phase[2], phase[4]) == ("7", "3", "1")
                for phase in phases)
        and {(item[0], item[1], item[2], item[3], item[4])
             for item in probes}
            == {("512", "32", "1", "1", "1"),
                ("640", "16", "1", "1", "1"),
                ("704", "32", "1", "1", "1")}
        and len(makespan) == 1 and int(makespan[0]) > 0
        and "[DRAIN] router_residual=0" in runtime.stdout
        and "[DRAIN] d2d_link_residual=0" in runtime.stdout
    )
    if source_sha != {
        str(path.relative_to(_ROOT)): _sha(path) for path in _SOURCE_FILES
    }:
        raise RuntimeError("router Python source drifted during scoped run")
    if tool_sha != {name: _sha(path) for name, path in tools.items()}:
        raise RuntimeError("router native tool drifted during scoped run")
    evidence = {
        "scope": "one signed EP2 router tile; public 0x27/0x28 only",
        "source_id": source.signed.id,
        "native_protocol_id": source.native_protocol_id,
        "weighted_forward_source_id": source.pair.weighted_forward.id,
        "score_backward_source_id": source.pair.score_backward.id,
        "route_sha256": source.signed.route_sha256,
        "manifest_id": manifest.id,
        "manifest_sha256": _sha(linked),
        "artifact_sha256": artifact_sha,
        "program_io_id": contract.id,
        "program_io_sha256": _sha(sidecar),
        "source_sha256": source_sha,
        "tool_sha256": tool_sha,
        "finalizer_exit": finalized.returncode,
        "resolver_exit": resolved.returncode,
        "npusim_exit": runtime.returncode,
        "npusim_stdout_sha256": _sha(transcript),
        "program_io_phases": phases,
        "router_stages": stages,
        "output_probes": probes,
        "makespan_cycles": int(makespan[0]) if makespan else None,
        "negative_forged_route_rejected": route_negative,
        "shared_dcombined_full_model_producer_bound":
            source.shared_dcombined_is_bound,
        "m1_moe_train_complete": False,
        "scoped_runtime_status": "PASS" if passed else "FAIL",
    }
    (output / "moe_signed_router.evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8")
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
        emitter=args.emitter.resolve(),
        finalizer=args.finalizer.resolve(),
        resolver=args.resolver.resolve(),
        npusim=args.npusim.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps({
        "scope": evidence["scope"],
        "scoped_runtime_status": evidence["scoped_runtime_status"],
        "shared_dcombined_full_model_producer_bound":
            evidence["shared_dcombined_full_model_producer_bound"],
        "m1_moe_train_complete": evidence["m1_moe_train_complete"],
        "artifact_sha256": evidence["artifact_sha256"],
        "program_io_sha256": evidence["program_io_sha256"],
    }, sort_keys=True))
    return 0 if evidence["scoped_runtime_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
