"""One source trace → one BORROWED INT32 route SRAM blob, read by both prims.

Rows are five little-endian uint32 fields: token, source rank, selected expert,
expert home rank, expert slot. No nested literal or second handwritten trace
may substitute the signed physical SRAM input to native 0x27/0x28.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import OperandKind, RecordOpcode
from ..schema.common import DType, stable_artifact_id
from ..schema.ir2 import BufferOwnership
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.program_io import (
    ProgramIoPurpose, ProgramIoTargetKind,
    ProgramIoContract as ProgramIOContract, ProgramOutputCapture,
)
from .moe_full_train_named_wgrad_operands import (
    MoeWgradPhysicalSlice, _buffer_slice, _one_record,
)
from .moe_full_train_router_native_protocol import (
    MoeRouterSourceNativeProtocol,
)
from .moe_full_train_router_return_protocol import (
    MoeRouterSignedReturnProtocol,
)
from .moe_full_train_router_score_source import (
    MoeTrainableSignedRouterRequirements,
)

_VERSION = "wafer_frontend.moe_router_route_sram/v1alpha1"
_FIELDS = ("token_index", "source_rank", "selected_expert",
           "expert_home_rank", "expert_slot_index")
# Reserved *proposal* IDs, not yet present in the public SemanticOperandId.
# Public bridge must establish these separate roles and refuse AdamW 13..20.
_DEXPERT_OPERAND_ID = 21
_ROUTE_OPERAND_ID = 22


@dataclass(frozen=True, slots=True)
class MoeRouterRouteSramSource:
    id: str
    source_moe_sequence_ref: str
    source_router_score_ref: str
    source_return_ref: str
    source_native_ref: str
    version: str
    routes: tuple["MoeRouterPhysicalRouteSeed", ...]

    def validate_against(self, score: MoeTrainableSignedRouterRequirements,
                         returned: MoeRouterSignedReturnProtocol,
                         native: MoeRouterSourceNativeProtocol,
                         sequence: MoeCompileSequence) -> None:
        if self != build_moe_router_route_sram(score, returned, native, sequence):
            raise SchemaError("route SRAM seed must originate in one frozen P2 source trace",
                              path="moe_router_route_sram.source")

    def require_physical_route_reads(
        self, score: MoeTrainableSignedRouterRequirements,
        returned: MoeRouterSignedReturnProtocol,
        native: MoeRouterSourceNativeProtocol,
        sequence: MoeCompileSequence,
        program_io_by_step_layer: dict[tuple[int, int], ProgramIOContract],
    ) -> None:
        """Strict late gate; current static production leaf cannot pass it."""
        self.validate_against(score, returned, native, sequence)
        op_forward = RecordOpcode._value2member_map_.get(0x27)
        op_backward = RecordOpcode._value2member_map_.get(0x28)
        if op_forward is None or op_backward is None:
            raise SchemaError("0x27/0x28 public native route address roles are not implemented",
                              path="moe_router_route_sram.physical")
        units = {(unit.step, unit.layer): unit for unit in sequence.units}
        paths = {(path.step, path.layer, path.source_rank): path
                 for path in score.paths if path.routes}
        pairs = {(pair.step, pair.layer, pair.source_rank): pair
                 for pair in native.pairs}
        for seed in self.routes:
            key = (seed.step, seed.layer)
            manifest = units[key].linked_manifest
            path = paths[key + (seed.source_rank,)]
            pair = pairs[key + (seed.source_rank,)]
            roots = [abi for fragment in manifest.fragments
                     for abi in fragment.buffer_abi
                     if abi.logical_core.die_id == seed.source_rank
                     and abi.value_id.endswith(".router_route_table")]
            if (len(roots) != 1 or roots[0].ownership is not BufferOwnership.BORROWED
                    or roots[0].dtype is not DType.INT32
                    or roots[0].size_bytes != seed.route_bytes
                    or roots[0].alias_of is not None):
                raise SchemaError("one independent, full-length BORROWED INT32 route SRAM BufferABI required",
                                  path=f"router_route.step{seed.step}.layer{seed.layer}")
            route_span = _buffer_slice(manifest, seed.source_rank,
                                       ".router_route_table", DType.INT32,
                                       0, seed.route_bytes)
            for action_ref, opcode, expected_address_count in (
                (path.weighted_combine_action_ref, op_forward, 4),
                (path.combine_backward_action_ref, op_backward, 6),
            ):
                record, fragment, stream, index = _one_record(
                    manifest, seed.source_rank, action_ref, (opcode,))
                physical_address_roles = [item for item in record.operands
                                          if item.kind is OperandKind.ADDRESS_SYMBOL]
                if (len(physical_address_roles) != expected_address_count
                        or any(item.name == "route_table"
                               and item.kind is OperandKind.LITERAL
                               for item in record.operands)):
                    raise SchemaError("native router must read route SRAM address, not hidden duplicate trace literal",
                                      path=f"router_route.step{seed.step}.layer{seed.layer}")
                route_address = next((item for item in physical_address_roles
                                      if item.name == "route_address"), None)
                if (route_address is None or route_address.operand_id.value
                        != _ROUTE_OPERAND_ID
                        or _route_operand_slice(manifest, fragment, stream,
                                                index, seed.route_bytes) != route_span):
                    raise SchemaError("both router prims must read the same signed BORROWED INT32 route SRAM bytes",
                                      path=f"router_route.step{seed.step}.layer{seed.layer}")
                if opcode is op_backward:
                    dexpert = next((item for item in physical_address_roles
                                    if item.name == "dexpert_address"), None)
                    dscore = next((item for item in physical_address_roles
                                   if item.name == "dscore_address"), None)
                    if (dexpert is None or dscore is None
                            or dexpert.operand_id.value != _DEXPERT_OPERAND_ID
                            or dexpert.symbol_ref == dscore.symbol_ref):
                        raise SchemaError("0x28 needs a separate sixth-address route input and fifth distinct dExpert output",
                                          path=f"router_route.step{seed.step}.layer{seed.layer}")
                    _require_distinct_gradient_output(manifest, fragment,
                        stream, index, pair.score_backward.operand_bytes[3],
                        pair.score_backward.operand_bytes[4])
            io = program_io_by_step_layer.get(key)
            if type(io) is not ProgramIOContract:
                raise SchemaError("router route must have a real linked ProgramIO contract",
                                  path=f"router_route.step{seed.step}.layer{seed.layer}")
            io.validate_against(manifest)
            for suffix, expected_bytes in ((".router_score_gradient",
                         pair.score_backward.operand_bytes[3]),
                        (".router_dexpert_gradient",
                         pair.score_backward.operand_bytes[4])):
                out = [abi for fragment in manifest.fragments
                       for abi in fragment.buffer_abi
                       if abi.logical_core.die_id == seed.source_rank
                       and abi.value_id.endswith(suffix)]
                if (len(out) != 1 or out[0].ownership is not
                        BufferOwnership.OWNED):
                    raise SchemaError("0x28 must own two distinct FP16 SRAM gradients",
                                      path=f"router_route.step{seed.step}.layer{seed.layer}")
                probes = [probe for probe in io.output_probes
                          if probe.target.kind is ProgramIoTargetKind.SRAM
                          and probe.target.buffer_abi_id == out[0].id
                          and probe.offset_bytes == 0
                          and probe.length_bytes == expected_bytes
                          and probe.capture is ProgramOutputCapture.AFTER_PROGRAM]
                if len(probes) != 1:
                    raise SchemaError("0x28 requires separate full-span AFTER_PROGRAM probes for both owned outputs",
                                      path=f"router_route.step{seed.step}.layer{seed.layer}")
            initializers = [item for item in io.initializations
                            if item.target.kind is ProgramIoTargetKind.SRAM
                            and item.target.buffer_abi_id == roots[0].id]
            blobs = {blob.id: blob for blob in io.blobs}
            if (len(initializers) != 1
                    or initializers[0].purpose is not ProgramIoPurpose.ACTIVATION
                    or initializers[0].offset_bytes != 0
                    or initializers[0].length_bytes != seed.route_bytes
                    or initializers[0].blob_ref not in blobs
                    or blobs[initializers[0].blob_ref].payload()
                    != seed.payload):
                raise SchemaError("ProgramIO route INT32 seed must be the full source-frozen nonzero 5-column blob",
                                  path=f"router_route.step{seed.step}.layer{seed.layer}")


def _route_operand_slice(manifest, fragment, stream, index, bytes_needed):
    record = stream.records[index]
    operand = next((item for item in record.operands
                    if item.name == "route_address"), None)
    if operand is None or operand.kind is not OperandKind.ADDRESS_SYMBOL:
        raise SchemaError("router route needs one real INT32 address operand",
                          path="moe_router_route_sram.route_address")
    reloc = next((item for item in stream.address_relocations
                  if item.record_index == index
                  and item.operand_id is operand.operand_id), None)
    closure = next((item for item in manifest.address_operand_bindings
                    if item.fragment_id == fragment.id
                    and item.logical_core == stream.logical_core
                    and item.fragment_record_index == index
                    and item.operand_id is operand.operand_id), None)
    if (reloc is None or closure is None
            or reloc.symbol_ref != operand.symbol_ref
            or len(closure.buffer_abi_ids) != 1
            or len(closure.tensor_slices) != 1):
        raise SchemaError("route SRAM address needs exact relocation and one typed BufferABI closure",
                          path="moe_router_route_sram.route_address")
    abi = next((abi for abi in fragment.buffer_abi
                if abi.id == closure.buffer_abi_ids[0]), None)
    if abi is None or abi.dtype is not DType.INT32:
        raise SchemaError("route SRAM compute input must be INT32",
                          path="moe_router_route_sram.route_address")
    view = closure.tensor_slices[0]
    if (reloc.addend != 0 or bytes_needed != abi.size_bytes
            or bytes_needed % 4 != 0
            or view.value_id != abi.value_id
            or view.offset != (0,)
            or view.shape != (bytes_needed // 4,)):
        raise SchemaError("route SRAM record must read the full signed INT32 payload",
                          path="moe_router_route_sram.route_address")
    return MoeWgradPhysicalSlice(abi.id, operand.symbol_ref,
                                 abi.value_id, abi.dtype, 0, bytes_needed)


def _require_distinct_gradient_output(manifest, fragment, stream,
                                      index, dscore_bytes, dexpert_bytes):
    from .moe_full_train_named_wgrad_operands import _operand_slice
    score = _operand_slice(manifest, fragment, stream, index,
                           "dscore_address", dscore_bytes,
                           require_exact_view=True)
    expert = _operand_slice(manifest, fragment, stream, index,
                            "dexpert_address", dexpert_bytes,
                            require_exact_view=True)
    if (score.buffer_abi_ref == expert.buffer_abi_ref
            or score.absolute_symbol_ref == expert.absolute_symbol_ref):
        raise SchemaError("router score/expert gradients alias one physical SRAM allocation",
                          path="moe_router_route_sram.outputs")
    definitions = {item.symbol.id: item for item in
                   manifest.program_symbol_definitions}
    ranges = []
    for item in (score, expert):
        definition = definitions.get(item.absolute_symbol_ref)
        if definition is None:
            raise SchemaError("dScore and dExpert need physical SRAM address definitions",
                              path="moe_router_route_sram.outputs")
        start = definition.value + item.offset_bytes
        end = start + item.size_bytes
        if not 0 < start < end <= (1 << 16):
            raise SchemaError("dScore and dExpert need two real nonzero 16-bit SRAM spans",
                              path="moe_router_route_sram.outputs")
        ranges.append((start, end))
    if ranges[0][0] < ranges[1][1] and ranges[1][0] < ranges[0][1]:
        raise SchemaError("dScore and dExpert physical SRAM spans overlap",
                          path="moe_router_route_sram.outputs")


@dataclass(frozen=True, slots=True)
class MoeRouterPhysicalRouteSeed:
    step: int
    layer: int
    source_rank: int
    route_rows: int
    route_bytes: int
    sha256: str
    payload_hex: str

    @property
    def payload(self) -> bytes:
        return bytes.fromhex(self.payload_hex)


def build_moe_router_route_sram(
    score: MoeTrainableSignedRouterRequirements,
    returned: MoeRouterSignedReturnProtocol,
    native: MoeRouterSourceNativeProtocol,
    sequence: MoeCompileSequence,
) -> MoeRouterRouteSramSource:
    score.validate_against(sequence)
    returned.validate_against(score, sequence)
    native.validate_against(score, returned, sequence)
    signed = {(path.step, path.layer, path.source_rank): path
              for path in score.paths if path.routes}
    routes = []
    for pair in native.pairs:
        key = (pair.step, pair.layer, pair.source_rank)
        path = signed[key]
        rows = path.routes
        payload = b"".join(struct.pack("<IIIII", *(getattr(route, name)
                          for name in _FIELDS)) for route in rows)
        if (len(rows) != pair.score_backward.rank_rows
                or len(payload) != len(rows) * 5 * 4
                or not any(byte != 0 for byte in payload)
                or tuple(tuple(getattr(route, name) for name in _FIELDS)
                         for route in pair.score_backward.routes)
                   != tuple(tuple(getattr(route, name) for name in _FIELDS)
                            for route in rows)):
            raise SchemaError("route SRAM must preserve actual selected expert/home/slot for each source token",
                              path=f"router_route.step{pair.step}.layer{pair.layer}")
        routes.append(MoeRouterPhysicalRouteSeed(
            pair.step, pair.layer, pair.source_rank, len(rows),
            len(payload), hashlib.sha256(payload).hexdigest(), payload.hex()))
    semantic = {"source_moe_sequence_ref": sequence.id,
                "source_router_score_ref": score.id,
                "source_return_ref": returned.id,
                "source_native_ref": native.id,
                "version": _VERSION, "routes": tuple(routes)}
    return MoeRouterRouteSramSource(
        stable_artifact_id("moe_router_route_sram", semantic,
                           schema_version=_VERSION), **semantic)


__all__ = ["MoeRouterPhysicalRouteSeed", "MoeRouterRouteSramSource",
           "build_moe_router_route_sram"]
