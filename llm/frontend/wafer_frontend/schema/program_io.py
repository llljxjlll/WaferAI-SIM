"""Versioned, symbol-aware host SRAM initialization and output probes.

The linked manifest proves symbolic allocation, BufferABI, and fixed physical
placement.  It deliberately does not embed the GlobalAction DAG itself, so
this module cannot prove cross-action read-before-write dominance.  A future
producer must additionally validate exact first-read coverage against its
``LoweringContext``.  The schema conservatively requires complete host data
for every BORROWED BufferABI visible in the manifest.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .artifact_manifest import (
    BufferABI,
    CommandFragment,
    LinkedProgramManifest,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RegionManifest,
    SemanticOperandId,
    StateABI,
)
from .common import (
    DType,
    UINT64_MAX,
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
)
from .ir2 import BufferOwnership, TensorSlice
from .persistent_state import PersistentStateAccess
from .serde import canonical_digest


PROGRAM_BLOB_SCHEMA_VERSION = "wafer_frontend.program_blob/v1alpha1"
PROGRAM_SRAM_INITIALIZATION_SCHEMA_VERSION = (
    "wafer_frontend.program_sram_initialization/v1alpha2"
)
PROGRAM_OUTPUT_PROBE_SCHEMA_VERSION = (
    "wafer_frontend.program_output_probe/v1alpha2"
)
PROGRAM_IO_CONTRACT_SCHEMA_VERSION = (
    "wafer_frontend.program_io_contract/v1alpha2"
)


class ProgramIoMode(str, Enum):
    TIMING = "timing"
    FUNCTIONAL = "functional"


class ProgramIoPurpose(str, Enum):
    ACTIVATION = "activation"
    WEIGHT = "weight"
    STATE = "state"
    TIMING_PARTIAL = "timing_partial"


class ProgramIoTargetKind(str, Enum):
    SRAM = "sram"
    HBM = "hbm"


class ProgramOutputComparison(str, Enum):
    EXACT_BYTES = "exact_bytes/v1"


class ProgramOutputCapture(str, Enum):
    AFTER_PROGRAM = "after_program/v1"


def _validate_sha256(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(
            "must be a canonical lowercase SHA-256 hex digest",
            path=path,
        )


def _validate_utf8_name(value: str, path: str) -> None:
    validate_nonempty(value, path)
    if "\x00" in value:
        raise SchemaError("must not contain NUL", path=path)
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise SchemaError(
            "must be valid UTF-8 without surrogate code points",
            path=path,
        ) from error
    if len(encoded) > 255:
        raise SchemaError(
            "must fit the 255-byte ProgramArtifact symbol-name field",
            path=path,
        )


def _checked_end(offset: int, length: int, path: str) -> int:
    validate_uint64(offset, f"{path}.offset_bytes")
    validate_uint64(length, f"{path}.length_bytes")
    if length == 0:
        raise SchemaError("must be positive", path=f"{path}.length_bytes")
    if offset > UINT64_MAX - length:
        raise SchemaError("byte range overflows uint64", path=path)
    return offset + length


@dataclass(frozen=True, slots=True)
class ProgramBlob:
    """Canonical inline bytes; C++ may later replace this with a blob section."""

    id: str
    bytes_base64: str
    length_bytes: int
    sha256: str

    @classmethod
    def create(cls, payload: bytes) -> "ProgramBlob":
        if type(payload) is not bytes or not payload:
            raise SchemaError("payload must be non-empty bytes", path="payload")
        encoded = base64.b64encode(payload).decode("ascii")
        semantic_key = {
            "bytes_base64": encoded,
            "length_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        return cls(
            id=stable_artifact_id(
                "program_blob",
                semantic_key,
                schema_version=PROGRAM_BLOB_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "bytes_base64": self.bytes_base64,
            "length_bytes": self.length_bytes,
            "sha256": self.sha256,
        }

    def payload(self) -> bytes:
        try:
            encoded = self.bytes_base64.encode("ascii")
            payload = base64.b64decode(encoded, validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as error:
            raise SchemaError(
                "must be strict canonical base64",
                path="program_blob.bytes_base64",
            ) from error
        if base64.b64encode(payload).decode("ascii") != self.bytes_base64:
            raise SchemaError(
                "must be strict canonical base64",
                path="program_blob.bytes_base64",
            )
        return payload

    def validate(self, path: str = "program_blob") -> None:
        validate_uint64(self.length_bytes, f"{path}.length_bytes")
        if self.length_bytes == 0:
            raise SchemaError("must be positive", path=f"{path}.length_bytes")
        _validate_sha256(self.sha256, f"{path}.sha256")
        try:
            encoded = self.bytes_base64.encode("ascii")
            payload = base64.b64decode(encoded, validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as error:
            raise SchemaError(
                "must be strict canonical base64",
                path=f"{path}.bytes_base64",
            ) from error
        if base64.b64encode(payload).decode("ascii") != self.bytes_base64:
            raise SchemaError(
                "must be strict canonical base64",
                path=f"{path}.bytes_base64",
            )
        if len(payload) != self.length_bytes:
            raise SchemaError(
                "decoded bytes disagree with length_bytes",
                path=f"{path}.length_bytes",
            )
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise SchemaError(
                "decoded bytes disagree with sha256",
                path=f"{path}.sha256",
            )
        expected_id = stable_artifact_id(
            "program_blob",
            self._semantic_key(),
            schema_version=PROGRAM_BLOB_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class ProgramSramTarget:
    kind: ProgramIoTargetKind
    runtime_core_id: int
    program_symbol_ref: str
    finalized_symbol_index: int
    expected_symbol_name: str
    buffer_abi_id: str
    storage_id: str
    value_id: str
    tensor_slice: TensorSlice
    dtype: DType
    layout: str

    def validate(self, path: str = "program_sram_target") -> None:
        if self.kind is not ProgramIoTargetKind.SRAM:
            raise SchemaError("must be SRAM", path=f"{path}.kind")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError(
                "must fit the ProgramArtifact uint16 core id",
                path=f"{path}.runtime_core_id",
            )
        validate_uint64(
            self.finalized_symbol_index,
            f"{path}.finalized_symbol_index",
        )
        for field_name in (
            "program_symbol_ref",
            "buffer_abi_id",
            "storage_id",
            "value_id",
            "layout",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_utf8_name(
            self.expected_symbol_name,
            f"{path}.expected_symbol_name",
        )
        if type(self.tensor_slice) is not TensorSlice:
            raise SchemaError(
                "must be a TensorSlice", path=f"{path}.tensor_slice"
            )
        self.tensor_slice.validate(f"{path}.tensor_slice")
        if self.tensor_slice.value_id != self.value_id:
            raise SchemaError(
                "tensor_slice disagrees with value_id",
                path=f"{path}.tensor_slice.value_id",
            )
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class ProgramHbmTarget:
    kind: ProgramIoTargetKind
    program_symbol_ref: str
    finalized_symbol_index: int
    expected_symbol_name: str
    state_abi_id: str
    state_ref: str
    hbm_binding_ref: str

    def validate(self, path: str = "program_hbm_target") -> None:
        if self.kind is not ProgramIoTargetKind.HBM:
            raise SchemaError("must be HBM", path=f"{path}.kind")
        validate_uint64(
            self.finalized_symbol_index,
            f"{path}.finalized_symbol_index",
        )
        for field_name in (
            "program_symbol_ref",
            "state_abi_id",
            "state_ref",
            "hbm_binding_ref",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_utf8_name(
            self.expected_symbol_name,
            f"{path}.expected_symbol_name",
        )


ProgramIoTarget = ProgramSramTarget | ProgramHbmTarget


@dataclass(frozen=True, slots=True)
class ProgramSramInitialization:
    id: str
    target: ProgramIoTarget
    offset_bytes: int
    length_bytes: int
    blob_ref: str
    purpose: ProgramIoPurpose

    @classmethod
    def create(cls, **semantic_key: object) -> "ProgramSramInitialization":
        return cls(
            id=stable_artifact_id(
                "program_sram_initialization",
                semantic_key,
                schema_version=PROGRAM_SRAM_INITIALIZATION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "target",
                "offset_bytes",
                "length_bytes",
                "blob_ref",
                "purpose",
            )
        }

    def validate(self, path: str = "program_sram_initialization") -> None:
        if type(self.target) not in (ProgramSramTarget, ProgramHbmTarget):
            raise SchemaError(
                "must be a ProgramSramTarget or ProgramHbmTarget",
                path=f"{path}.target",
            )
        self.target.validate(f"{path}.target")
        validate_nonempty(self.blob_ref, f"{path}.blob_ref")
        if type(self.purpose) is not ProgramIoPurpose:
            raise SchemaError("must be a ProgramIoPurpose", path=f"{path}.purpose")
        if (type(self.target) is ProgramHbmTarget) != (
            self.purpose is ProgramIoPurpose.STATE
        ):
            raise SchemaError(
                "STATE purpose must exactly accompany an HBM target",
                path=f"{path}.purpose",
            )
        _checked_end(self.offset_bytes, self.length_bytes, path)
        expected_id = stable_artifact_id(
            "program_sram_initialization",
            self._semantic_key(),
            schema_version=PROGRAM_SRAM_INITIALIZATION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class ProgramOutputProbe:
    id: str
    target: ProgramIoTarget
    offset_bytes: int
    length_bytes: int
    blob_ref: str
    comparison: ProgramOutputComparison
    capture: ProgramOutputCapture

    @classmethod
    def create(cls, **semantic_key: object) -> "ProgramOutputProbe":
        return cls(
            id=stable_artifact_id(
                "program_output_probe",
                semantic_key,
                schema_version=PROGRAM_OUTPUT_PROBE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "target",
                "offset_bytes",
                "length_bytes",
                "blob_ref",
                "comparison",
                "capture",
            )
        }

    def validate(self, path: str = "program_output_probe") -> None:
        if type(self.target) not in (ProgramSramTarget, ProgramHbmTarget):
            raise SchemaError(
                "must be a ProgramSramTarget or ProgramHbmTarget",
                path=f"{path}.target",
            )
        self.target.validate(f"{path}.target")
        validate_nonempty(self.blob_ref, f"{path}.blob_ref")
        if self.comparison is not ProgramOutputComparison.EXACT_BYTES:
            raise SchemaError(
                "MVP only supports exact byte comparison",
                path=f"{path}.comparison",
            )
        if self.capture is not ProgramOutputCapture.AFTER_PROGRAM:
            raise SchemaError(
                "MVP only supports after-program capture",
                path=f"{path}.capture",
            )
        _checked_end(self.offset_bytes, self.length_bytes, path)
        expected_id = stable_artifact_id(
            "program_output_probe",
            self._semantic_key(),
            schema_version=PROGRAM_OUTPUT_PROBE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class _Allocation:
    runtime_core_id: int
    label_symbol_ref: str
    label_definition_index: int
    label_definition: ProgramSymbolDefinition
    region_symbol_ref: str
    region_definition: ProgramSymbolDefinition
    abi: BufferABI
    region_offset_bytes: int
    size_bytes: int

    @property
    def absolute_start(self) -> int:
        return self.region_definition.value + self.region_offset_bytes


def _leaf(fragment: CommandFragment | RegionManifest) -> CommandFragment:
    return fragment.fragment if type(fragment) is RegionManifest else fragment


def _buffer_abis(manifest: LinkedProgramManifest) -> dict[str, BufferABI]:
    result: dict[str, BufferABI] = {}
    for linked in manifest.fragments:
        for abi in _leaf(linked).buffer_abi:
            previous = result.setdefault(abi.id, abi)
            if previous != abi:
                raise SchemaError(
                    "conflicting shared BufferABI definition",
                    path="linked_program_manifest.fragments",
                )
    return result


def _state_abis_and_directions(
    manifest: LinkedProgramManifest,
) -> tuple[dict[str, StateABI], dict[str, set[RecordOpcode]]]:
    by_id: dict[str, StateABI] = {}
    by_binding: dict[str, StateABI] = {}
    leaves = tuple(_leaf(linked) for linked in manifest.fragments)
    for fragment in leaves:
        for abi in fragment.state_abi:
            previous = by_id.setdefault(abi.id, abi)
            if previous != abi:
                raise SchemaError(
                    "conflicting shared StateABI definition",
                    path="linked_program_manifest.fragments",
                )
            previous = by_binding.setdefault(abi.hbm_binding_ref, abi)
            if previous != abi:
                raise SchemaError(
                    "conflicting StateABI for one HBM binding",
                    path="linked_program_manifest.fragments",
                )

    directions: dict[str, set[RecordOpcode]] = {
        abi_id: set() for abi_id in by_id
    }
    for fragment in leaves:
        symbols = {symbol.id: symbol for symbol in fragment.program_symbols}
        for stream in fragment.core_streams:
            for relocation in stream.address_relocations:
                if relocation.operand_id is not SemanticOperandId.HBM_ADDRESS:
                    continue
                symbol = symbols.get(relocation.symbol_ref)
                abi = (
                    None
                    if symbol is None
                    else by_binding.get(symbol.source_ref)
                )
                if abi is None:
                    raise SchemaError(
                        "HBM relocation does not resolve to one StateABI",
                        path="linked_program_manifest.fragments",
                    )
                record = stream.records[relocation.record_index]
                if record.opcode not in (
                    RecordOpcode.LSU_LOAD,
                    RecordOpcode.LSU_STORE,
                ):
                    raise SchemaError(
                        "HBM state relocation requires blocking LSU",
                        path="linked_program_manifest.fragments",
                    )
                directions[abi.id].add(record.opcode)
    return by_id, directions


def _manifest_allocations(
    manifest: LinkedProgramManifest,
    path: str,
) -> tuple[dict[tuple[int, str], _Allocation], dict[str, BufferABI]]:
    definitions = {
        definition.symbol.id: (index, definition)
        for index, definition in enumerate(manifest.program_symbol_definitions)
    }
    runtime_by_core = {
        binding.logical_core: binding.runtime_core_id
        for binding in manifest.core_bindings
    }
    address_bindings = {
        (
            binding.fragment_id,
            binding.logical_core,
            binding.fragment_record_index,
            binding.operand_id,
        ): binding
        for binding in manifest.address_operand_bindings
    }
    abis = _buffer_abis(manifest)
    by_storage: dict[str, list[BufferABI]] = {}
    for abi in abis.values():
        by_storage.setdefault(abi.storage_id, []).append(abi)
    nonalias_abi_ids: set[str] = set()
    for members in by_storage.values():
        roots = tuple(
            abi
            for abi in members
            if abi.ownership is not BufferOwnership.ALIASED
            and abi.alias_of is None
        )
        if len(roots) != 1:
            raise SchemaError(
                "one SRAM storage requires exactly one canonical non-alias root",
                path=f"{path}.fragments",
            )
        root = roots[0]
        nonalias_abi_ids.add(root.id)
        for alias in members:
            if alias is root:
                continue
            common_alias = (
                alias.ownership is not BufferOwnership.ALIASED
                or alias.alias_of != root.binding_id
                or (
                    alias.schedule_id,
                    alias.logical_core,
                    alias.region_ref,
                    alias.banks,
                    alias.storage_id,
                    alias.dtype,
                )
                != (
                    root.schedule_id,
                    root.logical_core,
                    root.region_ref,
                    root.banks,
                    root.storage_id,
                    root.dtype,
                )
                or root.lifetime_start > alias.lifetime_start
                or root.lifetime_end_exclusive
                < alias.lifetime_end_exclusive
            )
            whole_root_alias = (
                alias.region_offset_bytes,
                alias.size_bytes,
                alias.tensor_slice.offset,
                alias.tensor_slice.shape,
                alias.layout,
            ) == (
                root.region_offset_bytes,
                root.size_bytes,
                root.tensor_slice.offset,
                root.tensor_slice.shape,
                root.layout,
            )
            backward_subview_alias = (
                manifest.producer_pass in (
                    "lite_moe_backward_manifest_linker",
                    "lite_moe_dp4_backward_manifest_linker",
                )
                and alias.region_offset_bytes >= root.region_offset_bytes
                and alias.size_bytes > 0
                and alias.region_offset_bytes + alias.size_bytes
                <= root.region_offset_bytes + root.size_bytes
                and alias.layout == f"{root.layout}_view"
            )
            unfused_comparison_subview_alias = (
                manifest.producer_pass == "unfused_comparison_standard_linker"
                and root.layout == "unfused_comparison_storage/v1"
                and alias.region_offset_bytes >= root.region_offset_bytes
                and alias.size_bytes > 0
                and alias.region_offset_bytes + alias.size_bytes
                <= root.region_offset_bytes + root.size_bytes
                and alias.tensor_slice.value_id == alias.value_id
            )
            fused_terminal_subview_alias = (
                manifest.producer_pass == "swizzle_standard_linker"
                and root.layout == "swizzle_standard_terminal_root/v1"
                and alias.layout == "swizzle_standard_terminal_subview/v1"
                and root.ownership is BufferOwnership.OWNED
                and alias.value_id == root.value_id
                and alias.tensor_slice.value_id == root.tensor_slice.value_id
                and alias.region_offset_bytes >= root.region_offset_bytes
                and alias.size_bytes > 0
                and alias.region_offset_bytes + alias.size_bytes
                <= root.region_offset_bytes + root.size_bytes
                and len(alias.tensor_slice.shape) == len(root.tensor_slice.shape)
                and all(
                    root_offset <= alias_offset
                    and alias_offset + alias_extent
                    <= root_offset + root_extent
                    for root_offset, root_extent, alias_offset, alias_extent
                    in zip(
                        root.tensor_slice.offset,
                        root.tensor_slice.shape,
                        alias.tensor_slice.offset,
                        alias.tensor_slice.shape,
                        strict=True,
                    )
                )
            )
            fused_storage_subview_alias = (
                manifest.producer_pass == "swizzle_standard_linker"
                and root.layout == "swizzle_standard_storage_root/v1"
                and alias.layout == "swizzle_standard_storage_subview/v1"
                and alias.region_offset_bytes >= root.region_offset_bytes
                and alias.size_bytes > 0
                and alias.region_offset_bytes + alias.size_bytes
                <= root.region_offset_bytes + root.size_bytes
            )
            moe_storage_subview_alias = (
                manifest.producer_pass == "moe_swizzle_standard_linker"
                and root.layout.startswith("moe_swizzle_")
                and root.layout.endswith("_root/v1")
                and alias.layout
                == f"{root.layout.removesuffix('_root/v1')}_subview/v1"
                and alias.alignment_bytes <= root.alignment_bytes
                and root.alignment_bytes % alias.alignment_bytes == 0
                and alias.region_offset_bytes >= root.region_offset_bytes
                and alias.size_bytes > 0
                and alias.region_offset_bytes + alias.size_bytes
                <= root.region_offset_bytes + root.size_bytes
            )
            alignment_exact = alias.alignment_bytes == root.alignment_bytes
            if common_alias or not (
                moe_storage_subview_alias
                or alignment_exact and (
                whole_root_alias
                or backward_subview_alias
                or unfused_comparison_subview_alias
                or fused_terminal_subview_alias
                or fused_storage_subview_alias
                )
            ):
                raise SchemaError(
                    "aliased BufferABI must directly reuse its canonical root allocation",
                    path=f"{path}.fragments",
                )
    allocations: dict[tuple[int, str], _Allocation] = {}
    allocated_abi_ids: set[str] = set()

    for linked in manifest.fragments:
        fragment = _leaf(linked)
        for stream in fragment.core_streams:
            runtime_core_id = runtime_by_core.get(stream.logical_core)
            if runtime_core_id is None:
                raise SchemaError(
                    "allocation stream core has no runtime binding",
                    path=f"{path}.core_bindings",
                )
            for record_index, record in enumerate(stream.records):
                if record.opcode is not RecordOpcode.SRAM_ALLOC_AT:
                    continue
                relocations = {
                    relocation.operand_id: relocation
                    for relocation in stream.address_relocations
                    if relocation.record_index == record_index
                }
                if set(relocations) != {
                    SemanticOperandId.REGION_NAME,
                    SemanticOperandId.LABEL_SYMBOL,
                }:
                    raise SchemaError(
                        "SRAM_ALLOC_AT requires exact region and label relocations",
                        path=f"{path}.fragments",
                    )
                region_relocation = relocations[SemanticOperandId.REGION_NAME]
                label_relocation = relocations[SemanticOperandId.LABEL_SYMBOL]
                region_entry = definitions.get(region_relocation.symbol_ref)
                label_entry = definitions.get(label_relocation.symbol_ref)
                if region_entry is None or label_entry is None:
                    raise SchemaError(
                        "SRAM_ALLOC_AT references an undefined symbol",
                        path=f"{path}.program_symbol_definitions",
                    )
                label_index, label_definition = label_entry
                _region_index, region_definition = region_entry
                closure_key = (
                    fragment.id,
                    stream.logical_core,
                    record_index,
                    SemanticOperandId.LABEL_SYMBOL,
                )
                closure = address_bindings.get(closure_key)
                if closure is None or len(closure.buffer_abi_ids) != 1:
                    raise SchemaError(
                        "SRAM_ALLOC_AT label requires one exact BufferABI closure",
                        path=f"{path}.address_operand_bindings",
                    )
                abi = abis.get(closure.buffer_abi_ids[0])
                if abi is None:
                    raise SchemaError(
                        "SRAM_ALLOC_AT closure references an unknown BufferABI",
                        path=f"{path}.address_operand_bindings",
                    )
                region_closure = address_bindings.get(
                    (
                        fragment.id,
                        stream.logical_core,
                        record_index,
                        SemanticOperandId.REGION_NAME,
                    )
                )
                if (
                    region_closure is None
                    or region_closure.buffer_abi_ids != (abi.id,)
                ):
                    raise SchemaError(
                        "SRAM_ALLOC_AT region/label closures disagree",
                        path=f"{path}.address_operand_bindings",
                    )
                operands = {operand.name: operand for operand in record.operands}
                region_offset = operands["region_offset_bytes"].literal_value
                size_bytes = operands["size_bytes"].literal_value
                if type(region_offset) is not int or type(size_bytes) is not int:
                    raise SchemaError(
                        "SRAM_ALLOC_AT span must be literal integers",
                        path=f"{path}.fragments",
                    )
                if (
                    label_definition.symbol.kind
                    is not ProgramSymbolKind.SRAM_LABEL
                    or label_definition.symbol.source_ref != abi.storage_id
                    or region_definition.symbol.kind
                    is not ProgramSymbolKind.SRAM_REGION
                    or region_definition.symbol.source_ref != abi.region_ref
                    or abi.logical_core != stream.logical_core
                    or region_offset != abi.region_offset_bytes
                    or size_bytes != abi.size_bytes
                    or region_offset > region_definition.size_bytes
                    or size_bytes > region_definition.size_bytes - region_offset
                ):
                    raise SchemaError(
                        "SRAM_ALLOC_AT does not exactly preserve symbol, core, BufferABI and region span",
                        path=f"{path}.fragments",
                    )
                key = (runtime_core_id, label_definition.symbol.id)
                if key in allocations or abi.id in allocated_abi_ids:
                    raise SchemaError(
                        "each core/label and BufferABI requires exactly one SRAM_ALLOC_AT",
                        path=f"{path}.fragments",
                    )
                allocated_abi_ids.add(abi.id)
                allocations[key] = _Allocation(
                    runtime_core_id,
                    label_definition.symbol.id,
                    label_index,
                    label_definition,
                    region_definition.symbol.id,
                    region_definition,
                    abi,
                    region_offset,
                    size_bytes,
                )

    # MOE_SWIZZLE BORROWED roots are external host-visible SRAM spans.  They
    # deliberately have no ALLOC/FREE records, but still require one exact
    # canonical label, region, runtime core and physical span for ProgramIo.
    if manifest.producer_pass in (
        "moe_swizzle_standard_linker",
        "moe_swizzle_calibration_standard_linker",
    ):
        for abi in sorted(abis.values(), key=lambda item: item.id):
            if (
                abi.alias_of is not None
                or abi.ownership is not BufferOwnership.BORROWED
                or abi.id in allocated_abi_ids
            ):
                continue
            root_alias_bindings = {
                item.binding_id for item in abis.values()
                if item.alias_of == abi.binding_id
                and item.storage_id == abi.storage_id
                and item.region_offset_bytes == abi.region_offset_bytes
            }
            if manifest.producer_pass == "moe_swizzle_calibration_standard_linker":
                root_alias_bindings.add(abi.binding_id)
            labels = tuple(sorted((
                (index, definition)
                for index, definition in enumerate(
                    manifest.program_symbol_definitions
                )
                if definition.symbol.kind
                is ProgramSymbolKind.ABSOLUTE_ADDRESS
                and definition.symbol.source_ref in root_alias_bindings
                and abi.logical_core in definition.logical_cores
            ), key=lambda item: item[1].symbol.id))
            regions = tuple(
                definition
                for definition in manifest.program_symbol_definitions
                if definition.symbol.kind is ProgramSymbolKind.SRAM_REGION
                and definition.symbol.source_ref == abi.region_ref
                and abi.logical_core in definition.logical_cores
            )
            runtime_core_id = runtime_by_core.get(abi.logical_core)
            if not labels or len(regions) != 1 or runtime_core_id is None:
                raise SchemaError(
                    "external MoE BORROWED root lacks exact label/region/core closure",
                    path=f"{path}.fragments",
                )
            label_index, label_definition = labels[0]
            region_definition = regions[0]
            if (
                any(
                    item.value
                    != region_definition.value + abi.region_offset_bytes
                    for _index, item in labels
                )
                or label_definition.value
                != region_definition.value + abi.region_offset_bytes
                or abi.region_offset_bytes > region_definition.size_bytes
                or abi.size_bytes
                > region_definition.size_bytes - abi.region_offset_bytes
            ):
                raise SchemaError(
                    "external MoE BORROWED root escapes its SRAM region",
                    path=f"{path}.fragments",
                )
            key = (runtime_core_id, label_definition.symbol.id)
            if key in allocations:
                raise SchemaError(
                    "external MoE BORROWED root duplicates an allocation",
                    path=f"{path}.fragments",
                )
            allocated_abi_ids.add(abi.id)
            allocations[key] = _Allocation(
                runtime_core_id, label_definition.symbol.id, label_index,
                label_definition, region_definition.symbol.id,
                region_definition, abi, abi.region_offset_bytes, abi.size_bytes,
            )

    storage_ids = {abi.storage_id for abi in abis.values()}
    calibration_borrowed_storage_ids: set[str] = set()
    if manifest.producer_pass == "moe_swizzle_calibration_standard_linker":
        borrowed = tuple(
            abi for abi in abis.values()
            if abi.alias_of is None
            and abi.ownership is BufferOwnership.BORROWED
        )
        calibration_borrowed_storage_ids = {
            abi.storage_id for abi in borrowed
        }
        for abi in borrowed:
            input_labels = tuple(
                definition
                for definition in manifest.program_symbol_definitions
                if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
                and definition.symbol.source_ref == abi.storage_id
                and definition.logical_cores == (abi.logical_core,)
            )
            direct_allocations = tuple(
                allocation for allocation in allocations.values()
                if allocation.abi.id == abi.id
                and allocation.abi.ownership is BufferOwnership.BORROWED
            )
            if (
                len(input_labels) != 1
                or input_labels[0].value != 0
                or input_labels[0].size_bytes != 0
                or len(direct_allocations) != 1
            ):
                raise SchemaError(
                    "calibration BORROWED input label lacks exact direct-ABS closure",
                    path=f"{path}.fragments",
                )
    expected_labels = {
        (runtime_by_core[core], definition.symbol.id)
        for definition in manifest.program_symbol_definitions
        if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
        and definition.symbol.source_ref in storage_ids
        and definition.symbol.source_ref not in calibration_borrowed_storage_ids
        for core in definition.logical_cores
    }
    if manifest.producer_pass in (
        "moe_swizzle_standard_linker",
        "moe_swizzle_calibration_standard_linker",
    ):
        expected_labels.update(
            key for key, allocation in allocations.items()
            if allocation.abi.ownership is BufferOwnership.BORROWED
        )
    if set(allocations) != expected_labels:
        raise SchemaError(
            "storage-backed SRAM_LABEL definitions must be covered by one exact per-core SRAM_ALLOC_AT",
            path=f"{path}.fragments",
        )
    if allocated_abi_ids != nonalias_abi_ids:
        raise SchemaError(
            "every non-alias BufferABI requires one exact SRAM_ALLOC_AT and aliases require none",
            path=f"{path}.fragments",
        )
    return allocations, abis


def _entry_order(entry: ProgramSramInitialization | ProgramOutputProbe) -> tuple:
    target = entry.target
    if type(target) is ProgramSramTarget:
        target_key = (
            0,
            target.runtime_core_id,
            target.finalized_symbol_index,
            target.program_symbol_ref,
        )
    else:
        assert type(target) is ProgramHbmTarget
        target_key = (
            1, 0, target.finalized_symbol_index, target.program_symbol_ref
        )
    return (
        *target_key,
        entry.offset_bytes,
        entry.length_bytes,
        entry.id,
    )


def _validate_nonoverlap(
    spans: list[tuple[int, str, int, int, str]],
    path: str,
) -> None:
    previous: tuple[int, str, int, int, str] | None = None
    for span in sorted(spans):
        if (
            previous is not None
            and span[0] == previous[0]
            and span[1] == previous[1]
            and span[2] < previous[3]
        ):
            raise SchemaError(
                "physical byte ranges must not overlap",
                path=path,
            )
        previous = span


def _validate_allocation_nonoverlap(
    allocations: tuple[_Allocation, ...],
    producer_pass: str,
    core_stream_count: int,
    path: str,
) -> None:
    """Admit lifetime-disjoint UNFUSED reuse only for the four-stream scale ABI."""

    for index, left in enumerate(allocations):
        for right in allocations[index + 1 :]:
            physical_overlap = (
                left.runtime_core_id == right.runtime_core_id
                and left.region_symbol_ref == right.region_symbol_ref
                and left.absolute_start
                < right.absolute_start + right.size_bytes
                and right.absolute_start
                < left.absolute_start + left.size_bytes
            )
            if not physical_overlap:
                continue
            lifetime_disjoint = (
                left.abi.lifetime_end_exclusive
                <= right.abi.lifetime_start
                or right.abi.lifetime_end_exclusive
                <= left.abi.lifetime_start
            )
            if (
                producer_pass
                == "unfused_comparison_standard_linker"
                and core_stream_count == 4
                and lifetime_disjoint
                or producer_pass == "moe_swizzle_standard_linker"
                and lifetime_disjoint
            ):
                continue
            raise SchemaError(
                "physical allocation ranges overlap during live intervals",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class ProgramIoContract:
    """One ProgramArtifact-bound immutable host IO sidecar.

    Exact first-read dominance is intentionally a producer-layer obligation:
    ``LinkedProgramManifest`` carries action IDs and linked record order, but
    not the GlobalAction objects/dependency graph required to prove it.
    """

    schema_version: str
    producer_pass: str
    id: str
    mode: ProgramIoMode
    source_linked_manifest_id: str
    source_linked_manifest_digest: str
    program_artifact_sha256: str
    blobs: tuple[ProgramBlob, ...]
    initializations: tuple[ProgramSramInitialization, ...]
    output_probes: tuple[ProgramOutputProbe, ...]

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        mode: ProgramIoMode,
        source_manifest: LinkedProgramManifest,
        program_artifact_sha256: str,
        blobs: tuple[ProgramBlob, ...],
        initializations: tuple[ProgramSramInitialization, ...],
        output_probes: tuple[ProgramOutputProbe, ...],
    ) -> "ProgramIoContract":
        semantic_key = {
            "mode": mode,
            "source_linked_manifest_id": source_manifest.id,
            "source_linked_manifest_digest": canonical_digest(source_manifest),
            "program_artifact_sha256": program_artifact_sha256,
            "blobs": tuple(sorted(blobs, key=lambda blob: blob.id)),
            "initializations": tuple(sorted(initializations, key=_entry_order)),
            "output_probes": tuple(sorted(output_probes, key=_entry_order)),
        }
        return cls(
            schema_version=PROGRAM_IO_CONTRACT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "program_io_contract",
                semantic_key,
                schema_version=PROGRAM_IO_CONTRACT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "mode",
                "source_linked_manifest_id",
                "source_linked_manifest_digest",
                "program_artifact_sha256",
                "blobs",
                "initializations",
                "output_probes",
            )
        }

    def validate(self, path: str = "program_io_contract") -> None:
        if self.schema_version != PROGRAM_IO_CONTRACT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if not isinstance(self.mode, ProgramIoMode):
            raise SchemaError("must be a ProgramIoMode", path=f"{path}.mode")
        validate_nonempty(
            self.source_linked_manifest_id,
            f"{path}.source_linked_manifest_id",
        )
        _validate_sha256(
            self.source_linked_manifest_digest,
            f"{path}.source_linked_manifest_digest",
        )
        _validate_sha256(
            self.program_artifact_sha256,
            f"{path}.program_artifact_sha256",
        )
        if not self.blobs or not self.initializations or not self.output_probes:
            raise SchemaError(
                "MVP contract requires blobs, initializations and output probes",
                path=path,
            )
        blob_ids: list[str] = []
        blobs: dict[str, ProgramBlob] = {}
        for index, blob in enumerate(self.blobs):
            if type(blob) is not ProgramBlob:
                raise SchemaError(
                    "must be a ProgramBlob",
                    path=f"{path}.blobs[{index}]",
                )
            blob.validate(f"{path}.blobs[{index}]")
            blob_ids.append(blob.id)
            blobs[blob.id] = blob
        if blob_ids != sorted(set(blob_ids)):
            raise SchemaError(
                "blobs must have unique canonical ids",
                path=f"{path}.blobs",
            )

        used_blobs: set[str] = set()
        previous_initialization_key: tuple | None = None
        initialization_identity: set[tuple[object, ...]] = set()
        for index, entry in enumerate(self.initializations):
            entry.validate(f"{path}.initializations[{index}]")
            key = _entry_order(entry)
            if previous_initialization_key is not None and not (
                previous_initialization_key < key
            ):
                raise SchemaError(
                    "initializations must be unique and canonical",
                    path=f"{path}.initializations",
                )
            previous_initialization_key = key
            target = entry.target
            identity = (
                target.kind,
                getattr(target, "runtime_core_id", None),
                target.program_symbol_ref,
                entry.offset_bytes,
                entry.length_bytes,
            )
            if identity in initialization_identity:
                raise SchemaError(
                    "duplicate initialization byte range",
                    path=f"{path}.initializations[{index}]",
                )
            initialization_identity.add(identity)
            blob = blobs.get(entry.blob_ref)
            if blob is None or blob.length_bytes != entry.length_bytes:
                raise SchemaError(
                    "blob must exist and exactly match initialization length",
                    path=f"{path}.initializations[{index}].blob_ref",
                )
            used_blobs.add(entry.blob_ref)

        previous_probe_key: tuple | None = None
        probe_identity: set[tuple[object, ...]] = set()
        for index, entry in enumerate(self.output_probes):
            entry.validate(f"{path}.output_probes[{index}]")
            key = _entry_order(entry)
            if previous_probe_key is not None and not previous_probe_key < key:
                raise SchemaError(
                    "output probes must be unique and canonical",
                    path=f"{path}.output_probes",
                )
            previous_probe_key = key
            target = entry.target
            identity = (
                target.kind,
                getattr(target, "runtime_core_id", None),
                target.program_symbol_ref,
                entry.offset_bytes,
                entry.length_bytes,
            )
            if identity in probe_identity:
                raise SchemaError(
                    "duplicate output-probe byte range",
                    path=f"{path}.output_probes[{index}]",
                )
            probe_identity.add(identity)
            blob = blobs.get(entry.blob_ref)
            if blob is None or blob.length_bytes != entry.length_bytes:
                raise SchemaError(
                    "blob must exist and exactly match output-probe length",
                    path=f"{path}.output_probes[{index}].blob_ref",
                )
            used_blobs.add(entry.blob_ref)
        if used_blobs != set(blobs):
            raise SchemaError(
                "blobs must be used exactly by initializations/probes",
                path=f"{path}.blobs",
            )

        if self.mode is ProgramIoMode.FUNCTIONAL and any(
            entry.purpose is ProgramIoPurpose.TIMING_PARTIAL
            for entry in self.initializations
        ):
            raise SchemaError(
                "functional mode forbids timing_partial initialization",
                path=f"{path}.initializations",
            )
        expected_id = stable_artifact_id(
            "program_io_contract",
            self._semantic_key(),
            schema_version=PROGRAM_IO_CONTRACT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        manifest: LinkedProgramManifest,
        path: str = "program_io_contract",
    ) -> None:
        manifest.validate("linked_program_manifest")
        self.validate(path)
        if (
            self.source_linked_manifest_id != manifest.id
            or self.source_linked_manifest_digest != canonical_digest(manifest)
        ):
            raise SchemaError(
                "source id/digest do not identify the exact linked manifest",
                path=f"{path}.source_linked_manifest_id",
            )

        allocations, abis = _manifest_allocations(
            manifest,
            "linked_program_manifest",
        )
        state_abis, state_directions = _state_abis_and_directions(manifest)
        definitions = {
            definition.symbol.id: (index, definition)
            for index, definition in enumerate(
                manifest.program_symbol_definitions
            )
        }
        _validate_allocation_nonoverlap(
            tuple(allocations.values()),
            manifest.producer_pass,
            len(manifest.core_streams),
            f"{path}.output_probes",
        )
        initialization_ranges: list[tuple[int, str, int, int, str]] = []
        probe_ranges: list[tuple[int, str, int, int, str]] = []
        coverage: dict[str, list[tuple[int, int]]] = {}

        def validate_sram_entry(
            entry: ProgramSramInitialization | ProgramOutputProbe,
            target: ProgramSramTarget,
            entry_path: str,
        ) -> tuple[_Allocation, BufferABI, int, int]:
            allocation = allocations.get(
                (target.runtime_core_id, target.program_symbol_ref)
            )
            abi = abis.get(target.buffer_abi_id)
            moe_alias = (
                allocation is not None
                and abi is not None
                and manifest.producer_pass == "moe_swizzle_standard_linker"
                and abi.ownership is BufferOwnership.ALIASED
                and abi.alias_of == allocation.abi.binding_id
                and abi.storage_id == allocation.abi.storage_id
                and abi.logical_core == allocation.abi.logical_core
                and abi.region_ref == allocation.abi.region_ref
                and abi.region_offset_bytes >= allocation.abi.region_offset_bytes
                and abi.region_offset_bytes + abi.size_bytes
                <= allocation.abi.region_offset_bytes + allocation.abi.size_bytes
                and allocation.abi.layout.startswith("moe_swizzle_")
                and allocation.abi.layout.endswith("_root/v1")
                and abi.layout
                == f"{allocation.abi.layout.removesuffix('_root/v1')}_subview/v1"
            )
            if (
                allocation is None or abi is None
                or allocation.abi != abi and not moe_alias
            ):
                raise SchemaError(
                    "core/label does not resolve to the exact BufferABI allocation",
                    path=entry_path,
                )
            definition = allocation.label_definition
            if (
                target.finalized_symbol_index
                != allocation.label_definition_index
                or target.expected_symbol_name != definition.name
                or target.storage_id != abi.storage_id
                or target.value_id != abi.value_id
                or target.tensor_slice != abi.tensor_slice
                or target.dtype is not abi.dtype
                or target.layout != abi.layout
            ):
                raise SchemaError(
                    "symbol/index/name and BufferABI tensor metadata must match exactly",
                    path=entry_path,
                )
            end = _checked_end(
                entry.offset_bytes,
                entry.length_bytes,
                entry_path,
            )
            if end > allocation.size_bytes:
                raise SchemaError(
                    "byte range exceeds its SRAM_ALLOC_AT allocation",
                    path=entry_path,
                )
            absolute_start = allocation.absolute_start + entry.offset_bytes
            return allocation, abi, absolute_start, absolute_start + entry.length_bytes

        def validate_hbm_entry(
            entry: ProgramSramInitialization | ProgramOutputProbe,
            target: ProgramHbmTarget,
            entry_path: str,
        ) -> StateABI:
            abi = state_abis.get(target.state_abi_id)
            definition_entry = definitions.get(target.program_symbol_ref)
            if abi is None or definition_entry is None:
                raise SchemaError(
                    "HBM target does not resolve to StateABI and final symbol",
                    path=entry_path,
                )
            definition_index, definition = definition_entry
            if (
                target.state_ref != abi.state_ref
                or target.hbm_binding_ref != abi.hbm_binding_ref
                or definition_index != target.finalized_symbol_index
                or definition.name != target.expected_symbol_name
                or definition.symbol.kind
                is not ProgramSymbolKind.ABSOLUTE_ADDRESS
                or definition.symbol.source_ref != abi.hbm_binding_ref
            ):
                raise SchemaError(
                    "HBM target must exactly preserve StateABI and final symbol identity",
                    path=entry_path,
                )
            if entry.offset_bytes != 0 or entry.length_bytes != abi.size_bytes:
                raise SchemaError(
                    "HBM ProgramIo requires one whole-state byte range",
                    path=entry_path,
                )
            return abi

        for index, entry in enumerate(self.initializations):
            entry_path = f"{path}.initializations[{index}]"
            target = entry.target
            if type(target) is ProgramHbmTarget:
                state_abi = validate_hbm_entry(entry, target, entry_path)
                if (
                    state_abi.access is PersistentStateAccess.RESERVED
                    or RecordOpcode.LSU_LOAD
                    not in state_directions[state_abi.id]
                ):
                    raise SchemaError(
                        "HBM seed requires a readable state with an LSU_LOAD",
                        path=entry_path,
                    )
                continue

            assert type(target) is ProgramSramTarget
            allocation, abi, start, end = validate_sram_entry(
                entry, target, entry_path
            )
            if (
                entry.purpose
                in (ProgramIoPurpose.ACTIVATION, ProgramIoPurpose.WEIGHT)
                and abi.ownership is not BufferOwnership.BORROWED
            ):
                raise SchemaError(
                    "activation/weight initialization requires BORROWED BufferABI",
                    path=f"{entry_path}.purpose",
                )
            if entry.purpose is ProgramIoPurpose.TIMING_PARTIAL:
                if (
                    self.mode is not ProgramIoMode.TIMING
                    or abi.ownership is not BufferOwnership.OWNED
                ):
                    raise SchemaError(
                        "timing_partial requires timing mode and OWNED BufferABI",
                        path=f"{entry_path}.purpose",
                    )
            elif (
                self.mode is ProgramIoMode.FUNCTIONAL
                and abi.ownership is not BufferOwnership.BORROWED
            ):
                raise SchemaError(
                    "functional mode forbids initialization of OWNED storage",
                    path=f"{entry_path}.target.buffer_abi_id",
                )
            coverage.setdefault(abi.id, []).append(
                (entry.offset_bytes, entry.offset_bytes + entry.length_bytes)
            )
            initialization_ranges.append(
                (
                    target.runtime_core_id,
                    allocation.region_symbol_ref,
                    start,
                    end,
                    entry.id,
                )
            )

        borrowed = {
            abi.id: abi
            for abi in abis.values()
            if abi.ownership is BufferOwnership.BORROWED
        }
        for abi_id, abi in borrowed.items():
            intervals = sorted(coverage.get(abi_id, ()))
            cursor = 0
            for start, end in intervals:
                if start > cursor:
                    break
                cursor = max(cursor, end)
            if cursor != abi.size_bytes:
                raise SchemaError(
                    "every manifested BORROWED BufferABI requires complete host initialization; exact first-read dominance remains a producer-layer check",
                    path=f"{path}.initializations",
                )

        for index, entry in enumerate(self.output_probes):
            entry_path = f"{path}.output_probes[{index}]"
            target = entry.target
            if type(target) is ProgramHbmTarget:
                state_abi = validate_hbm_entry(entry, target, entry_path)
                if (
                    state_abi.access is not PersistentStateAccess.READ_WRITE
                    or RecordOpcode.LSU_STORE
                    not in state_directions[state_abi.id]
                ):
                    raise SchemaError(
                        "HBM probe requires READ_WRITE state with an LSU_STORE",
                        path=entry_path,
                    )
                continue

            assert type(target) is ProgramSramTarget
            allocation, abi, start, end = validate_sram_entry(
                entry, target, entry_path
            )
            probe_owned = (
                abi.ownership is BufferOwnership.OWNED
                or manifest.producer_pass == "moe_swizzle_standard_linker"
                and abi.ownership is BufferOwnership.ALIASED
                and allocation.abi.ownership is BufferOwnership.OWNED
                and abi.alias_of == allocation.abi.binding_id
            )
            if not probe_owned:
                raise SchemaError(
                    "output probe requires OWNED BufferABI or exact MoE terminal alias",
                    path=f"{entry_path}.target.buffer_abi_id",
                )
            probe_ranges.append(
                (
                    target.runtime_core_id,
                    allocation.region_symbol_ref,
                    start,
                    end,
                    entry.id,
                )
            )

        _validate_nonoverlap(
            initialization_ranges,
            f"{path}.initializations",
        )
        _validate_nonoverlap(probe_ranges, f"{path}.output_probes")

        # AFTER_PROGRAM is sound with UNFUSED reuse only when every overlapping
        # allocation finished before the probed terminal storage became live.
        all_allocations = tuple(allocations.values())
        for index, entry in enumerate(self.output_probes):
            sram_target = entry.target
            if type(sram_target) is ProgramHbmTarget:
                continue
            assert type(sram_target) is ProgramSramTarget
            target = allocations[
                (sram_target.runtime_core_id, sram_target.program_symbol_ref)
            ]
            target_start = target.absolute_start + entry.offset_bytes
            target_end = target_start + entry.length_bytes
            for other in all_allocations:
                if (
                    other is not target
                    and other.runtime_core_id == target.runtime_core_id
                    and other.region_symbol_ref == target.region_symbol_ref
                    and target_start < other.absolute_start + other.size_bytes
                    and other.absolute_start < target_end
                    and not (
                        manifest.producer_pass
                        == "unfused_comparison_standard_linker"
                        and other.abi.lifetime_end_exclusive
                        <= target.abi.lifetime_start
                        or manifest.producer_pass
                        == "moe_swizzle_standard_linker"
                        and other.abi.lifetime_end_exclusive
                        <= target.abi.lifetime_start
                    )
                ):
                    raise SchemaError(
                        "after-program probe rejects physical allocation reuse/aliasing",
                        path=f"{path}.output_probes[{index}].capture",
                    )
