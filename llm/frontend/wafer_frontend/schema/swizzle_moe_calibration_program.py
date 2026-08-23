"""Typed source and standard-wrapper carriers for isolated MoE calibration.

This module contains no lowering policy.  It freezes the exact 28-family
record skeleton and closes a later production lower/link/ProgramIo result back
to one calibrated primitive and one final C2 source program.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from ..errors import SchemaError
from .artifact_manifest import (
    CommandFragment,
    FragmentKind,
    LinkedProgramManifest,
    ProgramSymbolKind,
    RecordOpcode,
)
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import LogicalCoreRef
from .ir2 import BufferOwnership
from .program_io import (
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramSramTarget,
)
from .serde import canonical_digest
from .swizzle_moe_calibration import (
    MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES,
    MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES,
    MoeCalibrationKind,
)


MOE_SWIZZLE_CALIBRATION_PROGRAM_SOURCE_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_calibration_program_source/v1alpha1"
)
MOE_SWIZZLE_CALIBRATION_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_calibration_standard_linked_program/v1alpha1"
)
_SOURCE_PRODUCER = "build_moe_swizzle_calibration_program_source"
_LOWERING_PRODUCER = "moe_swizzle_calibration_standard_lowering"
_LINKER_PRODUCER = "moe_swizzle_calibration_standard_linker"
_PROGRAM_IO_PRODUCER = "build_moe_swizzle_calibration_program_io"


_TARGET_OPCODE = {
    MoeCalibrationKind.GROUP_GEMM: RecordOpcode.MATMUL,
    MoeCalibrationKind.SWIGLU_GROUP: RecordOpcode.SWIGLU,
    MoeCalibrationKind.DTE_LAUNCH: RecordOpcode.DTE_RECV,
    MoeCalibrationKind.DTE_SYNC: RecordOpcode.DTE_WAIT,
    MoeCalibrationKind.DTE_HOP: RecordOpcode.DTE_SEND,
    MoeCalibrationKind.SESSION_OPEN: RecordOpcode.DTE_RECV,
    MoeCalibrationKind.SESSION_RETIRE: RecordOpcode.DTE_WAIT,
    MoeCalibrationKind.LOCAL_COPY: RecordOpcode.DTE_ISSUE,
    MoeCalibrationKind.SRAM_ALLOC: RecordOpcode.SRAM_ALLOC_AT,
    MoeCalibrationKind.SRAM_BIND: RecordOpcode.SRAM_BIND,
    MoeCalibrationKind.SRAM_FREE: RecordOpcode.SRAM_FREE,
    MoeCalibrationKind.EVENT_SET: RecordOpcode.EVENT_SET,
    MoeCalibrationKind.EVENT_WAIT: RecordOpcode.EVENT_WAIT,
    MoeCalibrationKind.TERMINAL_DONE: None,
}

_LIFECYCLE_COPY = (
    RecordOpcode.DTE_ISSUE,
    RecordOpcode.DTE_WAIT,
    RecordOpcode.SRAM_ALLOC_AT,
)
_P2P = (
    RecordOpcode.DTE_SEND,
    RecordOpcode.DTE_RECV,
    RecordOpcode.DTE_WAIT,
    RecordOpcode.SRAM_ALLOC_AT,
)
_EXPECTED_QUOTIENT = {
    MoeCalibrationKind.GROUP_GEMM: (
        RecordOpcode.MATMUL,
        RecordOpcode.SRAM_BIND,
        RecordOpcode.SRAM_ALLOC_AT,
    ),
    MoeCalibrationKind.SWIGLU_GROUP: (
        RecordOpcode.SWIGLU,
        RecordOpcode.SRAM_BIND,
        RecordOpcode.SRAM_ALLOC_AT,
    ),
    MoeCalibrationKind.DTE_LAUNCH: _P2P,
    MoeCalibrationKind.DTE_SYNC: _P2P,
    MoeCalibrationKind.DTE_HOP: _P2P,
    MoeCalibrationKind.SESSION_OPEN: _P2P,
    MoeCalibrationKind.SESSION_RETIRE: _P2P,
    MoeCalibrationKind.LOCAL_COPY: _LIFECYCLE_COPY,
    MoeCalibrationKind.SRAM_ALLOC: _LIFECYCLE_COPY,
    MoeCalibrationKind.SRAM_BIND: (
        RecordOpcode.SWIGLU,
        RecordOpcode.SRAM_BIND,
        RecordOpcode.SRAM_ALLOC_AT,
    ),
    MoeCalibrationKind.SRAM_FREE: (
        RecordOpcode.DTE_ISSUE,
        RecordOpcode.DTE_WAIT,
        RecordOpcode.SRAM_ALLOC_AT,
        RecordOpcode.SRAM_ALLOC_AT,
        RecordOpcode.SRAM_FREE,
    ),
    MoeCalibrationKind.EVENT_SET: (
        RecordOpcode.DTE_ISSUE,
        RecordOpcode.DTE_WAIT,
        RecordOpcode.EVENT_SET,
        RecordOpcode.EVENT_WAIT,
        RecordOpcode.SRAM_ALLOC_AT,
    ),
    MoeCalibrationKind.EVENT_WAIT: (
        RecordOpcode.DTE_ISSUE,
        RecordOpcode.DTE_WAIT,
        RecordOpcode.EVENT_SET,
        RecordOpcode.EVENT_WAIT,
        RecordOpcode.SRAM_ALLOC_AT,
    ),
    MoeCalibrationKind.TERMINAL_DONE: _LIFECYCLE_COPY,
}
_TRANSPORT_KINDS = frozenset((
    MoeCalibrationKind.DTE_LAUNCH,
    MoeCalibrationKind.DTE_SYNC,
    MoeCalibrationKind.DTE_HOP,
    MoeCalibrationKind.SESSION_OPEN,
    MoeCalibrationKind.SESSION_RETIRE,
))


def required_moe_swizzle_calibration_target_opcode(
    kind: MoeCalibrationKind,
) -> RecordOpcode | None:
    if type(kind) is not MoeCalibrationKind:
        raise SchemaError("must use a typed calibration kind", path="kind")
    return _TARGET_OPCODE[kind]


def expected_moe_swizzle_calibration_record_quotient(
    kind: MoeCalibrationKind,
) -> tuple[tuple[RecordOpcode, int], ...]:
    if type(kind) is not MoeCalibrationKind:
        raise SchemaError("must use a typed calibration kind", path="kind")
    return tuple(sorted(
        Counter(_EXPECTED_QUOTIENT[kind]).items(),
        key=lambda item: int(item[0]),
    ))


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationRecordCount:
    opcode: RecordOpcode
    count: int

    def validate(self, path: str = "moe_swizzle_calibration_record_count") -> None:
        if type(self.opcode) is not RecordOpcode:
            raise SchemaError("must use a RecordOpcode", path=f"{path}.opcode")
        validate_uint64(self.count, f"{path}.count")
        if self.count == 0:
            raise SchemaError("record count must be positive", path=f"{path}.count")


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationTargetRecord:
    logical_core: LogicalCoreRef
    record_index: int
    opcode: RecordOpcode

    def validate(self, path: str = "moe_swizzle_calibration_target_record") -> None:
        if type(self.logical_core) is not LogicalCoreRef:
            raise SchemaError("must use a LogicalCoreRef", path=f"{path}.logical_core")
        self.logical_core.validate(f"{path}.logical_core")
        validate_uint64(self.record_index, f"{path}.record_index")
        if type(self.opcode) is not RecordOpcode:
            raise SchemaError("must use a RecordOpcode", path=f"{path}.opcode")


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationProgramSource:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_linked_program_id: str
    source_linked_program_digest: str
    source_fragment_id: str
    kind: MoeCalibrationKind
    shape: tuple[int, int, int] | None
    target_logical_core: LogicalCoreRef
    target_runtime_core_id: int
    target_records: tuple[MoeSwizzleCalibrationTargetRecord, ...]
    record_quotient: tuple[MoeSwizzleCalibrationRecordCount, ...]
    auxiliary_opcodes: tuple[RecordOpcode, ...]
    source_route_resource_refs: tuple[str, ...]
    input_buffer_abi_ids: tuple[str, ...]
    output_buffer_abi_ids: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleCalibrationProgramSource":
        result = cls(
            MOE_SWIZZLE_CALIBRATION_PROGRAM_SOURCE_SCHEMA_VERSION,
            _SOURCE_PRODUCER,
            stable_artifact_id(
                "moe_swizzle_calibration_program_source",
                semantic,
                schema_version=MOE_SWIZZLE_CALIBRATION_PROGRAM_SOURCE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_ir1_id",
                "source_linked_program_id",
                "source_linked_program_digest",
                "source_fragment_id",
                "kind",
                "shape",
                "target_logical_core",
                "target_runtime_core_id",
                "target_records",
                "record_quotient",
                "auxiliary_opcodes",
                "source_route_resource_refs",
                "input_buffer_abi_ids",
                "output_buffer_abi_ids",
            )
        }

    def validate(self, path: str = "moe_swizzle_calibration_program_source") -> None:
        if (
            self.schema_version
            != MOE_SWIZZLE_CALIBRATION_PROGRAM_SOURCE_SCHEMA_VERSION
            or self.producer_pass != _SOURCE_PRODUCER
        ):
            raise SchemaError("unsupported calibration source schema/producer", path=path)
        for name in (
            "source_ir1_id", "source_linked_program_id", "source_fragment_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_linked_program_digest) is None:
            raise SchemaError(
                "source program digest must be lowercase SHA-256",
                path=f"{path}.source_linked_program_digest",
            )
        if type(self.kind) is not MoeCalibrationKind:
            raise SchemaError("must use a typed calibration kind", path=f"{path}.kind")
        shaped = self.kind in (
            MoeCalibrationKind.GROUP_GEMM,
            MoeCalibrationKind.SWIGLU_GROUP,
        )
        allowed_shapes = (
            MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES
            if self.kind is MoeCalibrationKind.GROUP_GEMM
            else MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES
        )
        if shaped:
            if self.shape not in allowed_shapes:
                raise SchemaError(
                    "shape is outside exact production calibration coverage",
                    path=f"{path}.shape",
                )
        elif self.shape is not None:
            raise SchemaError("fixed calibration forbids shape", path=f"{path}.shape")
        if type(self.target_logical_core) is not LogicalCoreRef:
            raise SchemaError("must use a LogicalCoreRef", path=f"{path}.target_logical_core")
        self.target_logical_core.validate(f"{path}.target_logical_core")
        validate_uint64(self.target_runtime_core_id, f"{path}.target_runtime_core_id")
        if self.target_runtime_core_id > 0xFFFF:
            raise SchemaError("runtime core must fit uint16", path=f"{path}.target_runtime_core_id")

        expected_target = required_moe_swizzle_calibration_target_opcode(self.kind)
        if expected_target is None:
            if self.target_records:
                raise SchemaError(
                    "terminal-done calibration has no frontend target record",
                    path=f"{path}.target_records",
                )
        elif len(self.target_records) != 1:
            raise SchemaError(
                "calibration requires exactly one target record",
                path=f"{path}.target_records",
            )
        target_keys = []
        for index, target in enumerate(self.target_records):
            if type(target) is not MoeSwizzleCalibrationTargetRecord:
                raise SchemaError("must use a typed target record", path=f"{path}.target_records[{index}]")
            target.validate(f"{path}.target_records[{index}]")
            if (
                target.logical_core != self.target_logical_core
                or target.opcode is not expected_target
            ):
                raise SchemaError(
                    "target record disagrees with kind/core",
                    path=f"{path}.target_records[{index}]",
                )
            target_keys.append((
                target.logical_core.die_id,
                target.logical_core.local_core_id,
                target.record_index,
                int(target.opcode),
            ))
        if target_keys != sorted(set(target_keys)):
            raise SchemaError("target records must be unique canonical", path=f"{path}.target_records")

        quotient_keys = []
        actual_quotient = []
        for index, item in enumerate(self.record_quotient):
            if type(item) is not MoeSwizzleCalibrationRecordCount:
                raise SchemaError("must use a typed record count", path=f"{path}.record_quotient[{index}]")
            item.validate(f"{path}.record_quotient[{index}]")
            quotient_keys.append(int(item.opcode))
            actual_quotient.append((item.opcode, item.count))
        if quotient_keys != sorted(set(quotient_keys)):
            raise SchemaError("record quotient must be unique canonical", path=f"{path}.record_quotient")
        expected_quotient = expected_moe_swizzle_calibration_record_quotient(self.kind)
        if tuple(actual_quotient) != expected_quotient:
            raise SchemaError(
                "record quotient is not the exact isolated skeleton",
                path=f"{path}.record_quotient",
            )
        expected_auxiliary = tuple(
            opcode for opcode, _count in expected_quotient
            if opcode is not expected_target
        )
        if self.auxiliary_opcodes != expected_auxiliary:
            raise SchemaError(
                "auxiliary opcode allowlist is not exact",
                path=f"{path}.auxiliary_opcodes",
            )

        for name in (
            "source_route_resource_refs",
            "input_buffer_abi_ids",
            "output_buffer_abi_ids",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise SchemaError("must be unique canonical", path=f"{path}.{name}")
            for index, value in enumerate(values):
                validate_nonempty(value, f"{path}.{name}[{index}]")
        if (len(self.source_route_resource_refs) == 1) != (self.kind in _TRANSPORT_KINDS):
            raise SchemaError(
                "one-hop transport kinds require exactly one route resource",
                path=f"{path}.source_route_resource_refs",
            )
        expected_inputs = 2 if self.kind is MoeCalibrationKind.GROUP_GEMM else 1
        if len(self.input_buffer_abi_ids) != expected_inputs or len(self.output_buffer_abi_ids) != 1:
            raise SchemaError(
                "isolated source input/output ABI arity is not exact",
                path=path,
            )
        if set(self.input_buffer_abi_ids).intersection(self.output_buffer_abi_ids):
            raise SchemaError("input/output ABI ids must be disjoint", path=path)
        expected_id = stable_artifact_id(
            "moe_swizzle_calibration_program_source",
            self._semantic(),
            schema_version=MOE_SWIZZLE_CALIBRATION_PROGRAM_SOURCE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable calibration source id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationStandardLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: MoeSwizzleCalibrationProgramSource
    fragment: CommandFragment
    manifest: LinkedProgramManifest
    program_io: ProgramIoContract | None

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleCalibrationStandardLinkedProgram":
        result = cls(
            MOE_SWIZZLE_CALIBRATION_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
            _LINKER_PRODUCER,
            stable_artifact_id(
                "moe_swizzle_calibration_standard_linked_program",
                semantic,
                schema_version=(
                    MOE_SWIZZLE_CALIBRATION_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION
                ),
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            "source": self.source,
            "fragment": self.fragment,
            "manifest": self.manifest,
            "program_io": self.program_io,
        }

    def validate(
        self, path: str = "moe_swizzle_calibration_standard_linked_program"
    ) -> None:
        if (
            self.schema_version
            != MOE_SWIZZLE_CALIBRATION_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION
            or self.producer_pass != _LINKER_PRODUCER
        ):
            raise SchemaError("unsupported calibration wrapper schema/producer", path=path)
        if type(self.source) is not MoeSwizzleCalibrationProgramSource:
            raise SchemaError("must use a calibration source", path=f"{path}.source")
        if type(self.fragment) is not CommandFragment:
            raise SchemaError("must use a CommandFragment", path=f"{path}.fragment")
        if type(self.manifest) is not LinkedProgramManifest:
            raise SchemaError("must use a LinkedProgramManifest", path=f"{path}.manifest")
        self.source.validate(f"{path}.source")
        self.fragment.validate(f"{path}.fragment")
        self.manifest.validate(f"{path}.manifest")
        if any(
            definition.symbol.kind is ProgramSymbolKind.SRAM_REGION
            and definition.name != definition.symbol.source_ref
            for definition in self.manifest.program_symbol_definitions
        ):
            raise SchemaError(
                "calibration SRAM region name must equal its hardware region ref",
                path=f"{path}.manifest.program_symbol_definitions",
            )
        program_symbols = {
            symbol.id: symbol for symbol in self.fragment.program_symbols
        }
        abis = {abi.id: abi for abi in self.fragment.buffer_abi}
        output_ids = set(self.source.output_buffer_abi_ids)
        owned = {
            abi.storage_id: abi
            for abi in abis.values()
            if abi.alias_of is None
            and abi.ownership is BufferOwnership.OWNED
        }
        lifetime_by_storage: dict[str, int] = {}
        freed_storage: set[str] = set()
        for stream in self.fragment.core_streams:
            for record in stream.records:
                if record.opcode not in (
                    RecordOpcode.SRAM_ALLOC_AT,
                    RecordOpcode.SRAM_FREE,
                ):
                    continue
                operands = {item.name: item for item in record.operands}
                name = (
                    "label_symbol"
                    if record.opcode is RecordOpcode.SRAM_ALLOC_AT else "symbol"
                )
                symbol = program_symbols.get(operands[name].symbol_ref)
                if symbol is None or symbol.kind is not ProgramSymbolKind.SRAM_LABEL:
                    raise SchemaError(
                        "lifecycle record lacks an exact SRAM label",
                        path=f"{path}.fragment.core_streams",
                    )
                storage_id = symbol.source_ref
                if record.opcode is RecordOpcode.SRAM_ALLOC_AT:
                    lifetime = operands["lifetime"].literal_value
                    if storage_id in lifetime_by_storage or type(lifetime) is not int:
                        raise SchemaError(
                            "owned storage requires one literal allocation lifetime",
                            path=f"{path}.fragment.core_streams",
                        )
                    lifetime_by_storage[storage_id] = lifetime
                else:
                    if storage_id in freed_storage:
                        raise SchemaError(
                            "owned storage may be freed at most once",
                            path=f"{path}.fragment.core_streams",
                        )
                    freed_storage.add(storage_id)
        if set(lifetime_by_storage) != set(owned):
            raise SchemaError(
                "every owned calibration storage requires one exact allocation",
                path=f"{path}.fragment.core_streams",
            )
        for storage_id, abi in owned.items():
            terminal = abi.id in output_ids
            if (
                lifetime_by_storage[storage_id] != (2 if terminal else 0)
                or (storage_id in freed_storage) != (not terminal)
            ):
                raise SchemaError(
                    "terminal output must be PERSISTENT/live and scratch TASK/freed",
                    path=f"{path}.fragment.core_streams",
                )
        if (
            self.fragment.producer_pass != _LOWERING_PRODUCER
            or self.fragment.kind is not FragmentKind.MOE_SWIZZLE_CALIBRATION
            or self.fragment.source_global_dag_id != self.source.id
            or self.manifest.producer_pass != _LINKER_PRODUCER
            or self.manifest.source_ir1_id != self.source.source_ir1_id
            or self.manifest.source_projection_id != self.source.id
            or self.manifest.source_schedule_set_id != self.source.id
            or self.manifest.source_global_dag_id != self.source.id
            or len(self.manifest.fragments) != 1
            or self.manifest.fragments[0].id != self.fragment.id
        ):
            raise SchemaError("isolated fragment/manifest lineage is not exact", path=path)

        quotient = Counter(
            record.opcode
            for stream in self.fragment.core_streams
            for record in stream.records
        )
        observed = tuple(sorted(
            ((opcode, count) for opcode, count in quotient.items()),
            key=lambda item: int(item[0]),
        ))
        declared = tuple((item.opcode, item.count) for item in self.source.record_quotient)
        if observed != declared:
            raise SchemaError("fragment record quotient drifted", path=f"{path}.fragment")
        streams = {item.logical_core: item for item in self.fragment.core_streams}
        target_stream = streams.get(self.source.target_logical_core)
        if target_stream is None:
            raise SchemaError("target core lacks a fragment stream", path=f"{path}.fragment")
        for index, target in enumerate(self.source.target_records):
            if (
                target.record_index >= len(target_stream.records)
                or target_stream.records[target.record_index].opcode is not target.opcode
            ):
                raise SchemaError("target record does not resolve exactly", path=f"{path}.source.target_records[{index}]")
        bindings = tuple(
            item for item in self.manifest.core_bindings
            if item.logical_core == self.source.target_logical_core
            and item.runtime_core_id == self.source.target_runtime_core_id
        )
        if len(bindings) != 1:
            raise SchemaError("target core/runtime binding is not exact", path=f"{path}.manifest.core_bindings")
        if self.source.kind is MoeCalibrationKind.TERMINAL_DONE:
            if self.source.target_logical_core not in self.manifest.envelope.terminal_cores:
                raise SchemaError("terminal target core is not a manifest terminal", path=f"{path}.manifest.envelope")

        abis = {item.id: item for item in self.fragment.buffer_abi}
        if not set((*self.source.input_buffer_abi_ids, *self.source.output_buffer_abi_ids)).issubset(abis):
            raise SchemaError("source IO ABI id is absent from fragment", path=f"{path}.fragment.buffer_abi")
        if any(
            abis[abi_id].alias_of is not None
            or abis[abi_id].ownership is not BufferOwnership.BORROWED
            for abi_id in self.source.input_buffer_abi_ids
        ):
            raise SchemaError("inputs must select BORROWED roots", path=f"{path}.source.input_buffer_abi_ids")
        for abi_id in self.source.output_buffer_abi_ids:
            abi = abis[abi_id]
            root = abis.get(abi.alias_of) if abi.alias_of is not None else abi
            if root is None or root.ownership is not BufferOwnership.OWNED:
                raise SchemaError("outputs must select OWNED storage", path=f"{path}.source.output_buffer_abi_ids")

        if self.program_io is not None:
            if type(self.program_io) is not ProgramIoContract:
                raise SchemaError("must use a ProgramIoContract", path=f"{path}.program_io")
            self.program_io.validate(f"{path}.program_io")
            if (
                self.program_io.producer_pass != _PROGRAM_IO_PRODUCER
                or self.program_io.mode is not ProgramIoMode.TIMING
                or self.program_io.source_linked_manifest_id != self.manifest.id
                or self.program_io.source_linked_manifest_digest
                != canonical_digest(self.manifest)
                or self.program_io.program_artifact_sha256 == "0" * 64
            ):
                raise SchemaError("actual-SHA ProgramIo linkage is not exact", path=f"{path}.program_io")
            initializations = tuple(
                item.target.buffer_abi_id for item in self.program_io.initializations
                if type(item.target) is ProgramSramTarget
            )
            probes = tuple(
                item.target.buffer_abi_id for item in self.program_io.output_probes
                if type(item.target) is ProgramSramTarget
            )
            if (
                tuple(sorted(initializations))
                != tuple(sorted((
                    *self.source.input_buffer_abi_ids,
                    *self.source.output_buffer_abi_ids,
                )))
                or tuple(sorted(probes)) != self.source.output_buffer_abi_ids
                or len(initializations) != len(self.program_io.initializations)
                or len(probes) != len(self.program_io.output_probes)
            ):
                raise SchemaError("ProgramIo source IO ABI closure is not exact", path=f"{path}.program_io")
            initialization_by_abi = {
                item.target.buffer_abi_id: item
                for item in self.program_io.initializations
            }
            probe_by_abi = {
                item.target.buffer_abi_id: item
                for item in self.program_io.output_probes
            }
            if (
                len(initialization_by_abi) != len(initializations)
                or len(probe_by_abi) != len(probes)
                or any(
                    initialization_by_abi[abi_id].purpose
                    is not ProgramIoPurpose.ACTIVATION
                    for abi_id in self.source.input_buffer_abi_ids
                )
            ):
                raise SchemaError(
                    "ProgramIo input initialization closure is not exact",
                    path=f"{path}.program_io.initializations",
                )
            for abi_id in self.source.output_buffer_abi_ids:
                initialization = initialization_by_abi[abi_id]
                probe = probe_by_abi[abi_id]
                if (
                    initialization.purpose is not ProgramIoPurpose.TIMING_PARTIAL
                    or initialization.target != probe.target
                    or initialization.offset_bytes != probe.offset_bytes
                    or initialization.length_bytes != probe.length_bytes
                    or initialization.blob_ref != probe.blob_ref
                ):
                    raise SchemaError(
                        "ProgramIo timing terminal sentinel/probe closure is not exact",
                        path=f"{path}.program_io",
                    )
        expected_id = stable_artifact_id(
            "moe_swizzle_calibration_standard_linked_program",
            self._semantic(),
            schema_version=(
                MOE_SWIZZLE_CALIBRATION_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION
            ),
        )
        if self.id != expected_id:
            raise SchemaError("unstable calibration wrapper id", path=f"{path}.id")


__all__ = [
    "MOE_SWIZZLE_CALIBRATION_PROGRAM_SOURCE_SCHEMA_VERSION",
    "MOE_SWIZZLE_CALIBRATION_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION",
    "MoeSwizzleCalibrationProgramSource",
    "MoeSwizzleCalibrationRecordCount",
    "MoeSwizzleCalibrationStandardLinkedProgram",
    "MoeSwizzleCalibrationTargetRecord",
    "expected_moe_swizzle_calibration_record_quotient",
    "required_moe_swizzle_calibration_target_opcode",
]
