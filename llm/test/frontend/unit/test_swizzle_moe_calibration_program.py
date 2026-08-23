from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    CommandFragment,
    FragmentKind,
    LinkedProgramManifest,
    ProgramSymbolKind,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.global_action import LogicalCoreRef
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramBlob,
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramOutputProbe,
    ProgramSramInitialization,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_calibration_program_io import (
    build_moe_swizzle_calibration_program_io,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES,
    MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES,
    MoeCalibrationKind,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration_program import (
    MOE_SWIZZLE_CALIBRATION_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
    MoeSwizzleCalibrationProgramSource,
    MoeSwizzleCalibrationRecordCount,
    MoeSwizzleCalibrationStandardLinkedProgram,
    MoeSwizzleCalibrationTargetRecord,
    expected_moe_swizzle_calibration_record_quotient,
    required_moe_swizzle_calibration_target_opcode,
)


_TRANSPORT = {
    MoeCalibrationKind.DTE_LAUNCH,
    MoeCalibrationKind.DTE_SYNC,
    MoeCalibrationKind.DTE_HOP,
    MoeCalibrationKind.SESSION_OPEN,
    MoeCalibrationKind.SESSION_RETIRE,
}


def _shape(kind: MoeCalibrationKind) -> tuple[int, int, int] | None:
    if kind is MoeCalibrationKind.GROUP_GEMM:
        return MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES[0]
    if kind is MoeCalibrationKind.SWIGLU_GROUP:
        return MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES[0]
    return None


def _source(kind: MoeCalibrationKind) -> MoeSwizzleCalibrationProgramSource:
    quotient = expected_moe_swizzle_calibration_record_quotient(kind)
    target_opcode = required_moe_swizzle_calibration_target_opcode(kind)
    return MoeSwizzleCalibrationProgramSource.create(
        source_ir1_id="production.c2.ir1",
        source_linked_program_id="production.c2.linked",
        source_linked_program_digest="a" * 64,
        source_fragment_id="production.c2.fragment",
        kind=kind,
        shape=_shape(kind),
        target_logical_core=LogicalCoreRef(0, 0),
        target_runtime_core_id=7,
        target_records=(
            ()
            if target_opcode is None
            else (
                MoeSwizzleCalibrationTargetRecord(
                    LogicalCoreRef(0, 0), 0, target_opcode
                ),
            )
        ),
        record_quotient=tuple(
            MoeSwizzleCalibrationRecordCount(opcode, count)
            for opcode, count in quotient
        ),
        auxiliary_opcodes=tuple(
            opcode for opcode, _count in quotient
            if opcode is not target_opcode
        ),
        source_route_resource_refs=(
            ("moe.scale.link.0.1",) if kind in _TRANSPORT else ()
        ),
        input_buffer_abi_ids=(
            ("input.a", "input.b")
            if kind is MoeCalibrationKind.GROUP_GEMM
            else ("input",)
        ),
        output_buffer_abi_ids=("output",),
    )


class MoeSwizzleCalibrationProgramSchemaTest(unittest.TestCase):
    def test_timing_program_io_seeds_only_the_exact_terminal_probe_range(self) -> None:
        abis = tuple(
            SimpleNamespace(id=name, size_bytes=size)
            for name, size in (
                ("lhs", 16), ("rhs", 32), ("output", 8), ("scratch", 64)
            )
        )
        wrapper = object.__new__(MoeSwizzleCalibrationStandardLinkedProgram)
        object.__setattr__(wrapper, "program_io", None)
        object.__setattr__(wrapper, "source", SimpleNamespace(
            input_buffer_abi_ids=("lhs", "rhs"),
            output_buffer_abi_ids=("output",),
        ))
        object.__setattr__(wrapper, "fragment", SimpleNamespace(buffer_abi=abis))
        object.__setattr__(wrapper, "manifest", SimpleNamespace())
        captured: dict[str, object] = {}

        def create_contract(**semantic: object) -> object:
            captured.update(semantic)
            return SimpleNamespace(validate_against=lambda *_args: None)

        module = sys.modules[build_moe_swizzle_calibration_program_io.__module__]
        with (
            patch.object(
                MoeSwizzleCalibrationStandardLinkedProgram,
                "validate",
                lambda self, path="": None,
            ),
            patch.object(module, "_target", lambda _source, abi: f"target:{abi.id}"),
            patch.object(
                ProgramBlob,
                "create",
                side_effect=lambda payload: SimpleNamespace(
                    id=f"blob:{len(payload)}", bytes=payload
                ),
            ),
            patch.object(
                ProgramSramInitialization,
                "create",
                side_effect=lambda **semantic: SimpleNamespace(**semantic),
            ),
            patch.object(
                ProgramOutputProbe,
                "create",
                side_effect=lambda **semantic: SimpleNamespace(**semantic),
            ),
            patch.object(ProgramIoContract, "create", side_effect=create_contract),
        ):
            build_moe_swizzle_calibration_program_io(wrapper, "1" * 64)

        initializations = captured["initializations"]
        probes = captured["output_probes"]
        self.assertEqual(len(initializations), 3)
        self.assertEqual(len(probes), 1)
        terminal = initializations[-1]
        probe = probes[0]
        self.assertEqual(
            tuple(item.purpose for item in initializations),
            (
                ProgramIoPurpose.ACTIVATION,
                ProgramIoPurpose.ACTIVATION,
                ProgramIoPurpose.TIMING_PARTIAL,
            ),
        )
        self.assertEqual(terminal.target, "target:output")
        self.assertEqual(terminal.target, probe.target)
        self.assertEqual(terminal.blob_ref, probe.blob_ref)
        self.assertEqual(terminal.length_bytes, probe.length_bytes)
        self.assertNotIn(
            "target:scratch",
            tuple(item.target for item in initializations) + (probe.target,),
        )

    def test_persistent_allocation_is_rejected_outside_exact_calibration_fragment(self) -> None:
        allocation = SimpleNamespace(
            opcode=RecordOpcode.SRAM_ALLOC_AT,
            operands=tuple(
                SimpleNamespace(literal_value=(2 if index == 5 else 0))
                for index in range(7)
            ),
        )
        fragment = object.__new__(CommandFragment)
        for name, value in {
            "schema_version": COMMAND_FRAGMENT_SCHEMA_VERSION,
            "producer_pass": "generic_lowering",
            "id": "generic.fragment",
            "source_global_dag_id": "generic.dag",
            "kind": FragmentKind.ISA_REGION,
            "claimed_action_ids": ("generic.action",),
            "core_streams": (
                SimpleNamespace(
                    logical_core=LogicalCoreRef(0, 0),
                    records=(allocation,),
                ),
            ),
            "runtime_symbols": (),
            "program_symbols": (),
            "buffer_abi": (),
            "state_abi": (),
        }.items():
            object.__setattr__(fragment, name, value)
        with self.assertRaisesRegex(
            SchemaError,
            "reserved for the exact calibration producer",
        ):
            fragment.validate()

    def test_all_14_sources_are_stable_typed_and_serde_closed(self) -> None:
        sources = tuple(_source(kind) for kind in MoeCalibrationKind)
        self.assertEqual(len(sources), 14)
        self.assertEqual(len({item.id for item in sources}), 14)
        for source in sources:
            source.validate()
            text = canonical_json(source)
            rebuilt = loads_dataclass(MoeSwizzleCalibrationProgramSource, text)
            rebuilt.validate()
            self.assertEqual(rebuilt, source)
            self.assertEqual(canonical_json(rebuilt), text)
            self.assertEqual(
                tuple((item.opcode, item.count) for item in source.record_quotient),
                expected_moe_swizzle_calibration_record_quotient(source.kind),
            )

    def test_shape_target_quotient_aux_route_and_io_tamper_fail_closed(self) -> None:
        gemm = _source(MoeCalibrationKind.GROUP_GEMM)
        with self.assertRaisesRegex(SchemaError, "production calibration coverage"):
            replace(gemm, shape=(3, 8, 32)).validate()
        with self.assertRaisesRegex(SchemaError, "target record disagrees"):
            replace(
                gemm,
                target_records=(
                    replace(gemm.target_records[0], opcode=RecordOpcode.SWIGLU),
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "record quotient"):
            replace(gemm, record_quotient=gemm.record_quotient[:-1]).validate()
        with self.assertRaisesRegex(SchemaError, "auxiliary opcode"):
            replace(gemm, auxiliary_opcodes=()).validate()
        with self.assertRaisesRegex(SchemaError, "route resource"):
            replace(gemm, source_route_resource_refs=("moe.scale.link.0.1",)).validate()
        with self.assertRaisesRegex(SchemaError, "arity"):
            replace(gemm, input_buffer_abi_ids=("input.a",)).validate()
        with self.assertRaisesRegex(SchemaError, "lowercase SHA-256"):
            replace(gemm, source_linked_program_digest="A" * 64).validate()

        hop = _source(MoeCalibrationKind.DTE_HOP)
        with self.assertRaisesRegex(SchemaError, "route resource"):
            replace(hop, source_route_resource_refs=()).validate()
        terminal = _source(MoeCalibrationKind.TERMINAL_DONE)
        with self.assertRaisesRegex(SchemaError, "no frontend target"):
            replace(
                terminal,
                target_records=(
                    MoeSwizzleCalibrationTargetRecord(
                        terminal.target_logical_core, 0, RecordOpcode.DTE_WAIT
                    ),
                ),
            ).validate()

    def test_wrapper_types_and_zero_sha_program_io_fail_closed(self) -> None:
        source = _source(MoeCalibrationKind.LOCAL_COPY)
        wrong = MoeSwizzleCalibrationStandardLinkedProgram(
            MOE_SWIZZLE_CALIBRATION_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
            "moe_swizzle_calibration_standard_linker",
            "wrong",
            source,
            object(),
            object(),
            None,
        )
        with self.assertRaisesRegex(SchemaError, "CommandFragment"):
            wrong.validate()

        core = source.target_logical_core
        target_record = source.target_records[0]
        output_label = SimpleNamespace(
            id="label.output",
            kind=ProgramSymbolKind.SRAM_LABEL,
            source_ref="output.storage",
        )
        records = tuple(
            SimpleNamespace(
                opcode=item.opcode,
                operands=(
                    (
                        SimpleNamespace(
                            name="label_symbol",
                            symbol_ref=output_label.id,
                            literal_value=None,
                        ),
                        SimpleNamespace(
                            name="lifetime", symbol_ref=None, literal_value=2,
                        ),
                    )
                    if item.opcode is RecordOpcode.SRAM_ALLOC_AT else ()
                ),
            )
            for item in source.record_quotient
            for _ in range(item.count)
        )
        fragment = object.__new__(CommandFragment)
        for name, value in {
            "id": "isolated.fragment",
            "producer_pass": "moe_swizzle_calibration_standard_lowering",
            "kind": FragmentKind.MOE_SWIZZLE_CALIBRATION,
            "source_global_dag_id": source.id,
            "core_streams": (
                SimpleNamespace(
                    logical_core=core,
                    records=records,
                ),
            ),
            "program_symbols": (output_label,),
            "buffer_abi": (
                SimpleNamespace(
                    id="input", alias_of=None,
                    ownership=BufferOwnership.BORROWED,
                    storage_id="input.storage",
                ),
                SimpleNamespace(
                    id="output", alias_of=None,
                    ownership=BufferOwnership.OWNED,
                    storage_id="output.storage",
                ),
            ),
        }.items():
            object.__setattr__(fragment, name, value)
        # LOCAL_COPY target DTE_ISSUE is first in canonical quotient order.
        self.assertEqual(
            fragment.core_streams[0].records[target_record.record_index].opcode,
            target_record.opcode,
        )

        manifest = object.__new__(LinkedProgramManifest)
        for name, value in {
            "producer_pass": "moe_swizzle_calibration_standard_linker",
            "id": "isolated.manifest",
            "source_ir1_id": source.source_ir1_id,
            "source_projection_id": source.id,
            "source_schedule_set_id": source.id,
            "source_global_dag_id": source.id,
            "fragments": (fragment,),
            "core_bindings": (
                SimpleNamespace(
                    logical_core=core,
                    runtime_core_id=source.target_runtime_core_id,
                ),
            ),
            "program_symbol_definitions": (),
            "envelope": SimpleNamespace(terminal_cores=(core,)),
        }.items():
            object.__setattr__(manifest, name, value)

        input_target = object.__new__(ProgramSramTarget)
        object.__setattr__(input_target, "buffer_abi_id", "input")
        output_target = object.__new__(ProgramSramTarget)
        object.__setattr__(output_target, "buffer_abi_id", "output")
        program_io = object.__new__(ProgramIoContract)
        for name, value in {
            "producer_pass": "build_moe_swizzle_calibration_program_io",
            "mode": ProgramIoMode.TIMING,
            "source_linked_manifest_id": manifest.id,
            "source_linked_manifest_digest": "b" * 64,
            "program_artifact_sha256": "0" * 64,
            "initializations": (SimpleNamespace(target=input_target),),
            "output_probes": (SimpleNamespace(target=output_target),),
        }.items():
            object.__setattr__(program_io, name, value)
        wrapper = MoeSwizzleCalibrationStandardLinkedProgram(
            MOE_SWIZZLE_CALIBRATION_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
            "moe_swizzle_calibration_standard_linker",
            "wrong",
            source,
            fragment,
            manifest,
            program_io,
        )
        module = sys.modules[MoeSwizzleCalibrationStandardLinkedProgram.__module__]
        with (
            patch.object(CommandFragment, "validate", lambda self, path="": None),
            patch.object(LinkedProgramManifest, "validate", lambda self, path="": None),
            patch.object(ProgramIoContract, "validate", lambda self, path="": None),
            patch.object(module, "canonical_digest", return_value="b" * 64),
        ):
            region = SimpleNamespace(
                kind=ProgramSymbolKind.SRAM_REGION,
                source_ref="comm",
            )
            object.__setattr__(
                manifest,
                "program_symbol_definitions",
                (SimpleNamespace(symbol=region, name="synthetic-region"),),
            )
            with self.assertRaisesRegex(SchemaError, "hardware region ref"):
                wrapper.validate()
            object.__setattr__(manifest, "program_symbol_definitions", ())
            allocation = next(
                record for record in records
                if record.opcode is RecordOpcode.SRAM_ALLOC_AT
            )
            lifetime = next(
                operand for operand in allocation.operands
                if operand.name == "lifetime"
            )
            lifetime.literal_value = 0
            with self.assertRaisesRegex(SchemaError, "PERSISTENT/live"):
                wrapper.validate()
            lifetime.literal_value = 2
            with self.assertRaisesRegex(SchemaError, "actual-SHA"):
                wrapper.validate()


if __name__ == "__main__":
    unittest.main()
