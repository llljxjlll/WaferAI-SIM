from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import unittest

from test_train_forward_global_action import _global_action
from test_train_forward_ir0 import _tiny_train_spec

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
    build_train_forward_oracle,
    link_train,
    lower_train,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoTargetKind
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.train_forward_evidence import (
    TRAIN_FORWARD_MARKER_SCHEMA_VERSION,
    TRAIN_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION,
    TrainForwardArtifactEvidence,
    TrainForwardCeMarkerEvidence,
    TrainForwardControlEvidence,
    TrainForwardCoreCount,
    TrainForwardMemoryEvidence,
    TrainForwardNamedCount,
    TrainForwardOpcodeCount,
    TrainForwardProgramIoEvidence,
    TrainForwardRepeatEvidence,
    TrainForwardRuntimeReport,
    TrainForwardWorkEvidence,
)


_DIGEST = "1" * 64


def _case():
    spec = _tiny_train_spec()
    oracle = build_train_forward_oracle(spec)
    _, global_action = _global_action()
    linked = link_train(lower_train(global_action))
    state_seeds, state_expected = build_deterministic_timing_state_overrides(
        linked
    )
    if state_expected:
        raise AssertionError("forward-only Train must not expect state stores")
    program_io = build_timing_program_io(
        linked,
        _DIGEST,
        state_seed_overrides=state_seeds,
    )
    manifest = linked.manifest
    leaves = tuple(
        item.fragment if isinstance(item, RegionManifest) else item
        for item in manifest.fragments
    )
    records = tuple(
        record
        for leaf in leaves
        for stream in leaf.core_streams
        for record in stream.records
    )
    opcodes = Counter(record.opcode for record in records)
    runtime_relocations = sum(
        len(stream.runtime_relocations)
        for leaf in leaves
        for stream in leaf.core_streams
    )
    address_relocations = sum(
        len(stream.address_relocations)
        for leaf in leaves
        for stream in leaf.core_streams
    )
    artifact = TrainForwardArtifactEvidence(
        len({record.source_global_action_id for record in records}),
        len(manifest.fragments),
        len(records),
        runtime_relocations,
        address_relocations,
        runtime_relocations + address_relocations,
        len(manifest.address_operand_bindings),
        len(manifest.state_operand_bindings),
        tuple(
            TrainForwardOpcodeCount(opcode, opcodes[opcode])
            for opcode in sorted(opcodes, key=int)
        ),
    )
    observed_send_bytes = sum(
        next(
            operand.literal_value
            for operand in record.operands
            if operand.name == "length_bytes"
        )
        for record in records
        if record.opcode is RecordOpcode.DTE_SEND
    )
    work = TrainForwardWorkEvidence(
        oracle.parameters.unique_tensor_count,
        oracle.parameters.unique_bytes,
        oracle.parameters.tp_placed_bytes,
        oracle.parameters.dp_replicated_bytes,
        oracle.gemm_flops_per_microbatch,
        oracle.attention_flops_per_microbatch,
        oracle.logical_forward_flops_per_microbatch,
        oracle.rank_forward_flops_per_microbatch,
        oracle.cluster_forward_flops_per_step,
        oracle.collectives.node_count,
        oracle.collectives.node_count
        * oracle.collectives.logical_tensor_bytes_per_node
        * oracle.dp_degree,
        observed_send_bytes,
    )
    hbm_initializations = sum(
        item.target.kind is ProgramIoTargetKind.HBM
        for item in program_io.initializations
    )
    hbm_probes = sum(
        item.target.kind is ProgramIoTargetKind.HBM
        for item in program_io.output_probes
    )
    sidecar = TrainForwardProgramIoEvidence(
        program_io.mode,
        hbm_initializations,
        len(program_io.initializations) - hbm_initializations,
        hbm_probes,
        len(program_io.output_probes) - hbm_probes,
        oracle.dp_degree * oracle.tp_degree,
        oracle.dp_degree * oracle.tp_degree,
    )
    runtime_cores = tuple(
        binding.runtime_core_id for binding in manifest.core_bindings
    )
    per_core_parameter_bytes = oracle.parameters.dp_replicated_bytes // 4
    memory = tuple(
        TrainForwardMemoryEvidence(
            core,
            15,
            15,
            per_core_parameter_bytes,
            0,
            0,
            per_core_parameter_bytes,
            0,
            0,
        )
        for core in runtime_cores
    )
    ce_markers = tuple(
        TrainForwardCeMarkerEvidence(
            core,
            1,
            oracle.ce.rank_rows,
            oracle.ce.rank_label_bytes,
            oracle.ce.rank_loss_bytes,
        )
        for core in runtime_cores
    )
    control = TrainForwardControlEvidence(
        tuple(TrainForwardCoreCount(core, 2) for core in runtime_cores),
        tuple(TrainForwardCoreCount(core, 1) for core in runtime_cores),
        tuple(
            TrainForwardNamedCount(name, 0)
            for name in ("collective", "global", "p2p", "timing")
        ),
        True,
    )
    repeat_fields = {
        "makespan_cycles": 12345,
        "marker_digest": _DIGEST,
        "memory_digest": canonical_digest(memory),
        "ce_digest": canonical_digest(ce_markers),
        "control_digest": canonical_digest(control),
    }
    report = TrainForwardRuntimeReport.create(
        spec_digest=canonical_digest(spec),
        oracle_id=oracle.id,
        oracle_digest=canonical_digest(oracle),
        train_linked_id=linked.id,
        train_linked_digest=canonical_digest(linked),
        linked_manifest_id=manifest.id,
        linked_manifest_digest=canonical_digest(manifest),
        program_io_id=program_io.id,
        program_io_digest=canonical_digest(program_io),
        program_artifact_sha256=_DIGEST,
        artifact_size_bytes=1234,
        finalizer_sha256=_DIGEST,
        resolver_sha256=_DIGEST,
        npusim_sha256=_DIGEST,
        hardware_digest=_DIGEST,
        simulation_digest=_DIGEST,
        mapping_digest=_DIGEST,
        artifact=artifact,
        work=work,
        program_io=sidecar,
        memory=memory,
        ce_markers=ce_markers,
        control=control,
        marker_schema_version=TRAIN_FORWARD_MARKER_SCHEMA_VERSION,
        repeat_count=2,
        makespan_cycles=12345,
        repeats=tuple(
            TrainForwardRepeatEvidence(index, **repeat_fields)
            for index in range(2)
        ),
        timing_execution=True,
        train_structure_exact=True,
        analytic_work_exact=True,
        collective_accounting_exact=True,
        program_io_boundary_exact=True,
        compute_functional=False,
        model_functional=False,
    )
    return report, spec, oracle, linked, program_io


class TrainForwardEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report, cls.spec, cls.oracle, cls.linked, cls.program_io = _case()

    def test_strict_roundtrip_and_exact_source_closure(self) -> None:
        report = self.report
        self.assertEqual(
            TRAIN_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION,
            "wafer_frontend.train_forward_runtime_report/v1alpha1",
        )
        report.validate_against(
            self.spec,
            self.oracle,
            self.linked,
            self.program_io,
        )
        self.assertEqual(
            loads_dataclass(
                TrainForwardRuntimeReport,
                canonical_json(report),
                path="report",
            ),
            report,
        )
        self.assertEqual(
            (
                report.artifact.action_count,
                report.artifact.fragment_count,
                report.artifact.record_count,
                report.artifact.runtime_relocation_count,
                report.artifact.address_relocation_count,
                report.artifact.address_operand_binding_count,
                report.artifact.state_operand_binding_count,
                report.work.collective_unique_bytes,
                report.work.collective_observed_send_bytes,
            ),
            (308, 172, 996, 288, 1652, 1592, 60, 4096, 4096),
        )

    def test_strict_serde_and_stable_id_fail_closed(self) -> None:
        raw = json.loads(canonical_json(self.report))
        raw["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(
                TrainForwardRuntimeReport,
                json.dumps(raw),
                path="report",
            )
        del raw["unexpected"]
        del raw["control"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(
                TrainForwardRuntimeReport,
                json.dumps(raw),
                path="report",
            )
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(self.report, id="forged").validate()

    def test_timing_accounting_and_functional_claim_tamper_matrix(self) -> None:
        semantic = self.report._semantic_key()
        cases = (
            ("artifact", {"artifact": replace(self.report.artifact, record_count=1)}),
            ("parameters", {"work": replace(self.report.work, parameter_unique_bytes=1)}),
            ("flops", {"work": replace(self.report.work, cluster_forward_flops_per_step=1)}),
            ("collective", {"work": replace(self.report.work, collective_observed_send_bytes=1)}),
            ("memory", {"memory": (replace(self.report.memory[0], lsu_hbm_read_bytes=1), *self.report.memory[1:])}),
            ("ce", {"ce_markers": (replace(self.report.ce_markers[0], invocation_count=2), *self.report.ce_markers[1:])}),
            ("control", {"control": replace(self.report.control, all_done_boundary_reached=False)}),
            ("repeat", {"makespan_cycles": self.report.makespan_cycles + 1}),
            ("timing", {"timing_execution": False}),
            ("structure", {"train_structure_exact": False}),
            ("analytic", {"analytic_work_exact": False}),
            ("traffic", {"collective_accounting_exact": False}),
            ("program_io", {"program_io_boundary_exact": False}),
            ("compute", {"compute_functional": True}),
            ("model", {"model_functional": True}),
        )
        for name, changes in cases:
            with self.subTest(name=name), self.assertRaises(SchemaError):
                forged = TrainForwardRuntimeReport.create(
                    **(semantic | changes)
                )
                forged.validate_against(
                    self.spec,
                    self.oracle,
                    self.linked,
                    self.program_io,
                )

    def test_restable_source_and_program_io_tamper_fail_closed(self) -> None:
        semantic = self.report._semantic_key()
        forged = TrainForwardRuntimeReport.create(
            **(semantic | {"oracle_digest": "2" * 64})
        )
        with self.assertRaisesRegex(SchemaError, "source ids/digests"):
            forged.validate_against(
                self.spec,
                self.oracle,
                self.linked,
                self.program_io,
            )
        forged = TrainForwardRuntimeReport.create(
            **(
                semantic
                | {
                    "program_io": replace(
                        self.report.program_io,
                        sram_probe_count=(
                            self.report.program_io.sram_probe_count + 1
                        ),
                    )
                }
            )
        )
        with self.assertRaisesRegex(SchemaError, "ProgramIo evidence"):
            forged.validate_against(
                self.spec,
                self.oracle,
                self.linked,
                self.program_io,
            )


if __name__ == "__main__":
    unittest.main()
