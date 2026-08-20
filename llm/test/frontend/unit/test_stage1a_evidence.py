from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema import (
    STAGE1A_BASELINE_EPOCH,
    Stage1aArtifactEvidence,
    Stage1aCase,
    Stage1aControlEvidence,
    Stage1aCoreCount,
    Stage1aDecodeStartBoundary,
    Stage1aMemoryEvidence,
    Stage1aNamedCount,
    Stage1aOpcodeCount,
    Stage1aOracle,
    Stage1aPdWitness,
    Stage1aProbeEvidence,
    Stage1aRepeatEvidence,
    Stage1aRuntimeReport,
    Stage1aSidecarEvidence,
    Stage1aStatePair,
    Stage1aStatePayload,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.capability import CapabilityStatus
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramIoMode,
    ProgramIoTargetKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
    to_primitive,
)


_P1_SHA = "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"
_K1_SHAS = (
    "02d449a31fbb267c8f352e9968a79e3e5fc95c1bbeaa502fd6454ebde5a4bedc",
    "9f72ea0cf49536e3c66c787f705186df9a4378083753ae9536d65b3ad7fcddc4",
    "deb0e38ced1e41de6f92e70e80c418d2d356afaaa99e26f5939dbc7d3ef4772a",
    "bb391415c05e39d77ca17381d3be3f7d0cd5e5332e5a579311adaa0aa62106e9",
)
_PD_K_SHA = "60bf07c488aad18fda339df07e4fbc47b4f00be71711936f18d04d352ad01890"
_PD_V_SHA = "fc8b64001c5fdd0f2f40fb67dae4a865a2c5bd17836676d6d5b58b7917e33717"
_DRAINS = ("collective", "global", "p2p", "timing")


def _opcode_counts(case: Stage1aCase) -> tuple[Stage1aOpcodeCount, ...]:
    values = {
        Stage1aCase.P1: (
            Stage1aOpcodeCount(RecordOpcode.MATMUL, 1),
            Stage1aOpcodeCount(RecordOpcode.SRAM_BIND, 1),
            Stage1aOpcodeCount(RecordOpcode.LSU_LOAD, 1),
            Stage1aOpcodeCount(RecordOpcode.SRAM_FREE, 3),
            Stage1aOpcodeCount(RecordOpcode.SRAM_ALLOC_AT, 3),
        ),
        Stage1aCase.K1: (
            Stage1aOpcodeCount(RecordOpcode.MATMUL, 9),
            Stage1aOpcodeCount(RecordOpcode.ROPE_QK_EXACT, 2),
            Stage1aOpcodeCount(RecordOpcode.ATTENTION_EXACT, 2),
            Stage1aOpcodeCount(RecordOpcode.EMBEDDING_LOOKUP, 1),
            Stage1aOpcodeCount(RecordOpcode.SWIGLU, 2),
            Stage1aOpcodeCount(RecordOpcode.RESIDUAL, 4),
            Stage1aOpcodeCount(RecordOpcode.RMSNORM, 5),
            Stage1aOpcodeCount(RecordOpcode.SRAM_BIND, 25),
            Stage1aOpcodeCount(RecordOpcode.LSU_LOAD, 4),
            Stage1aOpcodeCount(RecordOpcode.LSU_STORE, 4),
            Stage1aOpcodeCount(RecordOpcode.SRAM_FREE, 49),
            Stage1aOpcodeCount(RecordOpcode.SRAM_ALLOC_AT, 49),
        ),
        Stage1aCase.PD1: (
            Stage1aOpcodeCount(RecordOpcode.ATTENTION_EXACT, 2),
            Stage1aOpcodeCount(RecordOpcode.DTE_SEND, 2),
            Stage1aOpcodeCount(RecordOpcode.DTE_RECV, 2),
            Stage1aOpcodeCount(RecordOpcode.SRAM_BIND, 2),
            Stage1aOpcodeCount(RecordOpcode.LSU_LOAD, 2),
            Stage1aOpcodeCount(RecordOpcode.LSU_STORE, 2),
            Stage1aOpcodeCount(RecordOpcode.SRAM_FREE, 8),
            Stage1aOpcodeCount(RecordOpcode.SRAM_ALLOC_AT, 8),
            Stage1aOpcodeCount(RecordOpcode.DTE_WAIT, 2),
        ),
    }[case]
    return tuple(sorted(values, key=lambda item: int(item.opcode)))

def _oracle(case: Stage1aCase) -> Stage1aOracle:
    if case is Stage1aCase.P1:
        payloads = (Stage1aStatePayload("p1.weight", 128, _P1_SHA),)
        pairs = ()
        numeric = (1, 1, 0, 128, 0, 0, 8, 0, 0, 128)
        sidecar = (1, 3, 0, 1)
        ack = (Stage1aCoreCount(0, 2),)
        done = (Stage1aCoreCount(0, 1),)
    elif case is Stage1aCase.K1:
        payloads = tuple(
            Stage1aStatePayload(state_ref, 32, digest)
            for state_ref, digest in zip(
                ("k1.l0.k", "k1.l0.v", "k1.l1.k", "k1.l1.v"),
                _K1_SHAS,
            )
        )
        pairs = ()
        numeric = (4, 4, 4, 128, 128, 0, 8, 8, 0, 0)
        sidecar = (0, 49, 4, 5)
        ack = (Stage1aCoreCount(0, 2),)
        done = (Stage1aCoreCount(0, 1),)
    else:
        payloads = (
            Stage1aStatePayload("pd1.destination.k", 32, _PD_K_SHA),
            Stage1aStatePayload("pd1.destination.v", 32, _PD_V_SHA),
            Stage1aStatePayload("pd1.source.k", 32, _PD_K_SHA),
            Stage1aStatePayload("pd1.source.v", 32, _PD_V_SHA),
        )
        pairs = (
            Stage1aStatePair("pd1.source.k", "pd1.destination.k", _PD_K_SHA),
            Stage1aStatePair("pd1.source.v", "pd1.destination.v", _PD_V_SHA),
        )
        numeric = (4, 2, 2, 64, 64, 64, 4, 4, 4, 0)
        sidecar = (2, 8, 2, 2)
        ack = (Stage1aCoreCount(0, 2), Stage1aCoreCount(16, 2))
        done = (Stage1aCoreCount(0, 1), Stage1aCoreCount(16, 1))
    if case is Stage1aCase.P1:
        probe_sources = (("probe.sram.0", payloads[0]),)
    elif case is Stage1aCase.K1:
        probe_sources = (
            *(
                (f"probe.hbm.{index}", payload)
                for index, payload in enumerate(payloads)
            ),
            *(
                (f"probe.sram.{index}", payload)
                for index, payload in enumerate(payloads)
            ),
            ("probe.sram.4", payloads[0]),
        )
    else:
        destination = tuple(
            payload
            for payload in payloads
            if ".destination." in payload.state_ref
        )
        probe_sources = (
            *(
                (f"probe.hbm.{index}", payload)
                for index, payload in enumerate(destination)
            ),
            *(
                (f"probe.sram.{index}", payload)
                for index, payload in enumerate(destination)
            ),
        )
    expected_probe_payloads = tuple(
        sorted(
            (
                Stage1aStatePayload(
                    probe_id,
                    payload.size_bytes,
                    payload.sha256,
                )
                for probe_id, payload in probe_sources
            ),
            key=lambda item: item.state_ref,
        )
    )
    return Stage1aOracle.create(
        case=case,
        capability_status=CapabilityStatus.E2E_TIMING,
        state_count=numeric[0],
        state_payloads=payloads,
        expected_probe_payloads=expected_probe_payloads,
        state_pairs=pairs,
        expected_dma_loads=numeric[1],
        expected_dma_stores=numeric[2],
        expected_opcode_counts=_opcode_counts(case),
        expected_hbm_read_bytes=numeric[3],
        expected_hbm_write_bytes=numeric[4],
        expected_d2d_bytes=numeric[5],
        hbm_read_capacity_floor_cycles=numeric[6],
        hbm_write_capacity_floor_cycles=numeric[7],
        d2d_capacity_floor_cycles=numeric[8],
        gemm_flops=numeric[9],
        expected_sidecar_mode=ProgramIoMode.TIMING,
        expected_hbm_initialization_count=sidecar[0],
        expected_sram_initialization_count=sidecar[1],
        expected_hbm_probe_count=sidecar[2],
        expected_sram_probe_count=sidecar[3],
        expected_ack_counts=ack,
        expected_done_counts=done,
        expected_drain_names=_DRAINS,
        timing_execution=True,
        state_transport_exact=True,
        compute_functional=False,
        model_functional=False,
        synthetic_pd=case is Stage1aCase.PD1,
        decode_start_boundary=(
            Stage1aDecodeStartBoundary.HOST_AFTER_ALL_DONE
            if case is Stage1aCase.PD1
            else None
        ),
        notes=(
            "sidecar counts cover all production entries",
            "state transport is byte-exact",
            "timing-only model execution",
        ),
    )


def _probes(oracle: Stage1aOracle) -> tuple[Stage1aProbeEvidence, ...]:
    return tuple(
        Stage1aProbeEvidence(
            probe_id=payload.state_ref,
            target_kind=(
                ProgramIoTargetKind.HBM
                if payload.state_ref.startswith("probe.hbm.")
                else ProgramIoTargetKind.SRAM
            ),
            length_bytes=payload.size_bytes,
            expected_sha256=payload.sha256,
            actual_sha256=payload.sha256,
            all_bytes_valid=True,
            exact_match=True,
            passed=True,
        )
        for payload in oracle.expected_probe_payloads
    )


def _report(oracle: Stage1aOracle) -> Stage1aRuntimeReport:
    if oracle.case is Stage1aCase.P1:
        memory = (
            Stage1aMemoryEvidence(0, 1, 1, 128, 0, 0, 128, 0, 0),
        )
    elif oracle.case is Stage1aCase.K1:
        memory = (
            Stage1aMemoryEvidence(0, 8, 8, 128, 128, 128, 128, 0, 0),
        )
    else:
        memory = (
            Stage1aMemoryEvidence(0, 2, 2, 64, 0, 0, 64, 0, 0),
            Stage1aMemoryEvidence(16, 2, 2, 0, 64, 64, 0, 0, 0),
        )
    probes = _probes(oracle)
    control = Stage1aControlEvidence(
        ack_counts=oracle.expected_ack_counts,
        done_counts=oracle.expected_done_counts,
        drain_residuals=tuple(Stage1aNamedCount(name, 0) for name in _DRAINS),
        all_done_boundary_reached=True,
    )
    makespan = {Stage1aCase.P1: 101, Stage1aCase.K1: 202, Stage1aCase.PD1: 303}[
        oracle.case
    ]
    repeats = tuple(
        Stage1aRepeatEvidence(
            run_index=index,
            makespan_cycles=makespan,
            marker_digest="a" * 64,
            memory_digest=canonical_digest(memory),
            probe_digest=canonical_digest(probes),
            control_digest=canonical_digest(control),
        )
        for index in range(2)
    )
    pd_witness = (
        Stage1aPdWitness(
            state_pairs=oracle.state_pairs,
            completion_action_ids=("action.store.k", "action.store.v"),
            done_runtime_core_ids=tuple(
                item.runtime_core_id for item in oracle.expected_done_counts
            ),
            decode_start_boundary=Stage1aDecodeStartBoundary.HOST_AFTER_ALL_DONE,
            all_completion_actions_before_boundary=True,
        )
        if oracle.case is Stage1aCase.PD1
        else None
    )
    return Stage1aRuntimeReport.create(
        baseline_epoch=STAGE1A_BASELINE_EPOCH,
        case=oracle.case,
        capability_status=CapabilityStatus.E2E_TIMING,
        oracle_id=oracle.id,
        oracle_digest=canonical_digest(oracle),
        hardware_digest="1" * 64,
        simulation_digest="2" * 64,
        mapping_digest="3" * 64,
        artifact=Stage1aArtifactEvidence(
            linked_manifest_id=f"manifest.{oracle.case.value.lower()}",
            linked_manifest_digest="4" * 64,
            program_artifact_sha256="5" * 64,
            artifact_size_bytes=4096,
            record_count=sum(item.count for item in oracle.expected_opcode_counts),
            relocation_count=1,
            opcode_counts=oracle.expected_opcode_counts,
        ),
        sidecar=Stage1aSidecarEvidence(
            contract_id=f"sidecar.{oracle.case.value.lower()}",
            contract_digest="6" * 64,
            mode=oracle.expected_sidecar_mode,
            hbm_initialization_count=oracle.expected_hbm_initialization_count,
            sram_initialization_count=oracle.expected_sram_initialization_count,
            hbm_probe_count=oracle.expected_hbm_probe_count,
            sram_probe_count=oracle.expected_sram_probe_count,
        ),
        memory=memory,
        probes=probes,
        control=control,
        observed_hbm_read_bytes=oracle.expected_hbm_read_bytes,
        observed_hbm_write_bytes=oracle.expected_hbm_write_bytes,
        observed_d2d_bytes=oracle.expected_d2d_bytes,
        repeat_count=2,
        makespan_cycles=makespan,
        repeats=repeats,
        timing_execution=True,
        state_transport_exact=True,
        compute_functional=False,
        model_functional=False,
        synthetic_pd=oracle.synthetic_pd,
        pd_witness=pd_witness,
    )


def _replace_oracle(oracle: Stage1aOracle, **changes: object) -> Stage1aOracle:
    fields = oracle._semantic_key()
    fields.update(changes)
    return Stage1aOracle.create(**fields)


def _replace_report(
    report: Stage1aRuntimeReport, **changes: object
) -> Stage1aRuntimeReport:
    fields = report._semantic_key()
    fields.update(changes)
    return Stage1aRuntimeReport.create(**fields)


class Stage1aEvidenceTest(unittest.TestCase):
    def test_all_cases_are_stable_strict_roundtrips(self) -> None:
        expected_sidecar_counts = {
            Stage1aCase.P1: (1, 3, 0, 1),
            Stage1aCase.K1: (0, 49, 4, 5),
            Stage1aCase.PD1: (2, 8, 2, 2),
        }
        for case in Stage1aCase:
            with self.subTest(case=case.value):
                oracle = _oracle(case)
                report = _report(oracle)
                report.validate_against(oracle)
                self.assertEqual(
                    loads_dataclass(Stage1aOracle, canonical_json(oracle)), oracle
                )
                self.assertEqual(
                    loads_dataclass(Stage1aRuntimeReport, canonical_json(report)),
                    report,
                )
                self.assertEqual(_oracle(case), oracle)
                self.assertEqual(_report(oracle), report)
                self.assertEqual(
                    (
                        report.sidecar.hbm_initialization_count,
                        report.sidecar.sram_initialization_count,
                        report.sidecar.hbm_probe_count,
                        report.sidecar.sram_probe_count,
                    ),
                    expected_sidecar_counts[case],
                )

    def test_strict_serde_rejects_missing_unknown_enum_and_old_version(self) -> None:
        oracle = _oracle(Stage1aCase.P1)
        raw = to_primitive(oracle)
        assert isinstance(raw, dict)
        cases = []
        missing = dict(raw)
        missing.pop("state_count")
        cases.append(("missing", missing))
        unknown = dict(raw)
        unknown["future"] = 1
        cases.append(("unknown", unknown))
        bad_enum = dict(raw)
        bad_enum["case"] = "P2"
        cases.append(("enum", bad_enum))
        old = dict(raw)
        old["schema_version"] = "wafer_frontend.stage1a_oracle/v1alpha0"
        cases.append(("old", old))
        for name, value in cases:
            with self.subTest(case=name), self.assertRaises(SchemaError):
                loads_dataclass(Stage1aOracle, canonical_json(value))

    def test_oracle_numeric_proof_digest_and_canonical_gates(self) -> None:
        oracle = _oracle(Stage1aCase.K1)
        cases = (
            ("numeric", {"expected_hbm_read_bytes": 127}),
            ("capability", {"capability_status": CapabilityStatus.E2E_FUNCTIONAL}),
            ("proof", {"compute_functional": True}),
            ("typed", {"expected_opcode_counts": (object(),)}),
            ("canonical", {"state_payloads": tuple(reversed(oracle.state_payloads))}),
            (
                "probe_canonical",
                {
                    "expected_probe_payloads": tuple(
                        reversed(oracle.expected_probe_payloads)
                    )
                },
            ),
            (
                "probe_count",
                {
                    "expected_probe_payloads": (
                        oracle.expected_probe_payloads[:-1]
                    )
                },
            ),
            (
                "digest",
                {
                    "state_payloads": (
                        replace(oracle.state_payloads[0], sha256="A" * 64),
                        *oracle.state_payloads[1:],
                    )
                },
            ),
            ("sidecar", {"expected_hbm_probe_count": 3}),
        )
        for name, changes in cases:
            with self.subTest(case=name), self.assertRaises(SchemaError):
                _replace_oracle(oracle, **changes)
        with self.assertRaises(SchemaError):
            replace(oracle, id="stage1a_oracle_stale").validate()

    def test_missing_any_lifecycle_opcode_fails_closed(self) -> None:
        lifecycle = (
            RecordOpcode.SRAM_ALLOC_AT,
            RecordOpcode.SRAM_BIND,
            RecordOpcode.SRAM_FREE,
        )
        for case in Stage1aCase:
            oracle = _oracle(case)
            for opcode in lifecycle:
                without_opcode = tuple(
                    item
                    for item in oracle.expected_opcode_counts
                    if item.opcode is not opcode
                )
                with self.subTest(case=case.value, opcode=opcode.name):
                    with self.assertRaisesRegex(
                        SchemaError,
                        "full artifact",
                    ):
                        _replace_oracle(
                            oracle,
                            expected_opcode_counts=without_opcode,
                        )

    def test_report_tamper_repeat_probe_control_and_order_fail_closed(self) -> None:
        oracle = _oracle(Stage1aCase.K1)
        report = _report(oracle)
        with self.subTest("capability"), self.assertRaises(SchemaError):
            _replace_report(
                report, capability_status=CapabilityStatus.E2E_FUNCTIONAL
            )
        with self.subTest("memory_sum"), self.assertRaises(SchemaError):
            _replace_report(report, observed_hbm_read_bytes=127)
        with self.subTest("memory_order"), self.assertRaises(SchemaError):
            pd_report = _report(_oracle(Stage1aCase.PD1))
            _replace_report(pd_report, memory=tuple(reversed(pd_report.memory)))
        with self.subTest("repeat_marker"), self.assertRaises(SchemaError):
            _replace_report(
                report,
                repeats=(
                    report.repeats[0],
                    replace(report.repeats[1], marker_digest="b" * 64),
                ),
            )
        with self.subTest("probe"), self.assertRaises(SchemaError):
            probes = (
                replace(report.probes[0], exact_match=False),
                *report.probes[1:],
            )
            _replace_report(report, probes=probes)
        with self.subTest("drain"), self.assertRaises(SchemaError):
            drains = (
                replace(report.control.drain_residuals[0], count=1),
                *report.control.drain_residuals[1:],
            )
            _replace_report(
                report,
                control=replace(report.control, drain_residuals=drains),
            )
        with self.assertRaises(SchemaError):
            replace(report, id="stage1a_runtime_report_stale").validate()

    def test_report_exactly_closes_supplied_oracle(self) -> None:
        oracle = _oracle(Stage1aCase.P1)
        report = _report(oracle)
        report.validate_against(oracle)
        bad_digest = _replace_report(report, oracle_digest="f" * 64)
        with self.assertRaises(SchemaError):
            bad_digest.validate_against(oracle)
        empty_probe_digest = canonical_digest(())
        missing_probe = _replace_report(
            report,
            probes=(),
            repeats=tuple(
                replace(repeat, probe_digest=empty_probe_digest)
                for repeat in report.repeats
            ),
        )
        with self.assertRaises(SchemaError):
            missing_probe.validate_against(oracle)

        kv_oracle = _oracle(Stage1aCase.K1)
        kv_report = _report(kv_oracle)
        sram_index = next(
            index
            for index, probe in enumerate(kv_report.probes)
            if probe.probe_id == "probe.sram.4"
        )
        for label, probe in (
            (
                "unwitnessed_hash",
                replace(
                    kv_report.probes[sram_index],
                    expected_sha256="7" * 64,
                    actual_sha256="7" * 64,
                ),
            ),
            (
                "unwitnessed_id",
                replace(
                    kv_report.probes[sram_index],
                    probe_id="probe.sram.9",
                ),
            ),
        ):
            with self.subTest(label=label), self.assertRaises(SchemaError):
                probes = (
                    *kv_report.probes[:sram_index],
                    probe,
                    *kv_report.probes[sram_index + 1 :],
                )
                probe_digest = canonical_digest(probes)
                forged = _replace_report(
                    kv_report,
                    probes=probes,
                    repeats=tuple(
                        replace(repeat, probe_digest=probe_digest)
                        for repeat in kv_report.repeats
                    ),
                )
                forged.validate_against(kv_oracle)

    def test_pd_pair_and_done_boundary_witness_is_exact(self) -> None:
        oracle = _oracle(Stage1aCase.PD1)
        report = _report(oracle)
        report.validate_against(oracle)
        assert report.pd_witness is not None
        cases = (
            replace(
                report.pd_witness,
                state_pairs=tuple(reversed(report.pd_witness.state_pairs)),
            ),
            replace(
                report.pd_witness,
                completion_action_ids=("action.store.k",),
            ),
            replace(
                report.pd_witness,
                done_runtime_core_ids=(0,),
            ),
            replace(
                report.pd_witness,
                all_completion_actions_before_boundary=False,
            ),
        )
        for witness in cases:
            with self.subTest(witness=witness), self.assertRaises(SchemaError):
                forged = _replace_report(report, pd_witness=witness)
                forged.validate_against(oracle)


if __name__ == "__main__":
    unittest.main()
