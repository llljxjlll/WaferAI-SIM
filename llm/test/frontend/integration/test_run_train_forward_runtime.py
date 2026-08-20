from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.passes import (
    build_deterministic_timing_state_overrides,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramHbmTarget,
    ProgramSramTarget,
)

from run_train_forward_runtime import (
    _ACTIVE_CORES,
    _artifact_evidence,
    _build_actual_sha_program_io,
    _build_report,
    _observe_runtime,
    _validate_static_case,
)
from train_forward_cases import build_train_forward_case


_DIGEST = hashlib.sha256(b"train-forward-runtime-parser").hexdigest()


@lru_cache(maxsize=1)
def _case_and_contract():
    case = build_train_forward_case()
    _validate_static_case(case)
    contract = _build_actual_sha_program_io(case, _DIGEST)
    return case, contract


def _synthetic_output() -> str:
    _case, contract = _case_and_contract()
    lines = [
        (
            f"[PROGRAM_IO] phase={phase} mode=timing checksum={_DIGEST} "
            "initializations=328 probes=4 pass=1"
        )
        for phase in ("resolved", "applied", "verify")
    ]
    blobs = {blob.id: blob for blob in contract.blobs}
    for probe in contract.output_probes:
        if type(probe.target) is not ProgramSramTarget:
            raise AssertionError("Train loss probe must target SRAM")
        sha256 = blobs[probe.blob_ref].sha256
        lines.append(
            f"[PROGRAM_IO_PROBE] id={probe.id} bytes={probe.length_bytes} "
            f"expected_checksum={sha256} checksum={sha256} "
            f"valid=1 exact=1 pass=1 core={probe.target.runtime_core_id}"
        )
    lines.extend(
        (
            "[PROGRAM_MEMORY] "
            f"core={core} lsu_issued=15 lsu_completed=15 "
            "lsu_hbm_read_bytes=7328 lsu_hbm_write_bytes=0 "
            "lsu_sram_read_bytes=0 lsu_sram_write_bytes=7328 "
            "lsu_residual=0 dte_residual=0"
        )
        for core in _ACTIVE_CORES
    )
    lines.extend(
        (
            "[TRAIN_CE] "
            f"core={core} invocations=1 rank_rows=4 "
            "label_read_bytes=16 loss_write_bytes=16"
        )
        for core in reversed(_ACTIVE_CORES)
    )
    lines.extend(
        (
            "[SIM_RESULT] makespan_cycles=12345",
            "End DONE reception",
            "[HOSTLANE] ack_total=8 done_total=4 mismatch=0",
            "[HOSTSIG] ack="
            + ",".join(
                f"{core}:{lane}:1"
                for core in _ACTIVE_CORES
                for lane in (0, 1)
            )
            + " done="
            + ",".join(f"{core}:1" for core in _ACTIVE_CORES),
            "[P5 P2P TIMING DRAIN] residual=0",
        )
    )
    lines.extend(
        f"[P5 P2P DRAIN] core={core} residual=0"
        for core in _ACTIVE_CORES
    )
    lines.extend(
        (
            "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
            "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0",
            "[DRAIN] router_residual=0 d2d_link_residual=0",
            "[D2D_TYPE] request_in=32 request_out=32 ack_in=64 "
            "ack_out=64 data_in=256 data_out=256",
        )
    )
    for index, (source, destination) in enumerate(
        ((0, 1), (1, 0), (2, 3), (3, 2))
    ):
        lines.append(
            f"[D2D_LINK] idx={index} die{source}->die{destination} "
            "dir=EAST req_in=8 req_out=8 ack_in=16 ack_out=16 "
            "data_in=64 data_out=64"
        )
    return "\n".join(lines)


class TrainForwardRuntimeParserTest(unittest.TestCase):
    def test_unified_static_pre_runtime_and_program_io(self) -> None:
        case, contract = _case_and_contract()
        artifact = _artifact_evidence(case)
        self.assertEqual(
            (
                artifact.action_count,
                artifact.fragment_count,
                artifact.record_count,
                artifact.runtime_relocation_count,
                artifact.address_relocation_count,
                artifact.relocation_count,
                artifact.address_operand_binding_count,
                artifact.state_operand_binding_count,
            ),
            (308, 172, 996, 288, 1652, 1940, 1592, 60),
        )
        self.assertEqual(len(case.linked.manifest.core_streams), 4)
        self.assertEqual(
            (
                len(case.linked.source.replicas),
                sum(
                    len(replica.fragments)
                    for replica in case.linked.source.replicas
                ),
                len(case.linked.manifest.fragments),
            ),
            (2, 172, 172),
            "both replicas must be merged into one linked manifest",
        )
        state_seeds, state_expected = (
            build_deterministic_timing_state_overrides(case.linked)
        )
        self.assertEqual((len(state_seeds), len(state_expected)), (30, 0))
        self.assertEqual(
            (
                len(contract.initializations),
                len(contract.output_probes),
                sum(
                    type(entry.target) is ProgramHbmTarget
                    for entry in contract.initializations
                ),
                sum(
                    type(entry.target) is ProgramSramTarget
                    for entry in contract.initializations
                ),
            ),
            (328, 4, 60, 268),
        )

    def test_marker_parser_closes_exact_timing_observation(self) -> None:
        _case, contract = _case_and_contract()
        output = _synthetic_output()
        first = _observe_runtime(output, _DIGEST, contract)
        second = _observe_runtime(output, _DIGEST, contract)
        self.assertEqual(first, second)
        self.assertEqual(first.makespan_cycles, 12345)
        self.assertEqual(
            tuple(marker.runtime_core_id for marker in first.ce_markers),
            _ACTIVE_CORES,
        )
        self.assertEqual(
            first.d2d_links,
            (
                (0, 1, 8, 16, 64),
                (1, 0, 8, 16, 64),
                (2, 3, 8, 16, 64),
                (3, 2, 8, 16, 64),
            ),
        )
        source_path = Path(__file__).resolve()
        report = _build_report(
            _case,
            contract,
            argparse.Namespace(
                finalizer=source_path,
                resolver=source_path,
                npusim=source_path,
                simulation=source_path,
            ),
            1,
            _DIGEST,
            (first, second),
        )
        self.assertTrue(report.timing_execution)
        self.assertFalse(report.compute_functional)
        self.assertFalse(report.model_functional)

    def test_marker_tamper_is_fail_closed(self) -> None:
        _case, contract = _case_and_contract()
        output = _synthetic_output()
        ce_line = next(
            line for line in output.splitlines() if line.startswith("[TRAIN_CE]")
        )
        probe_line = next(
            line
            for line in output.splitlines()
            if line.startswith("[PROGRAM_IO_PROBE]")
        )
        cases = {
            "missing_ce": output.replace(ce_line + "\n", "", 1),
            "ce_bytes": output.replace(
                "loss_write_bytes=16", "loss_write_bytes=15", 1
            ),
            "probe_sha": output.replace(
                probe_line,
                probe_line.replace(
                    probe_line.split("checksum=", 1)[1].split()[0],
                    "0" * 64,
                ),
                1,
            ),
            "residual": output.replace(
                "[P5 P2P TIMING DRAIN] residual=0",
                "[P5 P2P TIMING DRAIN] residual=1",
                1,
            ),
            "proto_wait": output + "\n[PROTO_WAIT] unresolved=1",
        }
        for name, tampered in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):
                    _observe_runtime(tampered, _DIGEST, contract)


if __name__ == "__main__":
    unittest.main()
