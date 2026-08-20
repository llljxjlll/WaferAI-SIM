from __future__ import annotations

from dataclasses import replace
import ctypes
import errno
from functools import lru_cache
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import patch

from llm.frontend.wafer_frontend.passes import (
    build_stage4_pd_oracle,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramHbmTarget,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.stage4_pd_evidence import (
    Stage4PdNamedDigest,
    Stage4PdRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.state_transfer import (
    SegmentedKvStateTransferContract,
)

import run_stage4_pd_runtime as runtime
from run_stage4_pd_runtime import (
    _build_report,
    _expected_link_traffic,
    _expected_report_names,
    _leaf_fragments,
    _memory_expected,
    _observe_runtime,
    _publish_report_texts,
    _rebuild_sidecar,
    _select_case,
    _transfer_units,
    _validate_static_case,
)
from stage4_pd_cases import Stage4PdCaseKind, build_stage4_pd_case


_DIGEST = "1" * 64


@lru_cache(maxsize=3)
def _case(kind: Stage4PdCaseKind):
    case = build_stage4_pd_case(kind)
    oracle = build_stage4_pd_oracle(case.pd_plan)
    static = _validate_static_case(case, oracle)
    contract = _rebuild_sidecar(
        case,
        hashlib.sha256(b"stage4-runtime-parser").hexdigest(),
    )
    memory = _memory_expected(case, _leaf_fragments(case))
    return case, oracle, static, contract, memory


def _synthetic_output(kind: Stage4PdCaseKind) -> str:
    case, oracle, _static, contract, memory = _case(kind)
    artifact_sha256 = contract.program_artifact_sha256
    lines = [
        (
            f"[PROGRAM_IO] phase={phase} mode=timing "
            f"checksum={artifact_sha256} "
            f"initializations={len(contract.initializations)} "
            f"probes={len(contract.output_probes)} pass=1"
        )
        for phase in ("resolved", "applied", "verify")
    ]
    blobs = {blob.id: blob for blob in contract.blobs}
    state_abis = {
        abi.id: abi
        for fragment in _leaf_fragments(case)
        for abi in fragment.state_abi
    }
    for entry in contract.output_probes:
        blob = blobs[entry.blob_ref]
        location = ""
        if type(entry.target) is ProgramSramTarget:
            location = f"core={entry.target.runtime_core_id}"
        elif type(entry.target) is ProgramHbmTarget:
            abi = state_abis[entry.target.state_abi_id]
            location = (
                f"die={abi.die_id} "
                f"address={abi.address + entry.offset_bytes}"
            )
        lines.append(
            f"[PROGRAM_IO_PROBE] id={entry.id} bytes={entry.length_bytes} "
            f"expected_checksum={blob.sha256} checksum={blob.sha256} "
            f"valid=1 exact=1 pass=1 {location}"
        )
    for core, row in memory.items():
        lines.append(
            "[PROGRAM_MEMORY] "
            f"core={core} "
            + " ".join(f"{name}={value}" for name, value in row.items())
        )
    lines.extend(("[SIM_RESULT] makespan_cycles=1234", "End DONE reception"))
    runtime_cores = tuple(memory)
    lines.append(
        f"[HOSTLANE] ack_total={2 * len(runtime_cores)} "
        f"done_total={len(runtime_cores)} mismatch=0"
    )
    ack = ",".join(
        f"{core}:{lane}:1"
        for core in runtime_cores
        for lane in (0, 1)
    )
    done = ",".join(f"{core}:1" for core in runtime_cores)
    lines.append(f"[HOSTSIG] ack={ack} done={done}")
    lines.append("[P5 P2P TIMING DRAIN] residual=0")
    if kind is not Stage4PdCaseKind.FUSED:
        lines.extend(
            f"[P5 P2P DRAIN] core={core} residual=0"
            for core in runtime_cores
        )
    lines.append(
        "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
        "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0"
    )
    lines.append("[DRAIN] router_residual=0 d2d_link_residual=0")
    links = _expected_link_traffic(case)
    requests = sum(value[0] for value in links.values())
    packets = sum(value[1] for value in links.values())
    lines.append(
        "[D2D_TYPE] "
        f"request_in={requests} request_out={requests} "
        f"ack_in={2 * requests} ack_out={2 * requests} "
        f"data_in={packets} data_out={packets}"
    )
    for index, ((source, destination), (request_count, data_count)) in enumerate(
        sorted(links.items())
    ):
        lines.append(
            f"[D2D_LINK] idx={2 * index} die{source}->die{destination} "
            f"dir=EAST req_in={request_count} req_out={request_count} "
            "ack_in=0 ack_out=0 "
            f"data_in={data_count} data_out={data_count}"
        )
        lines.append(
            f"[D2D_LINK] idx={2 * index + 1} die{destination}->die{source} "
            "dir=WEST req_in=0 req_out=0 "
            f"ack_in={2 * request_count} ack_out={2 * request_count} "
            "data_in=0 data_out=0"
        )
    return "\n".join(lines)


class Stage4PdRuntimeParserTest(unittest.TestCase):
    def test_pdr_static_witness_closes_wave_records_and_rejects_stale_values(
        self,
    ) -> None:
        case, oracle, static, _contract, _memory = _case(
            Stage4PdCaseKind.PDR
        )
        self.assertEqual(
            (static.record_count, static.runtime_relocation_count),
            (971, 858),
        )
        for record_count, runtime_relocations in (
            (861, 528),
            (977, 876),
            (983, 894),
        ):
            with self.subTest(
                record_count=record_count,
                runtime_relocations=runtime_relocations,
            ):
                stale = replace(
                    static,
                    record_count=record_count,
                    runtime_relocation_count=runtime_relocations,
                )
                with patch.dict(
                    runtime._STATIC_GOLDENS,
                    {Stage4PdCaseKind.PDR: stale},
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "formal carrier/artifact structure changed",
                    ):
                        _validate_static_case(case, oracle)

    def test_pdr_selection_and_segment_units_use_formal_carrier(self) -> None:
        self.assertIs(_select_case("pdr"), Stage4PdCaseKind.PDR)
        case, _oracle, _static, _contract, _memory = _case(
            Stage4PdCaseKind.PDR
        )
        segmented = case.profile.lowering_context.projection.state_transfers[0]
        self.assertIs(type(segmented), SegmentedKvStateTransferContract)
        units = _transfer_units(segmented)
        self.assertEqual(
            tuple(index for index, _size in units),
            tuple(range(len(segmented.segments))),
        )
        self.assertEqual(sum(size for _index, size in units), segmented.bytes)
        with self.assertRaisesRegex(TypeError, "sliced or segmented"):
            _transfer_units(object())  # type: ignore[arg-type]

    def test_fused_pds_and_pdr_static_parser_report_closure(self) -> None:
        for kind, expected in (
            (Stage4PdCaseKind.FUSED, (0, 0, 0)),
            (Stage4PdCaseKind.PDS, (4, 1024, 1)),
            (Stage4PdCaseKind.PDR, (8, 1024, 2)),
        ):
            with self.subTest(kind=kind.value):
                case, oracle, static, contract, memory = _case(kind)
                output = _synthetic_output(kind)
                observation = _observe_runtime(
                    case,
                    oracle,
                    output,
                    contract.program_artifact_sha256,
                    contract,
                    memory,
                )
                self.assertEqual(
                    (
                        observation.d2d.state_transfer_count,
                        observation.d2d.state_transport_logical_bytes,
                        observation.d2d.unique_endpoint_route_count,
                    ),
                    expected,
                )
                report = _build_report(
                    case,
                    oracle,
                    static,
                    {
                        "record_count": static.record_count,
                        "relocation_count": static.address_relocation_count,
                    },
                    b"stage4-runtime-parser",
                    contract,
                    (observation, observation),
                    tool_digests=tuple(
                        Stage4PdNamedDigest(name, _DIGEST)
                        for name in (
                            "finalizer",
                            "npusim",
                            "resolver",
                            "runner",
                        )
                    ),
                    hardware_digest=_DIGEST,
                    simulation_digest=_DIGEST,
                    mapping_digest=_DIGEST,
                )
                report.validate_against(
                    case.pd_plan,
                    oracle,
                    case.manifest,
                    case.planning_context,
                    case.scheduling_context,
                )
                self.assertEqual(
                    loads_dataclass(
                        Stage4PdRuntimeReport,
                        canonical_json(report),
                        path="stage4_pd_runtime_report",
                    ),
                    report,
                )
                self.assertEqual(
                    tuple(item.name for item in report.tool_digests),
                    ("finalizer", "npusim", "resolver", "runner"),
                )
                self.assertEqual(
                    tuple(item.name for item in report.input_digests),
                    (
                        "hardware",
                        "manifest",
                        "mapping",
                        "oracle",
                        "plan",
                        "policy",
                        "program_io",
                        "simulation",
                        "spec",
                    ),
                )
                self.assertFalse(report.model_functional)
                self.assertEqual(report.repeat_count, 2)

    def test_parser_fails_closed_on_transport_memory_program_io_and_done(self) -> None:
        kind = Stage4PdCaseKind.PDS
        case, oracle, _static, contract, memory = _case(kind)
        output = _synthetic_output(kind)
        first_core = next(iter(memory))
        read_bytes = memory[first_core]["lsu_hbm_read_bytes"]
        for label, corrupted in (
            ("transport", output.replace("data_out=64", "data_out=63", 1)),
            (
                "memory",
                output.replace(
                    f"lsu_hbm_read_bytes={read_bytes}",
                    f"lsu_hbm_read_bytes={read_bytes + 1}",
                    1,
                ),
            ),
            ("program_io", output.replace("pass=1", "pass=0", 1)),
            ("done", output.replace("End DONE reception", "")),
        ):
            with self.subTest(label=label):
                with self.assertRaises(RuntimeError):
                    _observe_runtime(
                        case,
                        oracle,
                        corrupted,
                        contract.program_artifact_sha256,
                        contract,
                        memory,
                    )

        observation = _observe_runtime(
            case,
            oracle,
            output,
            contract.program_artifact_sha256,
            contract,
            memory,
        )
        with self.assertRaisesRegex(RuntimeError, "repeat changed"):
            _build_report(
                case,
                oracle,
                _case(kind)[2],
                {
                    "record_count": _case(kind)[2].record_count,
                    "relocation_count": _case(kind)[2].address_relocation_count,
                },
                b"stage4-runtime-parser",
                contract,
                (
                    observation,
                    replace(observation, makespan_cycles=1235),
                ),
                tool_digests=tuple(
                    Stage4PdNamedDigest(name, _DIGEST)
                    for name in ("finalizer", "npusim", "resolver", "runner")
                ),
                hardware_digest=_DIGEST,
                simulation_digest=_DIGEST,
                mapping_digest=_DIGEST,
            )


class Stage4PdReportRootTest(unittest.TestCase):
    @staticmethod
    def _entries(
        kind: Stage4PdCaseKind = Stage4PdCaseKind.FUSED,
    ) -> tuple[tuple[str, str], ...]:
        return tuple((name, "") for name in _expected_report_names(kind))

    def test_exact_set_is_atomically_published_without_npup(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "evidence"
            _publish_report_texts(
                root, Stage4PdCaseKind.FUSED, self._entries()
            )
            self.assertEqual(
                tuple(sorted(path.name for path in root.iterdir())),
                _expected_report_names(Stage4PdCaseKind.FUSED),
            )
            self.assertFalse(
                any(path.suffix == ".npup" for path in root.iterdir())
            )

    def test_incomplete_existing_symlink_and_publish_race_fail_closed(
        self,
    ) -> None:
        kind = Stage4PdCaseKind.FUSED
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            with self.assertRaisesRegex(RuntimeError, "exact reviewed"):
                _publish_report_texts(
                    parent / "incomplete", kind, self._entries()[:-1]
                )
            existing = parent / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(RuntimeError, "already exist"):
                _publish_report_texts(existing, kind, self._entries())
            destination = parent / "destination"
            destination.mkdir()
            symlink = parent / "symlink"
            symlink.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink ancestor"):
                _publish_report_texts(symlink, kind, self._entries())

            class _Rename:
                argtypes = None
                restype = None

                def __call__(self, *_args: object) -> int:
                    ctypes.set_errno(errno.EEXIST)
                    return -1

            class _Lib:
                renameat2 = _Rename()

            raced = parent / "raced"
            with patch(
                "run_stage4_pd_runtime.ctypes.CDLL",
                return_value=_Lib(),
            ):
                with self.assertRaisesRegex(RuntimeError, "appeared"):
                    _publish_report_texts(raced, kind, self._entries())
            self.assertFalse(raced.exists())


if __name__ == "__main__":
    unittest.main()
