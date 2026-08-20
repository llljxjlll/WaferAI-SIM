#!/usr/bin/env python3
"""Finalize and run the real tiny E1 timing ProgramIo contract twice."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


_ROOT = Path(__file__).resolve().parents[4]
_UNIT = _ROOT / "llm/test/frontend/unit"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_UNIT))

from llm.frontend.wafer_frontend.passes import build_timing_program_io  # noqa: E402
from llm.frontend.wafer_frontend.passes.program_io import (  # noqa: E402
    _resolved_state_abis,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (  # noqa: E402
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.program_io import (  # noqa: E402
    ProgramIoContract,
    ProgramHbmTarget,
    ProgramIoPurpose,
    ProgramSramTarget,
    ProgramSramInitialization,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json  # noqa: E402
from llm.frontend.wafer_frontend.schema.persistent_state import (  # noqa: E402
    PersistentStateAccess,
)
from test_n6_pipeline import _compile_through_n6  # noqa: E402


_STATUS_PREFIX = "[PROGRAM_IO] "
_PROBE_PREFIX = "[PROGRAM_IO_PROBE] "
_HOST_PREFIX = "[HOSTLANE] "
_HOSTSIG_PREFIX = "[HOSTSIG] "
_P5_DRAIN_PREFIX = "[P5 P2P DRAIN] "
_P5_TIMING_PREFIX = "[P5 P2P TIMING DRAIN] "
_COLL_DRAIN_PREFIX = "[COLL_DRAIN] "
_GLOBAL_DRAIN_PREFIX = "[DRAIN] "
_SIM_RESULT_PREFIX = "[SIM_RESULT] "
_MEMORY_PREFIX = "[PROGRAM_MEMORY] "


def _fail(message: str) -> None:
    raise RuntimeError(f"[PROGRAM_IO E1-T] FAIL: {message}")


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    expect_success: bool,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if (completed.returncode == 0) != expect_success:
        expected = "success" if expect_success else "failure"
        _fail(
            f"expected {expected} from {' '.join(command)}; "
            f"exit={completed.returncode}\n{completed.stdout}"
        )
    return completed


def _row(line: str, prefix: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line[len(prefix) :].strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def _rows(output: str, prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position < 0:
            continue
        normalized = line[position:].split(" | ", 1)[0].rstrip()
        if normalized.endswith("."):
            normalized = normalized[:-1]
        rows.append(_row(normalized, prefix))
    return rows


def _marker_lines(output: str, prefixes: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for line in output.splitlines():
        positions = [line.find(prefix) for prefix in prefixes]
        positions = [position for position in positions if position >= 0]
        if positions:
            result.append(line[min(positions) :])
    return tuple(result)


def _number(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key], 10)
    except (KeyError, ValueError) as error:
        _fail(f"missing/non-decimal {key!r} in {row}")
        raise AssertionError from error


def _signature_counts(value: str, arity: int) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        parts = tuple(int(part, 10) for part in item.split(":"))
        if len(parts) != arity:
            _fail(f"bad HOSTSIG item {item!r}")
        result.append(parts)
    return sorted(result)


def _timing_partial_on_borrowed(
    contract: ProgramIoContract,
    manifest: object,
) -> ProgramIoContract:
    first = next(
        entry
        for entry in contract.initializations
        if entry.purpose is not ProgramIoPurpose.TIMING_PARTIAL
    )
    changed = ProgramSramInitialization.create(
        **{
            **first._semantic_key(),
            "purpose": ProgramIoPurpose.TIMING_PARTIAL,
        }
    )
    initializations = tuple(
        changed if entry.id == first.id else entry
        for entry in contract.initializations
    )
    # This is an intentionally stable forged input.  Do not validate it in
    # Python: both independent validators must reject it on ownership.
    return ProgramIoContract.create(
        producer_pass="program_io_e1_t_negative",
        mode=contract.mode,
        source_manifest=manifest,
        program_artifact_sha256=contract.program_artifact_sha256,
        blobs=contract.blobs,
        initializations=initializations,
        output_probes=contract.output_probes,
    )


def _state_payloads(source: object) -> tuple[dict[str, bytes], dict[str, bytes]]:
    resolved = _resolved_state_abis(source)
    seeds = {
        item.abi.state_ref: bytes([index + 1]) * item.abi.size_bytes
        for index, item in enumerate(resolved)
    }
    expected = {
        item.abi.state_ref: seeds[item.abi.state_ref]
        for item in resolved
        if item.abi.access is PersistentStateAccess.READ_WRITE
    }
    if len(resolved) != 12 or len(seeds) != 12 or len(expected) != 4:
        _fail(
            "wrong stateful E1 closure: "
            f"abis={len(resolved)} seeds={len(seeds)} expected={len(expected)}"
        )
    return seeds, expected


def _program_memory_oracle(source: object) -> dict[int, dict[str, int]]:
    fields = (
        "lsu_issued",
        "lsu_completed",
        "lsu_hbm_read_bytes",
        "lsu_hbm_write_bytes",
        "lsu_sram_read_bytes",
        "lsu_sram_write_bytes",
        "lsu_residual",
        "dte_residual",
    )
    manifest = source.manifest
    runtime_by_logical = {
        binding.logical_core: binding.runtime_core_id
        for binding in manifest.core_bindings
    }
    active_cores = tuple(sorted(set(runtime_by_logical.values())))
    if active_cores != (0, 16):
        _fail(f"wrong tiny E1 active runtime cores: {active_cores}")
    result = {
        core: {field: 0 for field in fields}
        for core in active_cores
    }
    fragments = {fragment.id: fragment for fragment in source.leaf_fragments}
    state_abis = {
        item.abi.id: item.abi for item in _resolved_state_abis(source)
    }
    if len(manifest.state_operand_bindings) != 16:
        _fail(
            "wrong state operand witness count: "
            f"{len(manifest.state_operand_bindings)}"
        )
    for binding in manifest.state_operand_bindings:
        fragment = fragments[binding.fragment_id]
        streams = tuple(
            stream
            for stream in fragment.core_streams
            if stream.logical_core == binding.logical_core
        )
        if len(streams) != 1:
            _fail("state operand did not resolve to one command stream")
        record = streams[0].records[binding.fragment_record_index]
        abi = state_abis[binding.state_abi_id]
        size_bytes = record.operands[1].literal_value
        if size_bytes != abi.size_bytes:
            _fail("LSU record bytes disagree with StateABI")
        core = runtime_by_logical[binding.logical_core]
        row = result[core]
        row["lsu_issued"] += 1
        row["lsu_completed"] += 1
        if record.opcode is RecordOpcode.LSU_LOAD:
            row["lsu_hbm_read_bytes"] += size_bytes
            row["lsu_sram_write_bytes"] += size_bytes
        elif record.opcode is RecordOpcode.LSU_STORE:
            row["lsu_hbm_write_bytes"] += size_bytes
            row["lsu_sram_read_bytes"] += size_bytes
        else:
            _fail(f"state operand references non-LSU opcode {record.opcode}")
    return result


def _validate_program_memory(
    output: str,
    expected: dict[int, dict[str, int]],
) -> tuple[str, ...]:
    rows = _rows(output, _MEMORY_PREFIX)
    cores = [_number(row, "core") for row in rows]
    if cores != list(expected):
        _fail(f"PROGRAM_MEMORY cores are missing, duplicated, or reordered: {cores}")
    required = {"core", *next(iter(expected.values())).keys()}
    for row in rows:
        if set(row) != required:
            _fail(f"PROGRAM_MEMORY fields changed: {row}")
        core = _number(row, "core")
        actual = {
            key: _number(row, key)
            for key in expected[core]
        }
        if actual != expected[core]:
            _fail(
                f"PROGRAM_MEMORY core {core} disagrees with linked LSU oracle: "
                f"{actual} != {expected[core]}"
            )
    return _marker_lines(output, (_MEMORY_PREFIX,))


def _runtime_hardware(source: Path, destination: Path) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    memory = raw["memory"]
    sram = memory["sram"]
    comm = next(region for region in sram["regions"] if region["name"] == "comm")
    memory["sram_size"] = 65536
    sram["capacity_bytes"] = 65536
    sram["regions"] = [
        {
            **comm,
            "base_bytes": 0,
            "size_bytes": 65536,
            "allocator": "block",
            "spillable": False,
        }
    ]
    destination.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _dramsys_hardware(source: Path, destination: Path) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    stacks = raw["memory_system"]["hbm_stacks"]
    if len(stacks) != 2:
        _fail(f"wrong tiny E1 HBM stack count: {len(stacks)}")
    for stack in stacks:
        stack["backend"] = "dramsys"
    destination.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _validate_program_io(
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
    source: object,
) -> tuple[str, ...]:
    statuses = _rows(output, _STATUS_PREFIX)
    if len(statuses) != 3 or [row.get("phase") for row in statuses] != [
        "resolved",
        "applied",
        "verify",
    ]:
        _fail(f"wrong ProgramIo phase closure: {statuses}")
    for row in statuses:
        if (
            row.get("mode") != "timing"
            or _number(row, "initializations") != 70
            or _number(row, "probes") != 6
            or row.get("pass") != "1"
        ):
            _fail(f"failed ProgramIo status: {row}")
    if statuses[0].get("checksum") != artifact_sha256 or statuses[1].get(
        "checksum"
    ) != artifact_sha256:
        _fail("resolved/applied checksum is not the actual artifact SHA-256")

    probes = _rows(output, _PROBE_PREFIX)
    if len(probes) != 6:
        _fail(f"expected six output probes, got {probes}")
    entries = {entry.id: entry for entry in contract.output_probes}
    state_abis = {item.abi.id: item.abi for item in _resolved_state_abis(source)}
    for probe in probes:
        entry = entries.get(probe.get("id", ""))
        if entry is None:
            _fail(f"unknown output probe marker: {probe}")
        if (
            probe.get("valid") != "1"
            or probe.get("exact") != "1"
            or probe.get("pass") != "1"
            or probe.get("checksum") != probe.get("expected_checksum")
        ):
            _fail(f"failed output probe: {probe}")
        common = {
            "id",
            "address",
            "bytes",
            "expected_checksum",
            "checksum",
            "valid",
            "exact",
            "pass",
        }
        if type(entry.target) is ProgramHbmTarget:
            abi = state_abis[entry.target.state_abi_id]
            if (
                set(probe) != common | {"die"}
                or _number(probe, "die") != abi.die_id
                or _number(probe, "address") != abi.address
                or _number(probe, "bytes") != abi.size_bytes
            ):
                _fail(f"HBM probe marker lost exact die/address closure: {probe}")
        elif type(entry.target) is ProgramSramTarget:
            if (
                set(probe) != common | {"core"}
                or _number(probe, "core") != entry.target.runtime_core_id
                or _number(probe, "bytes") != entry.length_bytes
            ):
                _fail(f"SRAM probe marker compatibility changed: {probe}")
        else:
            _fail(f"unknown output probe target: {entry.target!r}")
    return _marker_lines(output, (_STATUS_PREFIX, _PROBE_PREFIX))


def _validate_control_and_drains(output: str) -> tuple[str, ...]:
    if "[PROTO_WAIT]" in output or "End DONE reception" not in output:
        _fail("runtime did not close the DONE path")
    simulation = _rows(output, _SIM_RESULT_PREFIX)
    if len(simulation) != 1 or _number(simulation[0], "makespan_cycles") <= 0:
        _fail(f"missing/non-positive simulation makespan: {simulation}")
    host = _rows(output, _HOST_PREFIX)
    signatures = _rows(output, _HOSTSIG_PREFIX)
    if len(host) != 1 or len(signatures) != 1:
        _fail("HOSTLANE/HOSTSIG marker missing or duplicated")
    if (
        _number(host[0], "done_total"),
        _number(host[0], "ack_total"),
        _number(host[0], "mismatch"),
    ) != (2, 4, 0):
        _fail(f"wrong ACK/DONE totals: {host[0]}")
    done = _signature_counts(signatures[0].get("done", ""), 2)
    ack = _signature_counts(signatures[0].get("ack", ""), 3)
    ack_by_core: Counter[int] = Counter()
    for core, _lane, count in ack:
        ack_by_core[core] += count
    if done != [(0, 1), (16, 1)] or dict(ack_by_core) != {0: 2, 16: 2}:
        _fail(f"wrong ACK/DONE core signatures: done={done}, ack={ack}")

    p5 = _rows(output, _P5_DRAIN_PREFIX)
    if (
        len(p5) != 2
        or {_number(row, "core") for row in p5} != {0, 16}
        or any(_number(row, "residual") != 0 for row in p5)
    ):
        _fail(f"P2P endpoint drain is incomplete: {p5}")
    timing = _rows(output, _P5_TIMING_PREFIX)
    if len(timing) != 1 or _number(timing[0], "residual") != 0:
        _fail(f"P2P timing sideband did not drain: {timing}")
    collective = _rows(output, _COLL_DRAIN_PREFIX)
    if len(collective) != 1:
        _fail("global collective drain marker missing or duplicated")
    for key in (
        "tree_entries",
        "reduce_nodes",
        "barriers",
        "gather",
        "reduce_rx",
        "endpoints",
        "dte_tokens",
        "event",
    ):
        if _number(collective[0], key) != 0:
            _fail(f"collective drain {key} is non-zero: {collective[0]}")
    global_rows = _rows(output, _GLOBAL_DRAIN_PREFIX)
    residuals = {
        key: _number(row, key)
        for row in global_rows
        for key in ("router_residual", "d2d_link_residual")
        if key in row
    }
    if residuals != {"router_residual": 0, "d2d_link_residual": 0}:
        _fail(f"router/D2D drains are incomplete: {residuals}")
    return _marker_lines(
        output,
        (
            _SIM_RESULT_PREFIX,
            _HOST_PREFIX,
            _HOSTSIG_PREFIX,
            _P5_DRAIN_PREFIX,
            _P5_TIMING_PREFIX,
            _COLL_DRAIN_PREFIX,
            _GLOBAL_DRAIN_PREFIX,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    args.runtime_root.mkdir(parents=True, exist_ok=True)

    run = _compile_through_n6()
    if len(run.linked.entries) != 1:
        _fail("tiny E1 must contain exactly one linked profile")
    source = run.linked.entries[0]
    manifest = source.manifest

    with tempfile.TemporaryDirectory(
        prefix="program-io-e1-t-", dir=args.runtime_root
    ) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        artifact0_path = directory / "program-0.npup"
        artifact1_path = directory / "program-1.npup"
        report_path = directory / "report.json"
        sidecar_path = directory / "program-io.json"
        bad_purpose_path = directory / "bad-purpose.json"
        bad_artifact_path = directory / "bad-artifact.npup"
        hardware_path = directory / "hardware-e1.json"
        dramsys_hardware_path = directory / "hardware-e1-dramsys.json"
        manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
        _runtime_hardware(args.hardware, hardware_path)

        for artifact_path in (artifact0_path, artifact1_path):
            command = [
                str(args.finalizer),
                "--input",
                str(manifest_path),
                "--output",
                str(artifact_path),
            ]
            if artifact_path == artifact0_path:
                command += ["--report", str(report_path)]
            _run(command, cwd=args.runtime_root, timeout=30, expect_success=True)
        artifact = artifact0_path.read_bytes()
        if artifact1_path.read_bytes() != artifact:
            _fail("production finalizer is not byte deterministic")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("artifact_bytes"),
            report.get("core_count"),
            report.get("record_count"),
            report.get("relocation_count"),
        ) != (27676, 2, 220, 354):
            _fail(f"wrong final E1 artifact golden: {report}")
        artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        if report.get("artifact_sha256") != artifact_sha256:
            _fail("finalizer report SHA does not hash actual artifact bytes")

        state_seeds, state_expected = _state_payloads(source)
        contract = build_timing_program_io(
            source,
            artifact_sha256,
            state_seed_overrides=state_seeds,
            state_expected_overrides=state_expected,
        )
        second_contract = build_timing_program_io(
            source,
            artifact_sha256,
            state_seed_overrides=state_seeds,
            state_expected_overrides=state_expected,
        )
        if contract != second_contract or canonical_json(contract) != canonical_json(
            second_contract
        ):
            _fail("timing ProgramIo sidecar is not deterministic")
        if len(contract.initializations) != 70 or len(contract.output_probes) != 6:
            _fail("final E1 ProgramIo must contain 70 initializations and 6 probes")
        memory_oracle = _program_memory_oracle(source)
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")

        resolved = _run(
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(artifact0_path),
                str(sidecar_path),
            ],
            cwd=args.runtime_root,
            timeout=30,
            expect_success=True,
        )
        if "initializations=70 probes=6" not in resolved.stdout:
            _fail(f"C++ resolver lost E1 entry counts\n{resolved.stdout}")

        bad_purpose_path.write_text(
            canonical_json(_timing_partial_on_borrowed(contract, manifest)),
            encoding="utf-8",
        )
        purpose = _run(
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(artifact0_path),
                str(bad_purpose_path),
            ],
            cwd=args.runtime_root,
            timeout=30,
            expect_success=False,
        )
        if "timing_partial initialization requires OWNED BufferABI" not in purpose.stdout:
            _fail(f"wrong ownership rejection\n{purpose.stdout}")

        corrupted = bytearray(artifact)
        corrupted[-1] ^= 0x01
        bad_artifact_path.write_bytes(corrupted)
        artifact_failure = _run(
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(bad_artifact_path),
                str(sidecar_path),
            ],
            cwd=args.runtime_root,
            timeout=30,
            expect_success=False,
        )
        if "does not match the actual encoded ProgramArtifact bytes" not in artifact_failure.stdout:
            _fail(f"wrong artifact-byte rejection\n{artifact_failure.stdout}")

        runtime_signatures: list[
            tuple[
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ]
        ] = []
        makespan_cycles = 0
        for _repeat in range(2):
            execution = _run(
                [
                    str(args.npusim),
                    "--program",
                    str(artifact0_path),
                    "--linked-manifest",
                    str(manifest_path),
                    "--program-io",
                    str(sidecar_path),
                    "--hardware-config",
                    str(hardware_path),
                    "--simulation-config",
                    str(args.simulation),
                    "--mapping-config",
                    str(args.mapping),
                    "--trace-window",
                    "1000000",
                ],
                cwd=args.runtime_root,
                timeout=180,
                expect_success=True,
            )
            program_io_signature = _validate_program_io(
                execution.stdout,
                artifact_sha256,
                contract,
                source,
            )
            memory_signature = _validate_program_memory(
                execution.stdout, memory_oracle
            )
            control_signature = _validate_control_and_drains(execution.stdout)
            current_makespan = _number(
                _rows(execution.stdout, _SIM_RESULT_PREFIX)[0],
                "makespan_cycles",
            )
            if makespan_cycles == 0:
                makespan_cycles = current_makespan
            elif current_makespan != makespan_cycles:
                _fail(
                    "repeat runtime makespan changed: "
                    f"{makespan_cycles} != {current_makespan}"
                )
            runtime_signatures.append(
                (
                    program_io_signature,
                    memory_signature,
                    control_signature,
                )
            )
        if runtime_signatures[0] != runtime_signatures[1]:
            _fail("repeat runtime ProgramIo/control/drain signature changed")
        _dramsys_hardware(hardware_path, dramsys_hardware_path)
        dramsys = _run(
            [
                str(args.npusim),
                "--program",
                str(artifact0_path),
                "--linked-manifest",
                str(manifest_path),
                "--program-io",
                str(sidecar_path),
                "--hardware-config",
                str(dramsys_hardware_path),
                "--simulation-config",
                str(args.simulation),
                "--mapping-config",
                str(args.mapping),
                "--trace-window",
                "1000000",
            ],
            cwd=args.runtime_root,
            timeout=30,
            expect_success=False,
        )
        dramsys_status = _rows(dramsys.stdout, _STATUS_PREFIX)
        if (
            not dramsys_status
            or dramsys_status[-1].get("phase") != "apply"
            or dramsys_status[-1].get("pass") != "0"
            or "HBM backend does not support debug peeking" not in dramsys.stdout
            or _SIM_RESULT_PREFIX in dramsys.stdout
            or _MEMORY_PREFIX in dramsys.stdout
        ):
            _fail(
                "DRAMSys ProgramIo did not fail closed before simulation:\n"
                f"{dramsys.stdout}"
            )
        hbm_read_bytes = sum(
            row["lsu_hbm_read_bytes"] for row in memory_oracle.values()
        )
        hbm_write_bytes = sum(
            row["lsu_hbm_write_bytes"] for row in memory_oracle.values()
        )

    print(
        "[PROGRAM_IO E1-T] PASS: artifact=27676B records=220 relocs=354 "
        "initializations=70 probes=6 state_seeds=12 state_probes=4 "
        f"lsu_hbm_read_bytes={hbm_read_bytes} "
        f"lsu_hbm_write_bytes={hbm_write_bytes} repeat=2 "
        "dramsys_failclosed=1 ACK=4 DONE=2 drains=0 "
        f"makespan_cycles={makespan_cycles} "
        f"sha256={artifact_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
