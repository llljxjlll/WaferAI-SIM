"""Parse flexible-Mesh runtime evidence only from real NpuSim output."""

from __future__ import annotations

import hashlib
import re

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_runtime import (
    FLEXIBLE_MESH_RUNTIME_MARKER_SCHEMA_VERSION,
    FlexibleMeshRuntimeCase,
    FlexibleMeshRuntimeMarker,
    FlexibleMeshRuntimeResidual,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.serde import canonical_digest


_STATUS = "[PROGRAM_IO] "
_PROBE = "[PROGRAM_IO_PROBE] "
_SIM = "[SIM_RESULT] "
_HOST = "[HOSTLANE] "
_HOSTSIG = "[HOSTSIG] "
_P5 = "[P5 P2P DRAIN] "
_P5_TIMING = "[P5 P2P TIMING DRAIN] "
_COLL = "[COLL_DRAIN] "
_DRAIN = "[DRAIN] "
_D2D_TYPE = "[D2D_TYPE] "
_D2D_LINK = "[D2D_LINK] "
_MEMORY = "[PROGRAM_MEMORY] "
_CREDIT = "[CREDIT] "


def _row(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line[len(prefix):].strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value.rstrip(",.")
    return result


def _rows(output: str, prefix: str) -> tuple[dict[str, str], ...]:
    result = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position >= 0:
            normalized = line[position:].split(" | ", 1)[0].rstrip(". ")
            result.append(_row(normalized, prefix))
    return tuple(result)


def _active_link_routes(output: str) -> tuple[str, ...]:
    routes = []
    for line in output.splitlines():
        position = line.find(_D2D_LINK)
        if position < 0:
            continue
        normalized = line[position:].split(" | ", 1)[0].rstrip(". ")
        row = _row(normalized, _D2D_LINK)
        if _number(row, "data_out") == 0:
            continue
        route = next(
            (token.rstrip(",.") for token in normalized.split() if "->" in token),
            None,
        )
        if route is None:
            raise SchemaError(
                "D2D_LINK lacks physical endpoints",
                path="flexible_mesh_runtime.marker.active_routes",
            )
        routes.append(route)
    canonical = tuple(sorted(set(routes)))
    if len(canonical) != len(routes):
        raise SchemaError(
            "D2D_LINK active routes must be unique",
            path="flexible_mesh_runtime.marker.active_routes",
        )
    return canonical


def _number(row: dict[str, str], key: str) -> int:
    try:
        value = int(row[key], 10)
    except (KeyError, ValueError) as error:
        raise SchemaError(
            f"missing/non-decimal marker {key!r}",
            path="flexible_mesh_runtime.marker",
        ) from error
    if value < 0:
        raise SchemaError(
            "marker count cannot be negative",
            path=f"flexible_mesh_runtime.marker.{key}",
        )
    return value


def _signature_cores(value: str) -> tuple[int, ...]:
    result = []
    for raw in value.rstrip(",").split(","):
        if not raw:
            continue
        try:
            core, count = (int(item, 10) for item in raw.split(":"))
        except (ValueError, TypeError) as error:
            raise SchemaError(
                "invalid HOSTSIG done signature",
                path="flexible_mesh_runtime.marker.hostsig",
            ) from error
        if count <= 0:
            raise SchemaError(
                "DONE count must be positive",
                path="flexible_mesh_runtime.marker.hostsig",
            )
        result.append(core)
    canonical = tuple(sorted(set(result)))
    if len(canonical) != len(result):
        raise SchemaError(
            "HOSTSIG DONE cores must be unique",
            path="flexible_mesh_runtime.marker.hostsig",
        )
    return canonical


def _validate_program_io(
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
) -> None:
    status = _rows(output, _STATUS)
    if tuple(row.get("phase") for row in status) != (
        "resolved",
        "applied",
        "verify",
    ):
        raise SchemaError(
            "ProgramIO phases are incomplete",
            path="flexible_mesh_runtime.program_io",
        )
    for row in status:
        if (
            row.get("mode") != "timing"
            or _number(row, "initializations") != len(contract.initializations)
            or _number(row, "probes") != len(contract.output_probes)
            or row.get("pass") != "1"
        ):
            raise SchemaError(
                "ProgramIO status failed",
                path="flexible_mesh_runtime.program_io",
            )
    if any(row.get("checksum") != artifact_sha256 for row in status[:2]):
        raise SchemaError(
            "ProgramIO is not bound to actual artifact SHA",
            path="flexible_mesh_runtime.program_io",
        )
    probes = {item.id: item for item in contract.output_probes}
    blobs = {item.id: item for item in contract.blobs}
    observed = _rows(output, _PROBE)
    if len(observed) != len(probes) or not probes:
        raise SchemaError(
            "ProgramIO probe coverage is not exact",
            path="flexible_mesh_runtime.program_io.probes",
        )
    seen = set()
    for row in observed:
        probe = probes.get(row.get("id", ""))
        if probe is None:
            raise SchemaError(
                "unknown ProgramIO probe",
                path="flexible_mesh_runtime.program_io.probes",
            )
        expected = blobs[probe.blob_ref].sha256
        if (
            row.get("expected_checksum") != expected
            or row.get("checksum") != expected
            or row.get("valid") != "1"
            or row.get("exact") != "1"
            or row.get("pass") != "1"
        ):
            raise SchemaError(
                "ProgramIO probe failed",
                path="flexible_mesh_runtime.program_io.probes",
            )
        seen.add(probe.id)
    if seen != set(probes):
        raise SchemaError(
            "ProgramIO probe IDs are incomplete",
            path="flexible_mesh_runtime.program_io.probes",
        )


def validate_credit_balance(output: str) -> None:
    credits = tuple(re.findall(
        r"\[CREDIT\] data_balanced=(\d+) ctrl_balanced=(\d+)", output,
    ))
    if credits != (("1", "1"),):
        raise SchemaError(
            "requires one exact balanced data/control credit marker",
            path="npusim_output",
        )


def parse_flexible_mesh_runtime_marker(
    output: str,
    *,
    case: FlexibleMeshRuntimeCase,
    manifest: LinkedProgramManifest,
    contract: ProgramIoContract,
    artifact_sha256: str,
) -> FlexibleMeshRuntimeMarker:
    """Parse one successful observation; absence is never treated as zero."""

    case.validate()
    manifest.validate()
    contract.validate_against(manifest)
    _validate_program_io(output, artifact_sha256, contract)
    simulation = _rows(output, _SIM)
    host = _rows(output, _HOST)
    signatures = _rows(output, _HOSTSIG)
    collective = _rows(output, _COLL)
    drains = _rows(output, _DRAIN)
    memory = _rows(output, _MEMORY)
    credits = _rows(output, _CREDIT)
    if (
        len(simulation) != 1
        or len(host) != 1
        or len(signatures) != 1
        or len(collective) != 1
        or "End DONE reception" not in output
    ):
        raise SchemaError(
            "required simulator completion markers are absent",
            path="flexible_mesh_runtime.marker",
        )
    if credits != ({"data_balanced": "1", "ctrl_balanced": "1"},):
        raise SchemaError(
            "credit balance marker must appear exactly once and prove both planes",
            path="flexible_mesh_runtime.marker.residual.credit_residual",
        )

    makespan = _number(simulation[0], "makespan_cycles")
    if makespan == 0 or _number(host[0], "mismatch") != 0:
        raise SchemaError(
            "simulator completion marker failed",
            path="flexible_mesh_runtime.marker",
        )
    done_cores = _signature_cores(signatures[0].get("done", ""))
    binding_by_runtime = {
        item.runtime_core_id: item.logical_core
        for item in manifest.core_bindings
    }
    expected_done = tuple(sorted(
        binding.runtime_core_id
        for binding in manifest.core_bindings
        if binding.logical_core in manifest.envelope.expected_done_cores
    ))
    if done_cores != expected_done:
        raise SchemaError(
            "HOSTSIG core coverage differs from manifest DONE envelope",
            path="flexible_mesh_runtime.marker.core_coverage",
        )
    memory_cores = tuple(sorted(_number(row, "core") for row in memory))
    if memory_cores != done_cores or len(memory_cores) != len(set(memory_cores)):
        raise SchemaError(
            "PROGRAM_MEMORY coverage differs from completed cores",
            path="flexible_mesh_runtime.marker.residual",
        )
    try:
        rank_coverage = tuple(sorted(
            binding_by_runtime[core].die_id for core in done_cores
        ))
    except KeyError as error:
        raise SchemaError(
            "HOSTSIG references an unbound runtime core",
            path="flexible_mesh_runtime.marker.core_coverage",
        ) from error

    p2p = _rows(output, _P5) + _rows(output, _P5_TIMING)
    if not p2p:
        raise SchemaError(
            "P2P drain marker is absent",
            path="flexible_mesh_runtime.marker.residual",
        )
    coll = collective[0]
    router_drains = tuple(row for row in drains if "router_residual" in row)
    link_drains = tuple(row for row in drains if "d2d_link_residual" in row)
    if len(router_drains) != 1 or len(link_drains) != 1:
        raise SchemaError(
            "router and D2D drain markers must each appear exactly once",
            path="flexible_mesh_runtime.marker.residual",
        )
    global_residual = (
        _number(router_drains[0], "router_residual")
        + _number(link_drains[0], "d2d_link_residual")
    )
    p2p_residual = sum(_number(row, "residual") for row in p2p)
    dte_residual = sum(_number(row, "dte_residual") for row in memory)
    lsu_residual = sum(_number(row, "lsu_residual") for row in memory)
    proto_wait_count = output.count("[PROTO_WAIT]")
    state_abis = tuple(
        abi
        for fragment in manifest.fragments
        for abi in fragment.state_abi
    )
    if state_abis or manifest.state_operand_bindings:
        raise SchemaError(
            "stateless MeshSlice evidence cannot infer state completion",
            path="flexible_mesh_runtime.marker.state_completion",
        )
    residual = FlexibleMeshRuntimeResidual(
        active_endpoints=sum(
            _number(coll, key)
            for key in (
                "tree_entries",
                "reduce_nodes",
                "gather",
                "reduce_rx",
                "endpoints",
            )
        ),
        active_sessions=p2p_residual + global_residual + dte_residual,
        outstanding_tags=(
            _number(coll, "dte_tokens") + _number(coll, "event")
        ),
        incomplete_barriers=_number(coll, "barriers"),
        pending_state_writes=lsu_residual,
        proto_wait_count=proto_wait_count,
        credit_residual=0,
    )
    if not residual.is_zero:
        raise SchemaError(
            "runtime residual is non-zero",
            path="flexible_mesh_runtime.marker.residual",
        )

    active_routes = _active_link_routes(output)
    typed = _rows(output, _D2D_TYPE)
    if len(typed) != 1:
        raise SchemaError(
            "typed D2D summary must appear exactly once",
            path="flexible_mesh_runtime.marker.active_routes",
        )
    if case.expected_rank_count == 1:
        if active_routes or _number(typed[0], "data_out") != 0:
            raise SchemaError(
                "LOCAL runtime emitted transport",
                path="flexible_mesh_runtime.marker.active_routes",
            )
    elif not active_routes:
        raise SchemaError(
            "communicating Mesh lacks typed D2D evidence",
            path="flexible_mesh_runtime.marker.active_routes",
        )

    prefixes = (
        _STATUS,
        _PROBE,
        _SIM,
        _HOST,
        _HOSTSIG,
        _P5,
        _P5_TIMING,
        _COLL,
        _DRAIN,
        _D2D_TYPE,
        _D2D_LINK,
        _MEMORY,
        _CREDIT,
    )
    marker_lines = tuple(
        line[line.find(prefix):].split(" | ", 1)[0].rstrip(". ")
        for line in output.splitlines()
        for prefix in prefixes
        if line.find(prefix) >= 0
    )
    result = FlexibleMeshRuntimeMarker(
        schema_version=FLEXIBLE_MESH_RUNTIME_MARKER_SCHEMA_VERSION,
        mesh_digest=case.workload.mesh.digest,
        workload_digest=case.workload.digest,
        manifest_digest=canonical_digest(manifest),
        program_io_digest=canonical_digest(contract),
        makespan_cycles=makespan,
        rank_coverage=rank_coverage,
        core_coverage=done_cores,
        active_routes=active_routes,
        state_completion=(),
        residual=residual,
        marker_digest=hashlib.sha256(
            "\n".join(marker_lines).encode("utf-8")
        ).hexdigest(),
    )
    result.validate(case)
    return result


__all__ = ["parse_flexible_mesh_runtime_marker", "validate_credit_balance"]
