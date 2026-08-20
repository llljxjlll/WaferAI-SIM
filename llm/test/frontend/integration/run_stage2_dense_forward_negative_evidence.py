#!/usr/bin/env python3
"""Execute the reviewed Stage2 dense-forward fail-closed witnesses."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable


_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from llm.frontend.wafer_frontend.errors import (  # noqa: E402
    FrontendError,
    SchemaError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0  # noqa: E402
from llm.frontend.wafer_frontend.passes.logical_expand import (  # noqa: E402
    logical_expand,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (  # noqa: E402
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.capability import (  # noqa: E402
    CapabilityStatus,
)
from llm.frontend.wafer_frontend.schema.common import (  # noqa: E402
    DType,
    stable_artifact_id,
)
from llm.frontend.wafer_frontend.schema.experiment import (  # noqa: E402
    InferOutput,
)
from llm.frontend.wafer_frontend.schema.ir0 import (  # noqa: E402
    CrossEntropyForwardWorkload,
    CrossEntropyReduction,
    EdgeKind,
    GraphEdge,
    OpKind,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1  # noqa: E402
from llm.frontend.wafer_frontend.schema.program_io import (  # noqa: E402
    ProgramHbmTarget,
    ProgramIoMode,
    ProgramIoTargetKind,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
)
from llm.frontend.wafer_frontend.schema.stage2_dense_forward_evidence import (  # noqa: E402
    STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
    STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION,
    Stage2DenseForwardArtifactEvidence,
    Stage2DenseForwardCompileEvidence,
    Stage2DenseForwardControlEvidence,
    Stage2DenseForwardCoreCount,
    Stage2DenseForwardNamedCount,
    Stage2DenseForwardProbeEvidence,
    Stage2DenseForwardRepeatEvidence,
    Stage2DenseForwardRuntimeReport,
    Stage2DenseForwardSidecarEvidence,
    Stage2DenseForwardToolEvidence,
    _CASE_GOLDENS,
)
from stage2_dense_forward_cases import (  # noqa: E402
    _spec,
    build_stage2_dense_forward_case,
)


_SCHEMA_VERSION = "wafer_frontend.stage2_dense_forward_negative_evidence/v1alpha1"
_PRODUCER_PASS = "stage2_dense_forward_negative_runner"
_DIGEST = "1" * 64
_SOURCE_PATHS = (
    "llm/frontend/wafer_frontend/passes/logical_expand.py",
    "llm/frontend/wafer_frontend/policies/naive_project_to_ir2.py",
    "llm/frontend/wafer_frontend/schema/stage2_dense_forward_evidence.py",
    "llm/src/frontend/program_io.cpp",
    "llm/test/frontend/integration/stage2_dense_forward_cases.py",
    "llm/test/frontend/integration/run_stage2_dense_forward_negative_evidence.py",
    "llm/unittest/program_io_selftest_main.cpp",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _named_digest(name: str, digest: str) -> dict[str, str]:
    return {"name": name, "sha256": digest}


def _bindings(
    *,
    tools: tuple[tuple[str, Path], ...],
    sources: tuple[str, ...],
    inputs: tuple[tuple[str, str], ...],
) -> dict[str, tuple[dict[str, str], ...]]:
    result = {
        "tools": tuple(
            _named_digest(name, _sha256(path.resolve()))
            for name, path in sorted(tools)
        ),
        "sources": tuple(
            {"path": path, "sha256": _sha256(_ROOT / path)}
            for path in sorted(sources)
        ),
        "inputs": tuple(
            _named_digest(name, digest) for name, digest in sorted(inputs)
        ),
    }
    if not all(result.values()):
        raise RuntimeError("negative witness bindings must be non-empty")
    return result


def _expect_frontend_error(
    *,
    key: str,
    validator: str,
    expected_type: type[FrontendError],
    expected_message: str,
    mutation: Callable[[], None],
    bindings: dict[str, tuple[dict[str, str], ...]],
) -> dict[str, object]:
    try:
        mutation()
    except FrontendError as error:
        observed = str(error)
        if type(error) is not expected_type or observed != expected_message:
            raise RuntimeError(
                f"{key}: expected {expected_type.__name__} "
                f"{expected_message!r}, observed {type(error).__name__} "
                f"{observed!r}"
            ) from error
        return {
            "bindings": bindings,
            "expected_error": expected_type.__name__,
            "expected_message": expected_message,
            "key": key,
            "observed_error": observed,
            "passed": True,
            "validator": validator,
        }
    except Exception as error:  # pragma: no cover - fail-closed diagnostic
        raise RuntimeError(
            f"{key}: expected {expected_type.__name__}, observed "
            f"{type(error).__name__}: {error}"
        ) from error
    raise RuntimeError(f"{key}: mutation was incorrectly accepted")


def _greedy_witness(tp_degree: int) -> dict[str, object]:
    template = build_ir0(_spec(tp_degree, InferOutput.GREEDY_SAMPLE))
    return _expect_frontend_error(
        key=f"tp{tp_degree}_greedy_unsupported",
        validator="logical_expand",
        expected_type=UnsupportedFeatureError,
        expected_message=(
            "unsupported_feature at template.infer_output: "
            "TP-sharded greedy sampling is not implemented"
        ),
        mutation=lambda: logical_expand(template),
        bindings=_bindings(
            tools=(("python", Path(sys.executable)),),
            sources=(
                "llm/frontend/wafer_frontend/passes/logical_expand.py",
                "llm/test/frontend/integration/stage2_dense_forward_cases.py",
            ),
            inputs=(("ir0_template", canonical_digest(template)),),
        ),
    )


def _ce_graph(case) -> IR1:
    graph = case.graph
    lm_head = next(node for node in graph.nodes if node.id == "P0.lm_head")
    logits = next(value for value in graph.values if value.id == "P0.logits")
    token_ids = next(
        value for value in graph.values if value.id == "P0.token_ids"
    )
    ce_id = "P0.ce_forward"
    labels = replace(
        token_ids,
        id="P0.labels",
        logical_layout="M_labels",
        consumers=(ce_id,),
    )
    losses = replace(
        token_ids,
        id="P0.losses",
        dtype=DType.FP32,
        logical_layout="M_losses",
        producer=ce_id,
        consumers=(),
    )
    ce = replace(
        lm_head,
        id=ce_id,
        origin_node_id=ce_id,
        kind=OpKind.CE_FORWARD,
        inputs=(logits.id, labels.id),
        outputs=(losses.id,),
        workload=CrossEntropyForwardWorkload(
            profile=graph.profile,
            reduction=CrossEntropyReduction.NONE,
            logical_logits_shape=(8, 32),
            rank_logits_shape=(8, 32),
            logical_label_shape=(8,),
            rank_label_shape=(8,),
            logical_loss_shape=(8,),
            rank_loss_shape=(8,),
            logits_dtype=DType.FP16,
            label_dtype=DType.INT32,
            loss_dtype=DType.FP32,
        ),
        impl_ref="cross_entropy_forward",
    )
    fields = graph._semantic_key()
    fields.update(
        instances=(
            replace(
                graph.instances[0],
                node_ids=graph.instances[0].node_ids + (ce.id,),
            ),
        ),
        nodes=graph.nodes + (ce,),
        values=tuple(
            replace(value, consumers=(ce.id,))
            if value.id == logits.id
            else value
            for value in graph.values
        )
        + (labels, losses),
        edges=graph.edges
        + (
            GraphEdge(
                "P0.edge.lm_head.ce",
                EdgeKind.DATA,
                lm_head.id,
                ce.id,
                logits.id,
            ),
        ),
    )
    result = IR1.create(producer_pass=graph.producer_pass, **fields)
    result.validate()
    return result


def _ce_witness(case) -> dict[str, object]:
    graph = _ce_graph(case)
    node_index = next(
        index for index, node in enumerate(graph.nodes) if node.kind is OpKind.CE_FORWARD
    )
    return _expect_frontend_error(
        key="ce_projection_unsupported",
        validator="NaiveProjectToIR2.run",
        expected_type=UnsupportedFeatureError,
        expected_message=(
            f"unsupported_feature at ir1.nodes[{node_index}].kind: "
            "D2-3 Action/IR2 carrier support is not implemented for "
            "Stage2 op kind 'ce_forward'"
        ),
        mutation=lambda: NaiveProjectToIR2().run(
            graph,
            case.global_profile.fusion_plans,
            case.global_profile.standalone_plans,
            state_transfers=(),
        ),
        bindings=_bindings(
            tools=(("python", Path(sys.executable)),),
            sources=(
                "llm/frontend/wafer_frontend/policies/naive_project_to_ir2.py",
                "llm/test/frontend/integration/stage2_dense_forward_cases.py",
            ),
            inputs=(("ir1_with_ce", canonical_digest(graph)),),
        ),
    )


def _valid_runtime_report(case) -> Stage2DenseForwardRuntimeReport:
    golden = _CASE_GOLDENS[1]
    artifact = golden["artifact"]
    memory = golden["memory"]
    probes = (
        Stage2DenseForwardProbeEvidence(
            "probe.0",
            ProgramIoTargetKind.SRAM,
            512,
            _DIGEST,
            _DIGEST,
            True,
            True,
            True,
        ),
    )
    control = Stage2DenseForwardControlEvidence(
        tuple(
            Stage2DenseForwardCoreCount(item.runtime_core_id, 2)
            for item in memory
        ),
        tuple(
            Stage2DenseForwardCoreCount(item.runtime_core_id, 1)
            for item in memory
        ),
        tuple(
            Stage2DenseForwardNamedCount(name, 0)
            for name in ("collective", "global", "p2p", "timing")
        ),
        True,
    )
    repeat_key = {
        "makespan_cycles": golden["makespan"],
        "marker_digest": _DIGEST,
        "memory_digest": canonical_digest(memory),
        "probe_digest": canonical_digest(probes),
        "control_digest": canonical_digest(control),
        "d2d_digest": canonical_digest(golden["d2d"]),
    }
    python_digest = _sha256(Path(sys.executable).resolve())
    hbm_initializations = sum(
        type(item.target) is ProgramHbmTarget
        for item in case.program_io.initializations
    )
    sram_initializations = sum(
        type(item.target) is ProgramSramTarget
        for item in case.program_io.initializations
    )
    report = Stage2DenseForwardRuntimeReport.create(
        baseline_epoch=STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
        tp_degree=1,
        infer_output=InferOutput.LOGITS,
        capability_status=CapabilityStatus.E2E_TIMING,
        oracle_id=case.oracle.id,
        oracle_digest=canonical_digest(case.oracle),
        compile=Stage2DenseForwardCompileEvidence(
            case.template.id,
            canonical_digest(case.template),
            case.graph.id,
            canonical_digest(case.graph),
            case.global_dag.id,
            canonical_digest(case.global_dag),
            case.lowered.id,
            canonical_digest(case.lowered),
        ),
        tools=Stage2DenseForwardToolEvidence(
            python_digest, python_digest, python_digest
        ),
        hardware_digest=hashlib.sha256(
            case.runtime_hardware_inputs.hardware_json.encode("utf-8")
        ).hexdigest(),
        simulation_digest=_DIGEST,
        mapping_digest=hashlib.sha256(
            case.runtime_hardware_inputs.mapping_text.encode("utf-8")
        ).hexdigest(),
        artifact=Stage2DenseForwardArtifactEvidence(
            case.manifest.id,
            canonical_digest(case.manifest),
            artifact[6],
            artifact[0],
            artifact[1],
            artifact[2],
            artifact[3],
            artifact[4],
            artifact[5],
            golden["opcodes"],
        ),
        sidecar=Stage2DenseForwardSidecarEvidence(
            case.program_io.id,
            canonical_digest(case.program_io),
            ProgramIoMode.TIMING,
            hbm_initializations,
            sram_initializations,
            0,
            len(case.program_io.output_probes),
        ),
        memory=memory,
        probes=probes,
        control=control,
        d2d=golden["d2d"],
        marker_schema_version=STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION,
        repeat_count=2,
        makespan_cycles=golden["makespan"],
        repeats=tuple(
            Stage2DenseForwardRepeatEvidence(index, **repeat_key)
            for index in range(2)
        ),
        timing_execution=True,
        dense_forward_structure_exact=True,
        analytic_work_exact=True,
        program_io_boundary_exact=True,
        traffic_accounting_exact=True,
        compute_functional=False,
        model_functional=False,
    )
    report.validate_against(case.oracle)
    return report


def _report_witnesses(case) -> tuple[dict[str, object], ...]:
    report = _valid_runtime_report(case)
    semantic = report._semantic_key()
    python = (("python", Path(sys.executable)),)
    sources = (
        "llm/frontend/wafer_frontend/schema/stage2_dense_forward_evidence.py",
    )

    def witness(
        key: str,
        message: str,
        changes: dict[str, object],
    ) -> dict[str, object]:
        mutated = semantic | changes
        return _expect_frontend_error(
            key=key,
            validator="Stage2DenseForwardRuntimeReport.create",
            expected_type=SchemaError,
            expected_message=message,
            mutation=lambda: Stage2DenseForwardRuntimeReport.create(**mutated),
            bindings=_bindings(
                tools=python,
                sources=sources,
                inputs=(("mutated_runtime_semantic_key", canonical_digest(mutated)),),
            ),
        )

    artifact = replace(
        report.artifact,
        program_artifact_sha256="2" * 64,
    )
    second_repeat = replace(report.repeats[1], marker_digest="2" * 64)
    d2d = replace(
        report.d2d,
        physical_packet_count=1,
        byte_hop_bytes=16,
    )
    return (
        witness(
            "runtime_artifact_sha_tamper",
            "schema_error at stage2_dense_forward_runtime_report.artifact: "
            "artifact evidence disagrees with the frozen TP case",
            {"artifact": artifact},
        ),
        witness(
            "runtime_makespan_tamper",
            "schema_error at stage2_dense_forward_runtime_report.repeats: "
            "reviewed runtime evidence requires two exact repeats",
            {"makespan_cycles": report.makespan_cycles + 1},
        ),
        witness(
            "runtime_repeat_marker_mismatch",
            "schema_error at stage2_dense_forward_runtime_report.repeats[1].marker_digest: "
            "runtime markers are not deterministic",
            {"repeats": (report.repeats[0], second_repeat)},
        ),
        witness(
            "runtime_d2d_packet_tamper",
            "schema_error at stage2_dense_forward_runtime_report.d2d: "
            "link packets must close exact D2D totals",
            {"d2d": d2d},
        ),
        witness(
            "runtime_functional_overclaim",
            "schema_error at stage2_dense_forward_runtime_report: "
            "proof is exact dense-forward timing/accounting only",
            {"compute_functional": True},
        ),
    )


def _run_command(
    command: list[str], *, cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def _resolver_witness(args: argparse.Namespace, case) -> dict[str, object]:
    with tempfile.TemporaryDirectory(
        prefix="stage2-negative-resolver-", dir=args.runtime_root
    ) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        artifact_path = directory / "program.npup"
        finalization_path = directory / "finalization.json"
        sidecar_path = directory / "program_io.json"
        manifest_path.write_text(canonical_json(case.manifest), encoding="utf-8")
        finalized = _run_command(
            [
                str(args.finalizer),
                "--input",
                str(manifest_path),
                "--output",
                str(artifact_path),
                "--report",
                str(finalization_path),
            ],
            cwd=args.runtime_root,
            timeout=60,
        )
        if finalized.returncode != 0:
            raise RuntimeError(
                f"resolver_artifact_sha_mismatch: finalizer failed: {finalized.stdout}"
            )
        artifact_sha256 = _sha256(artifact_path)
        finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
        if (
            finalization.get("artifact_sha256") != artifact_sha256
            or finalization.get("linked_manifest_id") != case.manifest.id
            or finalization.get("linked_manifest_digest")
            != canonical_digest(case.manifest)
        ):
            raise RuntimeError(
                "resolver_artifact_sha_mismatch: finalizer closure changed"
            )
        if case.program_io.program_artifact_sha256 == artifact_sha256:
            raise RuntimeError(
                "resolver_artifact_sha_mismatch: sidecar unexpectedly matches artifact"
            )
        case.program_io.validate_against(case.manifest)
        sidecar_path.write_text(canonical_json(case.program_io), encoding="utf-8")
        resolved = _run_command(
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(artifact_path),
                str(sidecar_path),
            ],
            cwd=args.runtime_root,
            timeout=60,
        )
        expected = (
            "ProgramIo C++ selftest failed: "
            "program_io_contract.program_artifact_sha256: "
            "does not match the actual encoded ProgramArtifact bytes"
        )
        error_lines = tuple(
            line.strip()
            for line in resolved.stdout.splitlines()
            if line.strip().startswith("ProgramIo C++ selftest failed:")
        )
        if (
            resolved.returncode == 0
            or error_lines != (expected,)
            or "ProgramIo resolved id=" in resolved.stdout
        ):
            raise RuntimeError(
                "resolver_artifact_sha_mismatch: expected exact failure "
                f"{expected!r}, observed exit={resolved.returncode} "
                f"{resolved.stdout!r}"
            )
        return {
            "bindings": _bindings(
                tools=(
                    ("finalizer", args.finalizer),
                    ("resolver", args.resolver),
                ),
                sources=(
                    "llm/src/frontend/program_io.cpp",
                    "llm/unittest/program_io_selftest_main.cpp",
                ),
                inputs=(
                    ("linked_manifest", canonical_digest(case.manifest)),
                    ("program_artifact", artifact_sha256),
                    ("program_io", canonical_digest(case.program_io)),
                ),
            ),
            "exit_code": resolved.returncode,
            "expected_error": "resolver_failure",
            "expected_message": expected,
            "key": "resolver_artifact_sha_mismatch",
            "observed_error": error_lines[0],
            "passed": True,
            "raw_output_sha256": hashlib.sha256(
                resolved.stdout.encode("utf-8")
            ).hexdigest(),
            "validator": "npusim_program_io_selftest --resolve",
        }


def _validate_evidence(evidence: dict[str, object]) -> None:
    if evidence.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError("negative evidence schema version changed")
    witnesses = evidence.get("witnesses")
    if type(witnesses) is not tuple or len(witnesses) != 9:
        raise RuntimeError("negative evidence must contain exactly nine witnesses")
    expected_keys = (
        "ce_projection_unsupported",
        "resolver_artifact_sha_mismatch",
        "runtime_artifact_sha_tamper",
        "runtime_d2d_packet_tamper",
        "runtime_functional_overclaim",
        "runtime_makespan_tamper",
        "runtime_repeat_marker_mismatch",
        "tp2_greedy_unsupported",
        "tp4_greedy_unsupported",
    )
    if tuple(item.get("key") for item in witnesses) != expected_keys:
        raise RuntimeError("negative witness key set/order changed")
    for witness in witnesses:
        bindings = witness.get("bindings")
        if (
            witness.get("passed") is not True
            or type(bindings) is not dict
            or any(not bindings.get(name) for name in ("tools", "sources", "inputs"))
        ):
            raise RuntimeError(f"negative witness is incomplete: {witness}")
        for group in bindings.values():
            for item in group:
                digest = item.get("sha256")
                if (
                    type(digest) is not str
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise RuntimeError(f"negative witness digest is invalid: {item}")


def build_negative_evidence(args: argparse.Namespace) -> dict[str, object]:
    case = build_stage2_dense_forward_case(1)
    witnesses = tuple(
        sorted(
            (
                _greedy_witness(2),
                _greedy_witness(4),
                _ce_witness(case),
                _resolver_witness(args, case),
                *_report_witnesses(case),
            ),
            key=lambda item: str(item["key"]),
        )
    )
    source_digests = tuple(
        {"path": path, "sha256": _sha256(_ROOT / path)}
        for path in _SOURCE_PATHS
    )
    semantic_key = {
        "baseline_epoch": STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
        "command": (
            "python3 -B llm/test/frontend/integration/"
            "run_stage2_dense_forward_negative_evidence.py "
            "--finalizer <path> --resolver <path> --runtime-root <path> "
            "[--output <path>]"
        ),
        "source_digests": source_digests,
        "witnesses": witnesses,
    }
    evidence = {
        "schema_version": _SCHEMA_VERSION,
        "producer_pass": _PRODUCER_PASS,
        "id": stable_artifact_id(
            "stage2_dense_forward_negative_evidence",
            semantic_key,
            schema_version=_SCHEMA_VERSION,
        ),
        **semantic_key,
    }
    _validate_evidence(evidence)
    return evidence


def _write_new(path: Path, evidence: dict[str, object]) -> None:
    if path.suffix != ".json":
        raise RuntimeError("checked negative evidence output must be JSON, not NPUP")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite/symlink checked evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(evidence) + "\n")


def _executable(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalizer", required=True, type=_executable)
    parser.add_argument("--resolver", required=True, type=_executable)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        args.output = args.output.absolute()
        if args.output.suffix != ".json":
            raise RuntimeError(
                "checked negative evidence output must be JSON, not NPUP"
            )
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError(
                f"refusing to overwrite/symlink checked evidence: {args.output}"
            )
    first = build_negative_evidence(args)
    second = build_negative_evidence(args)
    first_bytes = (canonical_json(first) + "\n").encode("utf-8")
    second_bytes = (canonical_json(second) + "\n").encode("utf-8")
    if first_bytes != second_bytes:
        raise RuntimeError("negative evidence repeats are not byte-identical")
    if args.output is not None:
        _write_new(args.output, first)
    print(
        "[STAGE2 NEGATIVE] PASS: "
        f"witnesses={len(first['witnesses'])} deterministic=1 id={first['id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
