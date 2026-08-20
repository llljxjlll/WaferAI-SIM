#!/usr/bin/env python3
"""Execute and record the reviewed Stage1a fail-closed mutations."""

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

from llm.frontend.wafer_frontend.errors import SchemaError  # noqa: E402
from llm.frontend.wafer_frontend.passes import (  # noqa: E402
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.common import (  # noqa: E402
    stable_artifact_id,
)
from llm.frontend.wafer_frontend.schema.ir2 import (  # noqa: E402
    IR2ProjectionResult,
    IntraDieDAG,
    SemanticTaskKind,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (  # noqa: E402
    HbmAddressSpace,
    HbmBinding,
    PersistentStateDecl,
    PersistentStateManifest,
)
from llm.frontend.wafer_frontend.schema.program_io import (  # noqa: E402
    ProgramIoContract,
    ProgramSramInitialization,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
)
from llm.frontend.wafer_frontend.schema.stage1a_evidence import (  # noqa: E402
    STAGE1A_BASELINE_EPOCH,
    Stage1aCase,
)
from stage1a_state_cases import (  # noqa: E402
    build_cross_action_kv_foundation_case,
    build_parameter_foundation_case,
    build_pd1_case,
)
from run_stage1a_state_cases import (  # noqa: E402
    _dramsys_hardware,
    _rebuild_sidecar,
    _runtime_hardware,
)


_SCHEMA_VERSION = "wafer_frontend.stage1a_negative_evidence/v1alpha1"
_PRODUCER_PASS = "stage1a_negative_runner"
_ARTIFACT_SHA256 = "0" * 64
_SOURCE_PATHS = (
    "llm/frontend/wafer_frontend/schema/persistent_state.py",
    "llm/frontend/wafer_frontend/schema/ir2.py",
    "llm/frontend/wafer_frontend/schema/program_io.py",
    "llm/frontend/wafer_frontend/passes/program_io.py",
    "llm/test/frontend/integration/stage1a_state_cases.py",
    "llm/test/frontend/integration/run_stage1a_negative_evidence.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expect_schema_error(
    *,
    key: str,
    validator: str,
    expected_message: str,
    mutation: Callable[[], None],
) -> dict[str, object]:
    try:
        mutation()
    except SchemaError as error:
        observed = str(error)
        if expected_message not in observed:
            raise RuntimeError(
                f"{key}: expected SchemaError containing "
                f"{expected_message!r}, observed {observed!r}"
            ) from error
        return {
            "expected_error": "SchemaError",
            "expected_message": expected_message,
            "key": key,
            "observed_error": observed,
            "passed": True,
            "validator": validator,
        }
    except Exception as error:  # pragma: no cover - fail-closed diagnostic
        raise RuntimeError(
            f"{key}: expected SchemaError, observed {type(error).__name__}: "
            f"{error}"
        ) from error
    raise RuntimeError(f"{key}: mutation was incorrectly accepted")


def _duplicate_identity() -> None:
    manifest = build_cross_action_kv_foundation_case().graph.persistent_state_manifest
    assert manifest is not None
    declaration = manifest.declarations[0]
    duplicate = PersistentStateDecl.create(
        identity=declaration.identity,
        shape=(1, *declaration.shape[1:]),
        dtype=declaration.dtype,
        layout=declaration.layout,
        lifetime=declaration.lifetime,
        access=declaration.access,
    )
    PersistentStateManifest.create(
        address_spaces=manifest.address_spaces,
        declarations=(declaration, duplicate),
        bindings=(),
    )


def _misaligned_binding() -> None:
    manifest = build_cross_action_kv_foundation_case().graph.persistent_state_manifest
    assert manifest is not None
    declaration = manifest.declarations[0]
    binding = next(
        item for item in manifest.bindings if item.state_ref == declaration.id
    )
    PersistentStateManifest.create(
        address_spaces=manifest.address_spaces,
        declarations=(declaration,),
        bindings=(
            HbmBinding.create(
                state_ref=declaration.id,
                die_id=binding.die_id,
                address=binding.address + 1,
                size_bytes=binding.size_bytes,
            ),
        ),
    )


def _wrong_home_range() -> None:
    manifest = build_cross_action_kv_foundation_case().graph.persistent_state_manifest
    assert manifest is not None
    declaration = manifest.declarations[0]
    binding = next(
        item for item in manifest.bindings if item.state_ref == declaration.id
    )
    space = next(
        item for item in manifest.address_spaces if item.die_id == binding.die_id
    )
    too_small = HbmAddressSpace.create(
        die_id=space.die_id,
        base_address=space.base_address,
        size_bytes=space.alignment_bytes,
        alignment_bytes=space.alignment_bytes,
    )
    PersistentStateManifest.create(
        address_spaces=(too_small,),
        declarations=(declaration,),
        bindings=(binding,),
    )


def _uint64_overflow() -> None:
    HbmAddressSpace.create(
        die_id=0,
        base_address=(1 << 64) - 32,
        size_bytes=64,
        alignment_bytes=32,
    )


def _read_only_writeback() -> None:
    case = build_parameter_foundation_case()
    build_timing_program_io(
        case.linked_profile,
        _ARTIFACT_SHA256,
        state_seed_overrides={case.state_ref: case.seed},
        state_expected_overrides={case.state_ref: case.seed},
    )


def _projection_with_tasks(case, dag, tasks) -> IR2ProjectionResult:
    dag_key = dag._semantic_key()
    dag_key["tasks"] = tuple(tasks)
    forged_dag = IntraDieDAG.create(
        producer_pass=dag.producer_pass,
        **dag_key,
    )
    projection_key = case.projection._semantic_key()
    projection_key["dags"] = tuple(
        forged_dag if item.id == dag.id else item
        for item in case.projection.dags
    )
    return IR2ProjectionResult.create(
        producer_pass=case.projection.producer_pass,
        **projection_key,
    )


def _missing_dma_consumer() -> None:
    case = build_parameter_foundation_case()
    dag = case.projection.dags[0]
    dma = next(
        task
        for task in dag.tasks
        if task.kind is SemanticTaskKind.DMA_IN and task.dma is not None
    )
    assert dma.dma is not None
    forged_dma = replace(dma, dma=replace(dma.dma, access_task_refs=()))
    forged = _projection_with_tasks(
        case,
        dag,
        (forged_dma if task.id == dma.id else task for task in dag.tasks),
    )
    forged.validate_against(case.graph, (), ())


def _missing_dma_dependency() -> None:
    case = build_parameter_foundation_case()
    dag = case.projection.dags[0]
    compute = next(
        task for task in dag.tasks if task.kind is SemanticTaskKind.COMP
    )
    forged_compute = replace(compute, deps=())
    forged = _projection_with_tasks(
        case,
        dag,
        (
            forged_compute if task.id == compute.id else task
            for task in dag.tasks
        ),
    )
    forged.validate_against(case.graph, (), ())


def _sram_out_of_range() -> None:
    case = build_parameter_foundation_case()
    contract = case.program_io
    entry = next(
        item
        for item in contract.initializations
        if type(item) is ProgramSramInitialization
    )
    forged_entry = ProgramSramInitialization.create(
        **{**entry._semantic_key(), "offset_bytes": entry.offset_bytes + 1}
    )
    forged = ProgramIoContract.create(
        producer_pass=contract.producer_pass,
        mode=contract.mode,
        source_manifest=case.manifest,
        program_artifact_sha256=contract.program_artifact_sha256,
        blobs=contract.blobs,
        initializations=tuple(
            forged_entry if item.id == entry.id else item
            for item in contract.initializations
        ),
        output_probes=contract.output_probes,
    )
    forged.validate_against(case.manifest)

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


def _dramsys_preflight_witness(args: argparse.Namespace) -> dict[str, object]:
    case = build_pd1_case()
    with tempfile.TemporaryDirectory(
        prefix="stage1a-negative-dramsys-", dir=args.runtime_root
    ) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        artifact_path = directory / "program.npup"
        report_path = directory / "finalization.json"
        sidecar_path = directory / "program_io.json"
        hardware_path = directory / "hardware.json"
        dramsys_path = directory / "hardware.dramsys.json"
        manifest_path.write_text(canonical_json(case.manifest), encoding="utf-8")
        finalized = _run_command(
            [
                str(args.finalizer),
                "--input",
                str(manifest_path),
                "--output",
                str(artifact_path),
                "--report",
                str(report_path),
            ],
            cwd=args.runtime_root,
            timeout=60,
        )
        if finalized.returncode != 0:
            raise RuntimeError(
                "dramsys_debug_peek_preflight: finalizer failed: "
                f"{finalized.stdout}"
            )
        finalization = json.loads(report_path.read_text(encoding="utf-8"))
        artifact_sha256 = _sha256(artifact_path)
        if (
            finalization.get("artifact_sha256") != artifact_sha256
            or finalization.get("linked_manifest_id") != case.manifest.id
            or finalization.get("linked_manifest_digest")
            != canonical_digest(case.manifest)
        ):
            raise RuntimeError(
                "dramsys_debug_peek_preflight: finalizer closure mismatch"
            )
        sidecar = _rebuild_sidecar(case, artifact_sha256)
        sidecar_path.write_text(canonical_json(sidecar), encoding="utf-8")
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
        if resolved.returncode != 0:
            raise RuntimeError(
                "dramsys_debug_peek_preflight: resolver failed: "
                f"{resolved.stdout}"
            )
        _runtime_hardware(
            Stage1aCase.PD1,
            case,
            args.hardware,
            hardware_path,
        )
        _dramsys_hardware(
            Stage1aCase.PD1,
            hardware_path,
            dramsys_path,
        )
        execution = _run_command(
            [
                str(args.npusim),
                "--program",
                str(artifact_path),
                "--linked-manifest",
                str(manifest_path),
                "--program-io",
                str(sidecar_path),
                "--hardware-config",
                str(dramsys_path),
                "--simulation-config",
                str(args.simulation),
                "--mapping-config",
                str(args.mapping),
                "--trace-window",
                "1000000",
            ],
            cwd=args.runtime_root,
            timeout=60,
        )
        required = "HBM backend does not support debug peeking"
        forbidden = ("[SIM_RESULT]", "[PROGRAM_MEMORY]")
        if (
            execution.returncode == 0
            or required not in execution.stdout
            or any(marker in execution.stdout for marker in forbidden)
        ):
            raise RuntimeError(
                "dramsys_debug_peek_preflight: rejection was late or "
                f"ambiguous: exit={execution.returncode} output={execution.stdout}"
            )
        observed = next(
            line.strip()
            for line in execution.stdout.splitlines()
            if required in line
        )
        return {
            "expected_error": "runtime_preflight_failure",
            "expected_message": required,
            "exit_code_nonzero": True,
            "forbidden_markers_absent": True,
            "key": "dramsys_debug_peek_preflight",
            "observed_error": observed,
            "passed": True,
            "validator": "npusim ProgramIo HBM apply preflight",
        }


def _external_inputs(args: argparse.Namespace) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "name": name,
            "sha256": _sha256(getattr(args, name)),
        }
        for name in (
            "finalizer",
            "hardware",
            "mapping",
            "npusim",
            "resolver",
            "simulation",
        )
    )


def build_negative_evidence(args: argparse.Namespace) -> dict[str, object]:
    witnesses = (
        _expect_schema_error(
            key="state_identity_duplicate",
            validator="PersistentStateManifest.create",
            expected_message="duplicate logical state identity",
            mutation=_duplicate_identity,
        ),
        _expect_schema_error(
            key="hbm_binding_misaligned",
            validator="PersistentStateManifest.create",
            expected_message="alignment",
            mutation=_misaligned_binding,
        ),
        _expect_schema_error(
            key="hbm_binding_wrong_home_range",
            validator="PersistentStateManifest.create",
            expected_message="home address space",
            mutation=_wrong_home_range,
        ),
        _expect_schema_error(
            key="hbm_address_uint64_overflow",
            validator="HbmAddressSpace.create",
            expected_message="uint64",
            mutation=_uint64_overflow,
        ),
        _expect_schema_error(
            key="read_only_state_writeback",
            validator="build_timing_program_io",
            expected_message="READ_WRITE",
            mutation=_read_only_writeback,
        ),
        _expect_schema_error(
            key="dma_consumer_refs_incomplete",
            validator="IR2ProjectionResult.validate_against",
            expected_message="must identify at least one state-access task",
            mutation=_missing_dma_consumer,
        ),
        _expect_schema_error(
            key="dma_completion_dependency_missing",
            validator="IR2ProjectionResult.validate_against",
            expected_message="consumer dependency closure omits a relevant slice producer",
            mutation=_missing_dma_dependency,
        ),
        _expect_schema_error(
            key="program_io_sram_out_of_range",
            validator="ProgramIoContract.validate_against",
            expected_message="range",
            mutation=_sram_out_of_range,
        ),
        _dramsys_preflight_witness(args),
    )
    source_digests = tuple(
        {
            "path": relative,
            "sha256": _sha256(_ROOT / relative),
        }
        for relative in _SOURCE_PATHS
    )
    semantic_key = {
        "baseline_epoch": STAGE1A_BASELINE_EPOCH,
        "external_inputs": _external_inputs(args),
        "command": (
            "python3 -B llm/test/frontend/integration/"
            "run_stage1a_negative_evidence.py --npusim <path> "
            "--finalizer <path> --resolver <path> --hardware <path> "
            "--simulation <path> --mapping <path> --runtime-root <path> "
            "--output <path>"
        ),
        "source_digests": source_digests,
        "witnesses": witnesses,
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "producer_pass": _PRODUCER_PASS,
        "id": stable_artifact_id(
            "stage1a_negative_evidence",
            semantic_key,
            schema_version=_SCHEMA_VERSION,
        ),
        **semantic_key,
    }


def _write_new(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite checked evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", required=True, type=Path)
    parser.add_argument("--finalizer", required=True, type=Path)
    parser.add_argument("--resolver", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name in (
        "npusim",
        "finalizer",
        "resolver",
        "hardware",
        "simulation",
        "mapping",
    ):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"--{name} is not a file: {path}")
        setattr(args, name, path)
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    evidence = build_negative_evidence(args)
    _write_new(args.output.resolve(), evidence)
    print(
        "[STAGE1A NEGATIVE] PASS: "
        f"witnesses={len(evidence['witnesses'])} id={evidence['id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
