"""Run the strict 1x1 Dense Prefill -> Decode -> Decode sequence canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    compile_dense_e2e_sequence_runtime_profiles,
)
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_dense_compile_sequence import _one_die_case

from .flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)


_ROOT = Path(__file__).resolve().parents[4]


def _run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"returncode={completed.returncode}: {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def run(args: argparse.Namespace) -> None:
    manifest, template, fabric = _one_die_case()
    sequence, linked_profiles = compile_dense_e2e_sequence_runtime_profiles(
        manifest,
        template,
        fabric,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    sequence.validate()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []
    programs: list[Path] = []
    sidecars: list[Path] = []
    artifact_digests: list[str] = []
    for index, segment in enumerate(sequence.segments):
        manifest_path = output / f"segment_{index}.linked.json"
        artifact_path = output / f"segment_{index}.npup"
        report_path = output / f"segment_{index}.finalizer.json"
        manifest_path.write_text(
            canonical_json(segment.linked_manifest), encoding="utf-8"
        )
        _run(
            (
                str(args.finalizer.resolve()),
                "--input",
                str(manifest_path),
                "--output",
                str(artifact_path),
                "--report",
                str(report_path),
            ),
            cwd=output,
            timeout=120,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if (
            report.get("artifact_sha256") != artifact_digest
            or report.get("linked_manifest_id") != segment.linked_manifest.id
            or report.get("linked_manifest_digest")
            != canonical_digest(segment.linked_manifest)
        ):
            raise RuntimeError(f"segment {index} finalizer closure failed")
        manifests.append(manifest_path)
        programs.append(artifact_path)
        artifact_digests.append(artifact_digest)
        abi_by_binding = {
            abi.hbm_binding_ref: abi
            for fragment in linked_profiles[index].manifest.fragments
            for abi in fragment.state_abi
        }
        first_access: dict[str, StateUseAccess] = {}
        for action in linked_profiles[index].lowering_context.global_dag.actions:
            for use in action.state_uses:
                first_access.setdefault(use.hbm_binding_ref, use.access)
        state_seeds = {
            abi_by_binding[binding].state_ref: bytes(
                abi_by_binding[binding].size_bytes
            )
            for binding, access in first_access.items()
            if access is StateUseAccess.READ
        }
        contract = build_timing_program_io(
            linked_profiles[index],
            artifact_digest,
            state_seed_overrides=state_seeds,
        )
        contract.validate_against(segment.linked_manifest)
        sidecar_path = output / f"segment_{index}.program_io.json"
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        sidecars.append(sidecar_path)

    hardware_path = output / "hardware.json"
    mapping_path = output / "mapping.spec"
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 65536
    hardware["memory"]["sram"]["capacity_bytes"] = 65536
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 65536
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    mapping_path.write_text("0:0\n", encoding="utf-8")
    runtime_output = _run(
        (
            str(args.npusim.resolve()),
            "--program-sequence",
            ",".join(str(path) for path in programs),
            "--linked-manifest-sequence",
            ",".join(str(path) for path in manifests),
            "--program-io-sequence",
            ",".join(str(path) for path in sidecars),
            "--hardware-config",
            str(hardware_path),
            "--simulation-config",
            str(args.simulation.resolve()),
            "--mapping-config",
            str(mapping_path),
            "--trace-window",
            "1000000",
        ),
        cwd=output,
        timeout=args.timeout,
    )
    (output / "npusim.stdout.txt").write_text(runtime_output, encoding="utf-8")

    segment_markers = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)",
        runtime_output,
    )
    kv_markers = re.findall(
        r"\[DENSE_SEQUENCE_KV\] index=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) pass=1",
        runtime_output,
    )
    drain_markers = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",
        runtime_output,
    )
    if segment_markers != [("0", "0"), ("1", "0"), ("2", "1")]:
        raise RuntimeError(f"segment marker closure failed: {segment_markers}")
    if [tuple(item[:2]) for item in kv_markers] != [
        ("0", "512"),
        ("1", "640"),
        ("2", "768"),
    ]:
        raise RuntimeError(f"KV boundary closure failed: {kv_markers}")
    if drain_markers != [("3", "1")]:
        raise RuntimeError(f"one-shot drain closure failed: {drain_markers}")
    if runtime_output.count("[SIM_RESULT]") != 1:
        raise RuntimeError("runtime did not emit exactly one SIM_RESULT")
    print(
        "Dense sequence runtime canary PASS "
        f"sequence={sequence.digest} artifacts={','.join(artifact_digests)} "
        f"kv={','.join(item[2] for item in kv_markers)}"
    )


def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-debug-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=build / "dense-sequence-runtime-canary"
    )
    parser.add_argument(
        "--finalizer", type=Path, default=build / "npusim_program_finalizer"
    )
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument(
        "--simulation",
        type=Path,
        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    for name in ("finalizer", "npusim", "simulation"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name} must name an existing file")
    return args


if __name__ == "__main__":
    run(_parse_args())
