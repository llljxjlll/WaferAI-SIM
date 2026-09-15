"""Run-local tools and actual imported sources for physical TP4 SGD."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


_ROOT = Path(__file__).resolve().parents[4]
_DRAM = _ROOT / "DRAMSys/configs"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_files() -> dict[str, str]:
    root = (_ROOT / "llm").resolve()
    source = {
        str(path.resolve()): _sha(path)
        for module in tuple(sys.modules.values())
        for value in (getattr(module, "__file__", None),)
        if value is not None
        for path in (Path(value),)
        if path.suffix == ".py" and path.resolve().is_relative_to(root)
    }
    source[str(Path(__file__).resolve())] = _sha(Path(__file__).resolve())
    return dict(sorted(source.items()))


def _verify_sources(sources: dict[str, str]) -> None:
    drift = [path for path, digest in sources.items()
             if not Path(path).is_file() or _sha(Path(path)) != digest]
    if drift:
        raise RuntimeError(f"TP4 imported source bytes drifted: {drift[:3]}")


def _stage(command: tuple[str, ...], cwd: Path, stdout: Path,
           *, timeout: int) -> dict[str, object]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout.write_text(str(error.stdout or "") + "\nTIMEOUT\n")
        receipt = {"wall_seconds": round(time.monotonic() - started, 3),
                   "exit_code": None, "reason": f"stage exceeded {timeout}s",
                   "command": command}
        stdout.with_suffix(".stage.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(receipt["reason"]) from error
    stdout.write_text(result.stdout, encoding="utf-8")
    receipt = {
        "wall_seconds": round(time.monotonic() - started, 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "exit_code": result.returncode, "command": command,
    }
    stdout.with_suffix(".stage.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    if result.returncode:
        raise RuntimeError(
            f"stage exit={result.returncode} stdout={stdout}: "
            + result.stdout[-1200:]
        )
    return receipt


__all__ = ["_DRAM", "_ROOT", "_sha", "_source_files", "_stage", "_verify_sources"]
