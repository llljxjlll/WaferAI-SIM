#!/usr/bin/env python3
"""Thin exp1-1 adapter around the existing Swizzle frontend/runtime.

This module intentionally owns no compiler or simulator logic.  It constructs
one ordinary frontend graph for an :class:`ExperimentCase`, asks the production
Swizzle policies to lower either Wang-1D or the unfused baseline, and invokes
the checked-in finalizer/resolver/npusim tool chain once.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Literal


_ROOT = Path(__file__).resolve().parents[3]
_INTEGRATION = _ROOT / "llm/test/frontend/integration"
for _path in (str(_ROOT), str(_INTEGRATION)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from llm.frontend.wafer_frontend import compile_naive
from llm.frontend.wafer_frontend.lowering.swizzle_unfused import (
    allocate_unfused_comparison_core_abi,
    build_unfused_comparison_operand_abi,
    lower_unfused_comparison_opcodes,
)
from llm.frontend.wafer_frontend.lowering.swizzle_unfused_standard import (
    link_unfused_comparison_program,
)
from llm.frontend.wafer_frontend.passes import (
    build_ir0,
    hbm_address_spaces_from_data,
    logical_expand,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.placement import place_ir0
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
from llm.frontend.wafer_frontend.passes.project_unfused_comparison import (
    build_unfused_comparison_plan,
    project_unfused_comparison,
)
from llm.frontend.wafer_frontend.policies.swizzle_topo import (
    SwizzleFusionPartition,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    InterDiePolicyName,
    IntraDiePolicyName,
)
from llm.frontend.wafer_frontend.schema.intra_die_refine import SplitKRefineOptions
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleAlgorithm
from llm.frontend.wafer_frontend.schema.swizzle_scale import SwizzleScalePoint

from flexible_mesh_release_hardware import (
    p5_large_hardware_template_json,
    specialize_release_hardware,
)
from swizzle_scale_cases import _planner, _spec


Branch = Literal["swizzle", "naive"]
_SIM_RESULT = re.compile(r"\[SIM_RESULT\]\s+makespan_cycles=(\d+)")
_DRAIN_KEYS = (
    "router_residual=0",
    "d2d_link_residual=0",
)


def _tools() -> tuple[Path, Path, Path]:
    """Return a mutually compatible finalizer, resolver and simulator."""
    for build_name in ("build-release-final", "build"):
        build = _ROOT / build_name
        paths = (
            build / "npusim_program_finalizer",
            build / "npusim_program_io_selftest",
            build / "npusim",
        )
        if all(path.is_file() for path in paths):
            return paths
    raise FileNotFoundError("npusim/finalizer/resolver binaries are not built")


def _run(command: list[str], *, cwd: Path, timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(
            f"command failed ({completed.returncode}): {Path(command[0]).name}\n{tail}"
        )
    return completed.stdout


def _point(case: object) -> SwizzleScalePoint:
    d = int(case.mesh.dies)
    align = lambda value, multiple=d: (
        (int(value) + multiple - 1) // multiple
    ) * multiple
    hidden_alignment = math.lcm(d, 16)
    tokens = align(case.logical_mnk[0])
    if case.operator == "AG_GEMM":
        # The scale fixture represents the AG output width as 2*intermediate.
        # Align model fields before TP partitioning so no logical element is
        # dropped by rank-local integer division.
        hidden = align(case.logical_mnk[2], hidden_alignment)
        intermediate = align((int(case.logical_mnk[1]) + 1) // 2)
    elif case.operator == "GEMM_RS":
        hidden = align(case.logical_mnk[1], hidden_alignment)
        intermediate = align(case.logical_mnk[2])
    else:
        raise ValueError(f"unsupported operator: {case.operator}")
    return SwizzleScalePoint.create(
        name=f"EXP_{case.case_id}",
        tokens=tokens,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tp=d,
        mesh_rows=int(case.mesh.rows),
        mesh_columns=int(case.mesh.columns),
        dtype=__import__(
            "llm.frontend.wafer_frontend.schema.common", fromlist=["DType"]
        ).DType.FP16,
    )


def _frontend(case: object):
    point = _point(case)
    spec = _spec(point)
    source = logical_expand(build_ir0(spec)).entries[0].graph
    if (int(case.mesh.rows), int(case.mesh.columns)) == (1, 2):
        hardware_json = (_ROOT / "notes/frontend/examples/hardware_2x1.json").read_text(encoding="utf-8")
    else:
        hardware_json = specialize_release_hardware(
            p5_large_hardware_template_json(),
            int(case.mesh.rows),
            int(case.mesh.columns),
        )
    hardware = json.loads(hardware_json)
    sram_bytes = 3 * (1 << 20)
    hardware["memory"]["sram_size"] = sram_bytes
    hardware["memory"]["sram"]["capacity_bytes"] = sram_bytes
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = sram_bytes
    for port in hardware["memory"]["sram"]["ports"].values():
        port["read"]["width_bits"] = 2048
        port["write"]["width_bits"] = 2048
    hardware["die_ports"]["c2c"]["link_bw"] = 1
    hardware["gpu"]["dram_bandwidth"] = 256
    for core in hardware.get("cores", []):
        core["exu_x"] = 64
        core["dram_bw"] = 256
        core["dte_channel_count"] = 2
    total_hbm_bytes = 64 * (1 << 30)
    hbm_interleave_bytes = int(
        hardware["memory_system"]["address_policy"]["channel_interleave_bytes"]
    )
    per_die_hbm_bytes = (
        total_hbm_bytes // int(case.mesh.dies) // hbm_interleave_bytes
    ) * hbm_interleave_bytes
    per_die_hbm_bytes = min(
        per_die_hbm_bytes,
        *(int(stack["capacity_bytes"]) for stack in hardware["memory_system"]["hbm_stacks"]),
    )
    for rank, stack in enumerate(hardware["memory_system"]["hbm_stacks"]):
        stack["capacity_bytes"] = per_die_hbm_bytes
        stack["compute_die_id"] = rank
        stack["bandwidth_cap_GBps"] = min(
            256.0, float(stack.get("bandwidth_cap_GBps", 256.0))
        )
    hardware["memory_system"]["address_policy"]["home_ranges"] = [
        {"die_id": rank, "base": rank * per_die_hbm_bytes,
         "size_bytes": per_die_hbm_bytes}
        for rank in range(int(case.mesh.dies))
    ]
    hardware["memory_system"]["address_policy"]["stack_interleave_bytes"] = per_die_hbm_bytes
    hardware_json = json.dumps(hardware, sort_keys=True, separators=(",", ":"))
    context = PlacementContext.create(
        producer_pass="exp1_1_runtime_adapter",
        fabric=physical_fabric_from_data(hardware, path="exp1_1.hardware"),
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(
            hardware, path="exp1_1.hardware"
        ),
    )
    placed = place_ir0(source, context)
    ir1 = partition_ir1(placed, policy=SwizzleFusionPartition())
    pattern = (
        FusionPattern.AG_GEMM
        if case.operator == "AG_GEMM"
        else FusionPattern.GEMM_RS
    )
    expected_shape = tuple(getattr(case, "runtime_mnk", case.logical_mnk))
    actual_shapes: list[tuple[int, int, int]] = []
    for skeleton in ir1.fused_op_skeletons:
        if skeleton.semantic_contract.pattern is not pattern:
            continue
        decision = _planner(point).decide(ir1, skeleton)
        actual = decision.problem.gemm
        actual_shape = (actual.m, actual.n, actual.k)
        actual_shapes.append(actual_shape)
        if actual_shape == expected_shape:
            return ir1, decision, hardware_json, spec, context
    raise RuntimeError(
        "frontend GEMM shape mismatch: "
        f"expected runtime {expected_shape}, got candidates {actual_shapes}"
    )


def _linked_source(
    ir1: object,
    decision: object,
    branch: Branch,
    *,
    spec: object,
    context: PlacementContext,
):
    if branch == "naive":
        plan = build_unfused_comparison_plan(ir1, decision.problem, decision.baseline)
        projection = project_unfused_comparison(ir1, plan)
        core_abi = allocate_unfused_comparison_core_abi(ir1, plan, projection)
        operand_abi = build_unfused_comparison_operand_abi(ir1, plan, projection)
        lowered = lower_unfused_comparison_opcodes(plan, projection)
        return link_unfused_comparison_program(
            ir1, plan, projection, lowered, core_abi, operand_abi
        ), decision.baseline.algorithm.value

    production_spec = replace(
        spec,
        policy=replace(
            spec.policy,
            inter_die=InterDiePolicyName.SWIZZLE_TOPO,
            intra_die=IntraDiePolicyName.OPTIMIZED,
        ),
    )
    compilation = compile_naive(
        production_spec,
        context.fabric,
        hbm_address_spaces=context.hbm_address_spaces,
        producer_pass="exp1_1_runtime_adapter",
        intra_die_refine_options=SplitKRefineOptions(
            split_k_parts=16,
            enable_reduce=True,
            compute_groups_per_die=16,
            enable_tree_reduce=False,
            # Fused outputs may feed another local task, so they are reduced
            # on-chip rather than forced through the terminal direct-DMA path.
            enable_direct_dma=False,
        ),
    )
    if len(compilation.linked.entries) != 1:
        raise RuntimeError(
            "unsupported: exp1-1 expects exactly one linked inference profile"
        )
    return (
        compilation.linked.entries[0],
        SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL.value,
    )


def run_branch(case: object, branch: Branch, timeout: int = 600) -> dict[str, object]:
    """Run one case/branch and always return a JSON-serializable result.

    Unsupported shapes and runtime failures are data, not exceptions, so a
    simple outer sweep can continue.  ``time_seconds`` is simulator cycles at
    the repository's 1 GHz clock converted to seconds.
    """
    started = time.monotonic()
    result: dict[str, object] = {
        "case_id": getattr(case, "case_id", "unknown"),
        "branch": branch,
        "status": "failed",
        "makespan_cycles": None,
        "time_seconds": None,
        "algorithm": None,
        "error": "",
    }
    try:
        if branch not in ("swizzle", "naive"):
            raise ValueError("branch must be 'swizzle' or 'naive'")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        ir1, decision, hardware_json, spec, context = _frontend(case)
        source, algorithm = _linked_source(
            ir1, decision, branch, spec=spec, context=context
        )
        if branch == "naive":
            source.validate_against()
        finalizer, resolver, npusim = _tools()
        tool_root = finalizer.parent
        simulation = _ROOT / "llm/test/sram/simulation.json"
        mapping = _ROOT / "llm/test/default/mapping.spec"

        with tempfile.TemporaryDirectory(
            prefix=f"exp1_1_{case.case_id}_{branch}_", dir=tool_root
        ) as directory_text:
            directory = Path(directory_text)
            manifest = directory / "linked.json"
            artifact = directory / "program.npup"
            finalizer_report = directory / "finalizer.json"
            hardware = directory / "hardware.json"
            sidecar = directory / "program_io.json"
            manifest.write_text(canonical_json(source.manifest), encoding="utf-8")
            hardware.write_text(hardware_json, encoding="utf-8")
            _run(
                [
                    str(finalizer), "--input", str(manifest), "--output", str(artifact),
                    "--report", str(finalizer_report),
                ],
                cwd=tool_root,
                timeout=timeout,
            )
            artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            contract = build_timing_program_io(source, artifact_sha)
            sidecar.write_text(canonical_json(contract), encoding="utf-8")
            _run(
                [str(resolver), "--resolve", str(manifest), str(artifact), str(sidecar)],
                cwd=tool_root,
                timeout=timeout,
            )
            output = _run(
                [
                    str(npusim), "--program", str(artifact),
                    "--linked-manifest", str(manifest),
                    "--program-io", str(sidecar),
                    "--hardware-config", str(hardware),
                    "--simulation-config", str(simulation),
                    "--mapping-config", str(mapping), "--trace-window", "1000000",
                ],
                cwd=tool_root,
                timeout=timeout,
            )
        matches = _SIM_RESULT.findall(output)
        if len(matches) != 1:
            raise RuntimeError("npusim emitted no unique SIM_RESULT")
        missing_drains = [key for key in _DRAIN_KEYS if key not in output]
        if missing_drains:
            raise RuntimeError(f"npusim drain markers missing: {missing_drains}")
        cycles = int(matches[0])
        result.update(
            status="ok",
            makespan_cycles=cycles,
            time_seconds=cycles / 1e9,
            algorithm=algorithm,
        )
    except subprocess.TimeoutExpired as error:
        result.update(status="timeout", error=f"timeout: {error.cmd[0]}")
    except Exception as error:  # one bad case must not stop the 176-case sweep
        message = str(error)
        status = "unsupported" if any(token in message.lower() for token in ("unsupported", "exceeds the named sram region", "exceeds exact sram region", "must divide exactly by tp")) else "failed"
        result.update(status=status, error=message)
    result["wall_seconds"] = time.monotonic() - started
    return result


__all__ = ["run_branch"]
