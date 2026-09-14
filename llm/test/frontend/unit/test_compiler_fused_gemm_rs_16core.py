from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend import compile_naive
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.intra_die_refine import (
    SplitKRefineOptions,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.ir1 import SramAllocator
from llm.frontend.wafer_frontend.schema.ir2 import SwizzleNodeOrigin
from llm.frontend.wafer_frontend.schema.serde import from_data

from _fixtures import valid_spec
from test_n6_pipeline import _e1_compile_inputs


SRAM_BYTES_PER_CORE = 3 * 1024 * 1024
CORES_PER_DIE = 16


def _fused_gemm_rs_spec() -> ExperimentSpec:
    """Small TP2 GEMM_RS whose rank-local K is divisible by 16."""

    raw = valid_spec()
    model = raw["model"]
    assert isinstance(model, dict)
    model.update(
        {
            "V": 128,
            "H": 32,
            "I": 64,
            "NH": 4,
            "KVH": 2,
            "DH": 8,
            "rotary_dim": 8,
            "L": 1,
        }
    )
    infer = raw["workload"]["infer"]  # type: ignore[index]
    assert isinstance(infer, dict)
    profile = infer["profile"]
    assert isinstance(profile, dict)
    profile.update(
        {
            "prefill_tokens": 2,
            "context_sum": 2,
            "context_max": 2,
        }
    )
    policy = raw["policy"]
    assert isinstance(policy, dict)
    policy["inter_die"] = "swizzle_topo"
    policy["intra_die"] = "optimized"
    return from_data(ExperimentSpec, raw, path="spec")


def _three_mib_16core_compile_inputs():
    """Use the production 2-die x 16-core topology with 3 MiB/core."""

    fabric, hbm_address_spaces = _e1_compile_inputs()
    profiles = []
    for profile in fabric.sram_profiles:
        profiles.append(
            replace(
                profile,
                capacity_bytes=SRAM_BYTES_PER_CORE,
                regions=tuple(
                    replace(
                        region,
                        base_bytes=0,
                        size_bytes=SRAM_BYTES_PER_CORE,
                        allocator=SramAllocator.BLOCK,
                    )
                    for region in profile.regions
                ),
            )
        )
    result = replace(fabric, sram_profiles=tuple(profiles))
    result.validate("fused_gemm_rs_16core.fabric")
    for die in result.dies:
        assert len(die.cores) == CORES_PER_DIE
    return result, hbm_address_spaces


def compile_fused_gemm_rs_16core():
    """Compile through the production common-IR2 and manifest-link chain."""

    spec = _fused_gemm_rs_spec()
    fabric, hbm_address_spaces = _three_mib_16core_compile_inputs()
    options = SplitKRefineOptions(
        split_k_parts=CORES_PER_DIE,
        enable_reduce=True,
        compute_groups_per_die=CORES_PER_DIE,
        enable_tree_reduce=True,
        enable_direct_dma=True,
    )
    return compile_naive(
        spec,
        fabric,
        hbm_address_spaces=hbm_address_spaces,
        producer_pass="fused_gemm_rs_16core_fixture",
        intra_die_refine_options=options,
    )


class CompilerFusedGemmRs16CoreTest(unittest.TestCase):
    def test_fused_gemm_rs_uses_all_16_cores_and_fits_sram(self) -> None:
        compilation = compile_fused_gemm_rs_16core()

        planned = compilation.artifacts[5]
        projected = compilation.artifacts[6]
        refined = compilation.artifacts[7]
        scheduled = compilation.artifacts[8]
        gemm_rs_plan_ids = {
            plan.id
            for entry in planned.entries
            for plan in entry.fusion_plans
            if getattr(plan, "pattern", None) is FusionPattern.GEMM_RS
        }
        self.assertTrue(gemm_rs_plan_ids)

        fused_source_tasks = {
            task.id: task
            for entry in projected.entries
            for dag in entry.projection.dags
            for task in dag.tasks
            if isinstance(task.origin_ref, SwizzleNodeOrigin)
            and task.origin_ref.plan_id in gemm_rs_plan_ids
        }
        self.assertTrue(fused_source_tasks)
        rewrites = tuple(
            rewrite
            for entry in refined.entries
            if entry.split_k_refinement is not None
            for rewrite in entry.split_k_refinement.rewrites
            if rewrite.source_task_id in fused_source_tasks
        )
        self.assertTrue(rewrites, "no Swizzle GEMM_RS COMP task was split across cores")
        self.assertTrue(
            all(
                rewrite.compute_group_count == CORES_PER_DIE
                and set(rewrite.part_compute_groups) == set(range(CORES_PER_DIE))
                for rewrite in rewrites
            )
        )

        peak_end_by_core: dict[tuple[int, int], int] = defaultdict(int)
        active_cores_by_die: dict[int, set[int]] = defaultdict(set)
        expected_runtime_cores_by_die = {
            die.id: {core.runtime_core_id for core in die.cores}
            for die in compilation.fabric.dies
        }
        for entry in scheduled.entries:
            for schedule in entry.schedule_set.schedules:
                for order in schedule.core_orders:
                    if order.task_ids:
                        active_cores_by_die[schedule.die_id].add(order.core_id)
                for binding in schedule.buffer_bindings:
                    key = (schedule.die_id, binding.core_id)
                    peak_end_by_core[key] = max(
                        peak_end_by_core[key],
                        binding.region_offset_bytes + binding.size_bytes,
                    )
        self.assertTrue(active_cores_by_die)
        self.assertTrue(
            all(
                cores == expected_runtime_cores_by_die[die_id]
                for die_id, cores in active_cores_by_die.items()
            )
        )
        self.assertTrue(peak_end_by_core)
        self.assertLessEqual(max(peak_end_by_core.values()), SRAM_BYTES_PER_CORE)

        for entry in compilation.linked.entries:
            streams_by_die: dict[int, set[int]] = defaultdict(set)
            for stream in entry.manifest.core_streams:
                streams_by_die[stream.logical_core.die_id].add(
                    stream.logical_core.local_core_id
                )
            self.assertTrue(streams_by_die)
            self.assertTrue(
                all(
                    cores == set(range(CORES_PER_DIE))
                    for cores in streams_by_die.values()
                )
            )
            binding_keys = {
                (
                    binding.logical_core.die_id,
                    binding.logical_core.local_core_id,
                    binding.runtime_core_id,
                )
                for binding in entry.manifest.core_bindings
            }
            self.assertEqual(
                len(binding_keys),
                CORES_PER_DIE * len(streams_by_die),
            )


if __name__ == "__main__":
    unittest.main()
