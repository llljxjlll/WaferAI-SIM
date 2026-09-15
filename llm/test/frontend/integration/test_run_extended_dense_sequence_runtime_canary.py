"""Strict local tests and opt-in real TP16 integration executions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest

from .run_extended_dense_sequence_runtime_canary import (
    _KV_BYTES,
    _ROOT,
    bind_dram_resources,
    build_case,
    extended_hardware,
    observe_runtime,
    run,
    validate_physical_hbm,
)


_DIGEST = "a" * 64


def _valid_runtime_output() -> str:
    return "\n".join(
        [
            f"[DENSE_SEQUENCE_SEGMENT] index={step} status=done final={int(step == 2)}"
            for step in range(3)
        ]
        + [
            f"[DENSE_SEQUENCE_KV] index={step} bytes={size} digest={_DIGEST} pass=1"
            for step, size in enumerate(_KV_BYTES)
        ]
        + [
            "[DENSE_SEQUENCE_DRAIN] segments=3 one_shot=1",
            "[SIM_RESULT] makespan_cycles=12345",
        ]
        + [
            f"[P5 P2P DRAIN] core={rank * 4} residual=0"
            for rank in range(16)
        ]
        + [
            "[P5 P2P TIMING DRAIN] residual=0",
            "[DRAIN] router_residual=0",
            "[DRAIN] d2d_link_residual=0",
            "[CREDIT] data_balanced=1 ctrl_balanced=1",
        ]
    )


class ExtendedDenseCanaryContractTest(unittest.TestCase):
    def test_both_tp16_cases_use_all_dies_outside_release(self) -> None:
        for rows, columns in ((1, 16), (16, 1)):
            with self.subTest(shape=(rows, columns)):
                materialized, template, fabric, spaces = build_case(rows, columns)
                self.assertEqual(
                    materialized.placement.active_die_ids, tuple(range(16)),
                )
                self.assertEqual(template.parallel.instances[0].tp, 16)
                self.assertEqual(fabric.die_grid, (columns, rows))
                hardware = json.loads(extended_hardware(rows, columns, spaces))
                self.assertEqual(hardware["die"], {"x": columns, "y": rows})
                self.assertEqual(len(hardware["memory_system"]["hbm_stacks"]), 16)
                self.assertEqual(hardware["memory"]["sram_size"], 1 << 20)
                hbm_binding = validate_physical_hbm(
                    extended_hardware(rows, columns, spaces), spaces, rows, columns,
                )
                self.assertEqual(hbm_binding["hbm_bytes_per_die"], [1 << 20] * 16)
                self.assertEqual(hbm_binding["hbm_total_bytes"], 16 << 20)

    def test_physical_hbm_capacity_or_home_range_drift_is_rejected(self) -> None:
        materialized, template, fabric, spaces = build_case(1, 16)
        hardware = json.loads(extended_hardware(1, 16, spaces))
        hardware["memory_system"]["hbm_stacks"][3]["capacity_bytes"] += 64
        with self.assertRaisesRegex(RuntimeError, "physical HBM capacity"):
            validate_physical_hbm(json.dumps(hardware), spaces, 1, 16)
        hardware["memory_system"]["hbm_stacks"][3]["capacity_bytes"] -= 64
        hardware["memory_system"]["address_policy"]["home_ranges"][5]["size_bytes"] += 64
        with self.assertRaisesRegex(RuntimeError, "physical HBM capacity"):
            validate_physical_hbm(json.dumps(hardware), spaces, 1, 16)

    def test_behavioral_dram_resources_bind_monitor_and_all_channels(self) -> None:
        _, _, _, spaces = build_case(1, 16)
        hardware = extended_hardware(1, 16, spaces)
        simulation = _ROOT / "llm/test/program/p5_behavioral_simulation.json"
        binding = bind_dram_resources(
            hardware, simulation, _ROOT / "build-debug-final",
        )
        self.assertEqual(binding["channel_reference_count"], 17)
        self.assertEqual(
            binding["hbm2_config"]["path"],
            str((_ROOT / "DRAMSys/configs/hbm2-example.json").resolve()),
        )
        self.assertEqual(
            set(binding["dependencies"]),
            {"addressmapping", "mcconfig", "memspec", "simconfig"},
        )
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(RuntimeError, "behavioral DRAMSys resources"):
                bind_dram_resources(hardware, simulation, Path(raw) / "tools")
        mutated = json.loads(hardware)
        mutated["memory_system"]["hbm_stacks"][0]["channel_dram_config"] = (
            "../DRAMSys/configs/absent.json"
        )
        with self.assertRaisesRegex(RuntimeError, "monitor/channel DRAM config"):
            bind_dram_resources(
                json.dumps(mutated), simulation, _ROOT / "build-debug-final",
            )

    def test_release_and_unmeasured_hardware_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside release"):
            build_case(2, 2)
        with self.assertRaisesRegex(ValueError, "measured TP16 shapes"):
            extended_hardware(2, 2, ())
        with self.assertRaisesRegex(ValueError, "measured TP16 shapes"):
            extended_hardware(12, 12, ())

    def test_prevalidated_io_equals_public_builder_on_real_frozen_source(self) -> None:
        from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
            compile_dense_e2e_sequence_runtime_profiles,
        )
        from llm.frontend.wafer_frontend.passes.program_io import (
            _build_timing_program_io_prevalidated,
            build_timing_program_io,
        )
        from llm.frontend.wafer_frontend.schema._validation_session import (
            builder_validation_session,
        )
        from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
        from llm.test.frontend.unit.test_dense_compile_sequence import _one_die_case
        from .run_extended_dense_sequence_runtime_canary import _seeds

        manifest, template, fabric = _one_die_case()
        with builder_validation_session():
            sequence, profiles = compile_dense_e2e_sequence_runtime_profiles(
                manifest, template, fabric,
                hbm_address_spaces=valid_hbm_address_spaces(fabric),
            )
            segment, source = sequence.segments[0], profiles[0]
            self.assertEqual(source.id, segment.linked_profile_id)
            self.assertEqual(source.manifest, segment.linked_manifest)
            seeds = _seeds(source)
            public = build_timing_program_io(
                source, _DIGEST, state_seed_overrides=seeds,
            )
            private = _build_timing_program_io_prevalidated(
                source, _DIGEST, state_seed_overrides=seeds,
            )
            self.assertEqual(public, private)
            private.validate_against(segment.linked_manifest)
            self.assertEqual(private.program_artifact_sha256, _DIGEST)

    def test_complete_runtime_observation(self) -> None:
        observed = observe_runtime(_valid_runtime_output())
        self.assertEqual(observed["kv_bytes"], list(_KV_BYTES))
        self.assertEqual(observed["p2p_drained_cores"], list(range(0, 64, 4)))
        self.assertEqual(observed["makespan_cycles"], 12345)

    def test_old_kv_extent_and_missing_active_die_fail(self) -> None:
        old_kv = _valid_runtime_output().replace(
            "index=2 bytes=12288", "index=2 bytes=8192",
        )
        with self.assertRaisesRegex(RuntimeError, "KV closure"):
            observe_runtime(old_kv)
        incomplete = _valid_runtime_output().replace(
            "[P5 P2P DRAIN] core=60 residual=0", "",
        )
        with self.assertRaisesRegex(RuntimeError, "16 active P2P"):
            observe_runtime(incomplete)

    def test_missing_final_marker_or_residual_fail(self) -> None:
        reset_final = _valid_runtime_output().replace(
            "index=2 status=done final=1", "index=2 status=done final=0",
        )
        with self.assertRaisesRegex(RuntimeError, "segment closure"):
            observe_runtime(reset_final)
        residual = _valid_runtime_output().replace(
            "core=4 residual=0", "core=4 residual=1",
        )
        with self.assertRaisesRegex(RuntimeError, "16 active P2P"):
            observe_runtime(residual)


@unittest.skipUnless(
    os.environ.get("NPUSIM_EXTENDED_DENSE_CANARY") == "1",
    "requires built production tools and an explicit TP16 runtime budget",
)
class ExtendedDenseRealCanary(unittest.TestCase):
    def _run_shape(self, shape: str) -> None:
        build = _ROOT / "build-debug-final"
        with tempfile.TemporaryDirectory(prefix=f"extended-dense-{shape}-") as raw:
            evidence = run(argparse.Namespace(
                mesh_size=shape,
                output=Path(raw),
                finalizer=build / "npusim_program_finalizer",
                resolver=build / "npusim_program_io_selftest",
                npusim=build / "npusim",
                simulation=_ROOT / "llm/test/program/p5_behavioral_simulation.json",
                timeout=900,
                compile_timeout=1800,
                program_io_timeout=900,
            ))
            self.assertEqual(len(evidence["executions"]), 2)
            self.assertEqual(evidence["active_dies"], list(range(16)))
            self.assertFalse(evidence["published_release_matrix"])
            self.assertTrue(all(
                len(execution["stages"]) == 3
                for execution in evidence["executions"]
            ))

    def test_1x16_two_independent_executions(self) -> None:
        self._run_shape("1x16")

    def test_16x1_two_independent_executions(self) -> None:
        self._run_shape("16x1")


if __name__ == "__main__":
    unittest.main()
