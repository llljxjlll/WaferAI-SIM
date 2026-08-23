from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from threading import Lock
import tempfile
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import SwiGluWorkload
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MoeCalibrationKind,
)
from llm.test.frontend.integration.moe_swizzle_calibration_provider import (
    MOE_SWIZZLE_CALIBRATION_SIMULATION_CONFIG,
    MoeSwizzleCalibrationArtifactFamily,
    MoeSwizzleCalibrationFamilyKey,
    ProductionMoeSwizzleCalibrationProvider,
    canonical_moe_swizzle_calibration_family_keys,
    materialize_moe_swizzle_calibration_runtime_configs,
)
from llm.test.frontend.integration.run_moe_swizzle_calibration import (
    MoeSwizzleCalibrationExecutable,
    MoeSwizzleCalibrationKey,
    MoeSwizzleSwiGluCalibrationArtifact,
    canonical_moe_swizzle_calibration_keys,
)
from llm.test.frontend.integration.run_swizzle_runtime import (
    validate_runtime_config_paths,
)


@dataclass
class _Builder:
    calls: int = 0
    lock: Lock = field(default_factory=Lock, repr=False)

    def build_family(
        self, family: MoeSwizzleCalibrationFamilyKey, output_root: Path
    ) -> MoeSwizzleCalibrationArtifactFamily:
        with self.lock:
            self.calls += 1
        paths = {}
        for name, payload in (
            ("program", f"program:{family}".encode()),
            ("linked_manifest", f"manifest:{family}".encode()),
            ("program_io", f"program-io:{family}".encode()),
            ("hardware_config", b"hardware"),
            ("simulation_config", b"simulation"),
            ("mapping_config", b"mapping"),
        ):
            path = output_root / name
            path.write_bytes(payload)
            paths[name] = path
        swiglu = None
        if family.kind is MoeCalibrationKind.SWIGLU_GROUP:
            assert family.shape is not None
            _, _, flattened = family.shape
            swiglu = MoeSwizzleSwiGluCalibrationArtifact(
                SwiGluWorkload(
                    (1, 2 * flattened),
                    (1, flattened),
                    (1, 2 * flattened),
                    (1, flattened),
                    DType.FP16,
                ),
                1,
                4 * flattened,
                2 * flattened,
                0,
                0,
            )
        return MoeSwizzleCalibrationArtifactFamily(
            family,
            object(),
            f"production_family::{family.kind.value}::{family.shape}",
            7,
            **paths,
            swiglu_group_artifact=swiglu,
        )


@patch.object(
    MoeSwizzleCalibrationArtifactFamily,
    "validate",
    lambda self, path="": None,
)
@patch.object(
    MoeSwizzleCalibrationExecutable,
    "validate",
    lambda self, path="": None,
)
class ProductionMoeSwizzleCalibrationProviderTest(unittest.TestCase):
    def test_production_simulation_default_is_official_hbm2_dte(self) -> None:
        expected = (
            Path(__file__).resolve().parents[4]
            / "llm/test/sram/simulation.json"
        )
        self.assertEqual(MOE_SWIZZLE_CALIBRATION_SIMULATION_CONFIG, expected)
        self.assertTrue(expected.is_absolute())
        payload = expected.read_text(encoding="utf-8")
        self.assertIn("hbm2-example.json", payload)
        self.assertIn('"use_beha_dte": true', payload)

    def test_runtime_configs_are_rebased_once_and_cwd_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            reference_root = root / "build"
            reference_root.mkdir()
            dram = root / "DRAMSys/configs/hbm2-example.json"
            dram.parent.mkdir(parents=True)
            dram.write_text("{}", encoding="utf-8")
            hardware = root / "hardware.json"
            simulation = root / "simulation.json"
            relative = "../DRAMSys/configs/hbm2-example.json"
            hardware.write_text(json.dumps({
                "memory_system": {
                    "hbm_stacks": [
                        {"channel_dram_config": relative} for _ in range(4)
                    ],
                },
            }), encoding="utf-8")
            simulation.write_text(json.dumps({
                "gpu": {"dram_config_file": relative},
            }), encoding="utf-8")
            outputs = []
            for ordinal in range(2):
                output = root / f"provider-{ordinal}"
                output.mkdir()
                outputs.append(materialize_moe_swizzle_calibration_runtime_configs(
                    output_root=output,
                    hardware_config=hardware,
                    simulation_config=simulation,
                    runtime_reference_root=reference_root,
                ))
            self.assertEqual(
                tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs[0]),
                tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs[1]),
            )
            deep_cwd = root / "runtime/sample-000-group_gemm"
            deep_cwd.mkdir(parents=True)
            resolved = validate_runtime_config_paths(
                hardware_json=outputs[0][0].read_text(encoding="utf-8"),
                simulation_json=outputs[0][1].read_text(encoding="utf-8"),
                runtime_root=deep_cwd,
            )
            self.assertEqual(resolved, (dram.resolve(),) * 5)
            rebased_hardware = json.loads(outputs[0][0].read_text(encoding="utf-8"))
            rebased_simulation = json.loads(outputs[0][1].read_text(encoding="utf-8"))
            self.assertEqual(
                rebased_simulation["gpu"]["dram_config_file"], str(dram.resolve())
            )
            self.assertEqual(
                {
                    item["channel_dram_config"]
                    for item in rebased_hardware["memory_system"]["hbm_stacks"]
                },
                {str(dram.resolve())},
            )

    def test_exact_14_kind_registry_and_one_family_reuse_smoke(self) -> None:
        families = canonical_moe_swizzle_calibration_family_keys()
        self.assertEqual(len(families), 28)
        self.assertEqual({item.kind for item in families}, set(MoeCalibrationKind))
        builder = _Builder()
        builders = {kind: builder for kind in MoeCalibrationKind}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = ProductionMoeSwizzleCalibrationProvider(
                artifact_root=root / "families",
                builders=builders,
            )
            self.assertEqual(
                tuple(item.kind for item in provider.bindings),
                tuple(MoeCalibrationKind),
            )
            sample0 = root / "sample0"
            sample1 = root / "sample1"
            sample0.mkdir()
            sample1.mkdir()
            keys = canonical_moe_swizzle_calibration_keys()
            first = keys[0]
            second = next(
                item for item in keys
                if (item.kind, item.shape) == (first.kind, first.shape)
                and item != first
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                one = executor.submit(provider.materialize, first, sample0)
                two = executor.submit(provider.materialize, second, sample1)
                executable0 = one.result()
                executable1 = two.result()
            self.assertEqual(builder.calls, 1)
            self.assertEqual(executable0.key, first)
            self.assertEqual(executable1.key, second)
            self.assertEqual(executable0.program, executable1.program)
            self.assertEqual(
                executable0.linked_manifest, executable1.linked_manifest
            )
            self.assertEqual(executable0.program_io, executable1.program_io)
            executable0.validate()
            executable1.validate()

    def test_all_168_keys_build_exactly_28_immutable_families(self) -> None:
        builder = _Builder()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = ProductionMoeSwizzleCalibrationProvider(
                artifact_root=root / "families",
                builders={kind: builder for kind in MoeCalibrationKind},
            )
            keys = canonical_moe_swizzle_calibration_keys()

            def materialize(item: tuple[int, MoeSwizzleCalibrationKey]):
                ordinal, key = item
                sample = root / f"sample-{ordinal:03d}"
                sample.mkdir()
                return provider.materialize(key, sample)

            with ThreadPoolExecutor(max_workers=4) as executor:
                executables = tuple(executor.map(materialize, enumerate(keys)))
            self.assertEqual(builder.calls, 28)
            artifacts = {}
            for executable in executables:
                family = (executable.key.kind, executable.key.shape)
                witness = (
                    executable.production_source_ref,
                    executable.program,
                    executable.linked_manifest,
                    executable.program_io,
                )
                self.assertEqual(artifacts.setdefault(family, witness), witness)
            self.assertEqual(len(artifacts), 28)

    def test_registry_and_family_contract_fail_closed(self) -> None:
        builder = _Builder()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = {
                kind: builder
                for kind in MoeCalibrationKind
                if kind is not MoeCalibrationKind.TERMINAL_DONE
            }
            with self.assertRaisesRegex(SchemaError, "exactly all 14"):
                ProductionMoeSwizzleCalibrationProvider(
                    artifact_root=root / "missing",
                    builders=missing,
                )
            with self.assertRaisesRegex(SchemaError, "artifact root must be absolute"):
                ProductionMoeSwizzleCalibrationProvider(
                    artifact_root=Path("families"),
                    builders={kind: builder for kind in MoeCalibrationKind},
                )


if __name__ == "__main__":
    unittest.main()
