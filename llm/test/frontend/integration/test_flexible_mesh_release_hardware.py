from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseFamily,
    FlexibleMeshReleaseTool,
    FlexibleMeshReleaseToolKind,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec

from flexible_mesh_release_dense import FlexibleDenseReleaseAdapter
from flexible_mesh_release_hardware import (
    p5_large_hardware_template_json,
    specialize_p5_large_release_hardware,
    specialize_release_hardware,
)
from flexible_mesh_release_meshslice import FlexibleMeshSliceReleaseAdapter
from flexible_mesh_release_moe import FlexibleMoeReleaseAdapter
from flexible_mesh_release_profiles import release_trace_model_digest
from run_flexible_mesh_release import FlexibleMeshReleaseRunner


_EXACT = {
    (1, 1): (
        2229,
        "1fa3219dc11bd3342b1348fe6c001cb905f05c3796a84e17360f7440e5e54b74",
    ),
    (10, 10): (
        37222,
        "00590b3eb0730fbfa5782d61d149fd998c78dc035cbb814d80893f2aeab82b6e",
    ),
}
_ROOT = Path(__file__).resolve().parents[4]
_BUILD = _ROOT / "build-debug-final"
_SIMULATION = _ROOT / "llm/test/program/p5_behavioral_simulation.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"


def _case(
    family: FlexibleMeshReleaseFamily,
    rows: int,
    columns: int,
) -> FlexibleMeshReleaseCase:
    return FlexibleMeshReleaseCase.create(
        family=family,
        mesh=RectMeshSpec(rows, columns),
        trace_model_digest=release_trace_model_digest(family),
        runtime_profile_version="flexible-mesh-timing-v2",
    )


class FlexibleMeshReleaseHardwareTest(unittest.TestCase):
    def test_1x1_and_10x10_exact_bytes_and_sha(self) -> None:
        for shape, (byte_count, expected_sha) in _EXACT.items():
            with self.subTest(shape=shape):
                hardware_json = specialize_p5_large_release_hardware(*shape)
                raw = json.loads(hardware_json)
                self.assertEqual(len(hardware_json.encode("utf-8")), byte_count)
                self.assertEqual(
                    hashlib.sha256(hardware_json.encode("utf-8")).hexdigest(),
                    expected_sha,
                )
                self.assertEqual(raw["die"], {"x": shape[1], "y": shape[0]})
                self.assertEqual(raw["memory"]["sram_size"], 1 << 20)
                self.assertEqual(raw["memory"]["sram"]["capacity_bytes"], 1 << 20)
                self.assertEqual(raw["memory"]["sram"]["regions"], [{
                    "name": "dense_release",
                    "base_bytes": 0,
                    "size_bytes": 1 << 20,
                    "allocator": "block",
                    "spillable": False,
                    "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
                }])
                self.assertEqual(
                    len(raw["memory_system"]["hbm_stacks"]),
                    shape[0] * shape[1],
                )
                self.assertEqual(
                    len(raw["memory_system"]["address_policy"]["home_ranges"]),
                    shape[0] * shape[1],
                )
                self.assertTrue(all(
                    override["idx"] == 2
                    for override in raw["die_ports"]["overrides"]
                ))

    def test_three_adapters_emit_identical_hardware_bytes(self) -> None:
        template = p5_large_hardware_template_json()
        dense = FlexibleDenseReleaseAdapter(mapping_text="0:0\n")
        moe = FlexibleMoeReleaseAdapter(
            FlexibleMeshReleaseFamily.MOE_INFERENCE,
            hardware_template_json=template,
            mapping_text="0:0\n",
        )
        meshslice = FlexibleMeshSliceReleaseAdapter(
            FlexibleMeshReleaseFamily.MESHSLICE_AG,
            mapping_text="0:0\n",
            hardware_template_json=template,
        )
        for rows, columns in ((1, 1), (10, 10)):
            with self.subTest(shape=(rows, columns)):
                dense_bytes = dense._hardware_json(_case(
                    FlexibleMeshReleaseFamily.DENSE_TRAIN, rows, columns,
                ))
                moe_bytes = moe._hardware_json(_case(
                    FlexibleMeshReleaseFamily.MOE_INFERENCE, rows, columns,
                ))
                meshslice_bytes = meshslice._hardware_json(_case(
                    FlexibleMeshReleaseFamily.MESHSLICE_AG, rows, columns,
                ))
                self.assertEqual(dense_bytes, moe_bytes)
                self.assertEqual(moe_bytes, meshslice_bytes)
                self.assertEqual(
                    hashlib.sha256(dense_bytes.encode("utf-8")).hexdigest(),
                    _EXACT[(rows, columns)][1],
                )

    def test_specializer_fails_closed(self) -> None:
        template = p5_large_hardware_template_json()
        for rows, columns in ((0, 1), (1, 11), (True, 1)):
            with self.subTest(shape=(rows, columns)):
                with self.assertRaisesRegex(SchemaError, "1..10 rectangle"):
                    specialize_release_hardware(template, rows, columns)
        with self.assertRaisesRegex(SchemaError, "invalid"):
            specialize_release_hardware("{", 1, 1)
        with self.assertRaisesRegex(SchemaError, "lacks SRAM"):
            specialize_release_hardware("{}", 1, 1)


class _UnavailableAdapter:
    def __init__(self, family: FlexibleMeshReleaseFamily) -> None:
        self.family = family


@unittest.skipUnless(
    os.environ.get("NPUSIM_FLEXIBLE_RELEASE_HARDWARE_CANARY") == "1",
    "requires built production tools",
)
class FlexibleMeshReleaseHardwareCanary(unittest.TestCase):
    def _run(
        self,
        case: FlexibleMeshReleaseCase,
        adapter: object,
    ) -> None:
        prepared = adapter.materialize(case)
        tool_paths = (
            _BUILD / "npusim_program_finalizer",
            _BUILD / "npusim_program_io_selftest",
            _BUILD / "npusim",
        )
        tools = tuple(
            FlexibleMeshReleaseTool(
                kind=kind,
                binary_path=str(path.resolve()),
                version="build-debug-final-canary",
                sha256=(digest := hashlib.sha256(path.read_bytes()).hexdigest()),
                allowlisted_sha256=(digest,),
            )
            for kind, path in zip(FlexibleMeshReleaseToolKind, tool_paths)
        )
        binding = FlexibleMeshReleaseBinding.create(
            runtime_profile_version="flexible-mesh-timing-v2",
            environment_profile_version="p5-rect-release-v1",
            tools=tools,
            hardware_config_sha256=hashlib.sha256(
                p5_large_hardware_template_json().encode("utf-8")
            ).hexdigest(),
            simulation_config_sha256=hashlib.sha256(
                _SIMULATION.read_bytes()
            ).hexdigest(),
            mapping_config_sha256=hashlib.sha256(
                prepared.mapping_text.encode("utf-8")
            ).hexdigest(),
        )
        adapters = tuple(
            adapter if family is case.family else _UnavailableAdapter(family)
            for family in FlexibleMeshReleaseFamily
        )
        runtime_root = Path(tempfile.mkdtemp(
            prefix="flexible-mesh-release-hardware-canary-",
        ))
        print(f"CANARY_RUNTIME_ROOT={runtime_root}")
        runner = FlexibleMeshReleaseRunner(
            binding=binding,
            adapters=adapters,
            finalizer=tool_paths[0],
            resolver=tool_paths[1],
            npusim=tool_paths[2],
            simulation=_SIMULATION,
            runtime_root=runtime_root,
            timeout=180,
        )
        evidence = runner.run_case(case)
        self.assertTrue(evidence.runtime_verified)
        self.assertTrue(evidence.repeatability_verified)
        self.assertEqual(len(evidence.executions), 2)
        actual_hardware_sha = hashlib.sha256(
            prepared.hardware_json.encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            {execution.hardware_config_sha256 for execution in evidence.executions},
            {actual_hardware_sha},
        )

    def test_dense_2x1(self) -> None:
        self._run(
            _case(FlexibleMeshReleaseFamily.DENSE_TRAIN, 2, 1),
            FlexibleDenseReleaseAdapter(mapping_text=_MAPPING.read_text()),
        )

    def test_moe_1x2(self) -> None:
        self._run(
            _case(FlexibleMeshReleaseFamily.MOE_INFERENCE, 1, 2),
            FlexibleMoeReleaseAdapter(
                FlexibleMeshReleaseFamily.MOE_INFERENCE,
                hardware_template_json=p5_large_hardware_template_json(),
                mapping_text=_MAPPING.read_text(),
            ),
        )

    def test_meshslice_1x3(self) -> None:
        self._run(
            _case(FlexibleMeshReleaseFamily.MESHSLICE_AG, 1, 3),
            FlexibleMeshSliceReleaseAdapter(
                FlexibleMeshReleaseFamily.MESHSLICE_AG,
                mapping_text=_MAPPING.read_text(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
