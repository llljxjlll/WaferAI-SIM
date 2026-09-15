"""Native core IDs and C2C ports must match the production Fabric."""

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.integration.flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from llm.test.frontend.integration.run_dense_sequence_runtime_canary import (
    _bind_native_hardware_to_fabric,
)


class DenseNativeCoregridBindingTest(unittest.TestCase):
    def test_all_representative_core_ids_and_edge_ports_are_physically_identical(self) -> None:
        for rows, columns in ((1, 1), (1, 4), (4, 1), (2, 2),
                              (2, 3), (3, 2), (3, 3), (10, 10)):
            with self.subTest(mesh=(rows, columns)):
                fabric = physical_fabric_from_data(
                    minimal_hardware(columns, rows, sram_bytes=65536)
                )
                hardware = json.loads(
                    specialize_p5_large_release_hardware(rows, columns)
                )
                self.assertEqual(hardware['x'], 4)
                self.assertNotIn('y', hardware)
                grid = _bind_native_hardware_to_fabric(hardware, fabric)
                self.assertEqual(grid, (2, 2))
                self.assertEqual((hardware['x'], hardware['y']), grid)
                self.assertEqual(hardware['die'], {'x': columns, 'y': rows})
                self.assertTrue(all(port['idx'] == 0 for port in
                                    hardware['die_ports']['overrides']))
                for die in fabric.dies:
                    for core in die.cores:
                        self.assertEqual(core.runtime_core_id,
                                         die.id * 4 + core.local_core_id)

    def test_wrong_native_die_shape_and_runtime_stride_reject_before_simulation(self) -> None:
        fabric = physical_fabric_from_data(minimal_hardware(4, 1))
        hardware = json.loads(specialize_p5_large_release_hardware(1, 4))
        hardware['die']['x'] = 3
        with self.assertRaisesRegex(ValueError, 'physical die grid'):
            _bind_native_hardware_to_fabric(hardware, fabric)
        hardware['die']['x'] = 4
        die = fabric.dies[1]
        wrong_core = replace(die.cores[0], runtime_core_id=16)
        wrong_die = replace(die, cores=(wrong_core, *die.cores[1:]))
        wrong_fabric = replace(fabric, dies=(fabric.dies[0], wrong_die,
                                            *fabric.dies[2:]))
        with self.assertRaisesRegex(ValueError, 'runtime core ID'):
            _bind_native_hardware_to_fabric(hardware, wrong_fabric)


if __name__ == '__main__':
    unittest.main()
