from __future__ import annotations

import math
import unittest

from exps.exp4.blocker_models import (
    DTETransfer,
    build_hbm3_topology,
    interleaved_stack_bytes,
    schedule_dte,
    service_d2d_edge,
    service_hbm3,
    stripe_flow,
)
from exps.exp4.candidate_loader import load_candidates, pe_organization
from exps.exp4.hardware_resources import SRAMSharedContract


class CandidateLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.candidates = load_candidates()

    def test_all_383_candidates_close_frozen_semantics(self) -> None:
        self.assertEqual(len(self.candidates), 383)
        self.assertEqual(len({item.candidate_digest for item in self.candidates}), 383)
        for candidate in self.candidates:
            self.assertEqual(candidate.status, "valid")
            self.assertEqual(candidate.f_Hz, 500_000_000)
            self.assertEqual(candidate.P_core_FLOPs, 2 * candidate.N_PE * candidate.f_Hz)
            self.assertEqual(candidate.DTE_channel, math.ceil(candidate.B_GBs / 128))
            self.assertEqual(candidate.dte_aggregate_width_bits, candidate.DTE_channel * 2048)
            self.assertEqual(candidate.D2D_edge_one_dir_GBs, min(candidate.d_d2d * candidate.B_GBs, 512))
            self.assertEqual(candidate.HBM_stack_count, candidate.e_H * candidate.m)
            self.assertEqual(candidate.HBM_stack_capacity_GB, 16)
            self.assertEqual(candidate.HBM_stack_peak_GBs, 819.2)
            self.assertTrue(set(candidate.hbm_port_positions).isdisjoint(candidate.d2d_port_positions_hbm_edge))
            self.assertEqual(candidate.sram_read_GBs, candidate.B_s_GBs)
            self.assertEqual(candidate.sram_write_GBs, candidate.B_s_GBs)

    def test_pe_mapping_is_shape_consistent(self) -> None:
        expected = {
            1024: (32, 32, 1), 4096: (64, 64, 1), 8192: (64, 64, 2),
            12288: (64, 64, 3), 16384: (64, 64, 4),
        }
        for n_pe, organization in expected.items():
            actual = pe_organization(n_pe)
            self.assertEqual((actual.exu_x, actual.exu_y, actual.sa_count), organization)
            self.assertEqual(actual.exu_x * actual.exu_y * actual.sa_count, n_pe)

    def test_runner_aliases(self) -> None:
        candidate = self.candidates[0]
        data = candidate.as_dict()
        self.assertEqual(candidate.K_bytes, candidate.K_MiB * 1024 * 1024)
        self.assertEqual(data["digest"], candidate.candidate_digest)
        self.assertEqual(data["dte_channels"], candidate.DTE_channel)
        self.assertEqual(data["d2d_edge_GBs"], candidate.D2D_edge_one_dir_GBs)


class BlockerAnalyticalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.candidates = load_candidates()

    def test_dte_channels_are_independent(self) -> None:
        transfers = [DTETransfer("a", 256, "read"), DTETransfer("b", 256, "write")]
        one = schedule_dte(transfers, 1)
        two = schedule_dte(transfers, 2)
        self.assertEqual(one.makespan_cycles, 2)
        self.assertEqual(two.makespan_cycles, 1)
        self.assertEqual(sum(entry.byte_count for entry in two.ledger), 512)
        self.assertTrue(two.analytical_parallel_channels)

    def test_d2d_ports_share_one_physical_edge(self) -> None:
        low = service_d2d_edge("0->1", 4096, 2, 128)
        saturated = service_d2d_edge("0->1", 4096, 9, 512)
        self.assertEqual(low.capacity_GBs, 256)
        self.assertEqual(saturated.capacity_GBs, 512)
        self.assertEqual(low.byte_count, 4096)

    def test_arbitrary_stripes_conserve_bytes_and_obey_bound(self) -> None:
        for stripe_count in (3, 5, 9):
            allocation = stripe_flow("flow-x", 10_003, tuple(range(stripe_count)))
            self.assertEqual(allocation.stripe_count, stripe_count)
            self.assertEqual(sum(value for _, value in allocation.port_bytes), 10_003)
            self.assertLessEqual(allocation.imbalance, 1 + allocation.payload_bytes / allocation.byte_count)
            self.assertEqual(allocation, stripe_flow("flow-x", 10_003, tuple(range(stripe_count))))

    def test_hbm3_noncontiguous_ports_and_interleave(self) -> None:
        candidate = next(c for c in self.candidates if c.t_hbm >= 3 and c.m >= 2)
        topology = build_hbm3_topology(candidate)
        self.assertEqual(len(topology.stacks), candidate.e_H * candidate.m)
        self.assertEqual(topology.total_capacity_bytes, len(topology.stacks) * 16_000_000_000)
        self.assertEqual(topology.stacks[0].port_positions, candidate.hbm_port_positions)
        distribution = interleaved_stack_bytes(17, 8193, len(topology.stacks))
        self.assertEqual(sum(distribution), 8193)
        service = service_hbm3(candidate, 17, 8193, first_byte_latency_cycles=7)
        self.assertEqual(sum(service.stack_bytes), 8193)
        self.assertGreaterEqual(service.service_cycles, 7)


class SRAMContractTests(unittest.TestCase):
    def test_all_initiators_share_one_read_and_one_write_budget(self) -> None:
        contract = SRAMSharedContract(core_id=7, B_s_GBs=512, bank_count=16)
        read_ids = {contract.resource_for(name, "read") for name in contract.initiators}
        write_ids = {contract.resource_for(name, "write") for name in contract.initiators}
        self.assertEqual(read_ids, {"sram.read.core[7]"})
        self.assertEqual(write_ids, {"sram.write.core[7]"})
        self.assertNotEqual(read_ids, write_ids)
        self.assertEqual(contract.bank_for_address(0), 0)
        self.assertEqual(contract.bank_for_address(256), 1)
        with self.assertRaises(ValueError):
            contract.assert_shared_budget([("dte", "read", "sram.read.dte.core[7]")])


if __name__ == "__main__":
    unittest.main()
