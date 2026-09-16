"""Public ProgramIO coverage for the linked EP1 two-layer MoE forward."""

import hashlib
import struct
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.linker import NaiveManifestLinker
from llm.frontend.wafer_frontend.passes.lower_program import (
    _lower_fragments, _resolve_dependencies,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_route_table_source import (
    build_moe_full_train_route_table_source,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    MoeFullTrainForwardLinkedSource, _resolved_state_abis,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest,
    build_single_die_moe_train_physical_source,
)
from llm.test.frontend.unit.test_moe_full_train_expert_microplan import (
    MoeFullTrainExpertMicroplanTest as Fixture,
)


class MoeFullTrainForwardProgramIoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        leaves = _lower_fragments(
            Fixture.context, _resolve_dependencies(None, None, None, None, None),
        )
        linked = NaiveManifestLinker().link(Fixture.context, leaves)
        cls.source = MoeFullTrainForwardLinkedSource(linked, Fixture.context)
        phase, sequence, _, _ = build_single_die_moe_train_physical_source(
            MoeFullTrainEpPlacementTest,
        )
        route = build_moe_full_train_route_table_source(phase, sequence)
        cls.route = route
        route_by_state = {phase.route_state_refs[seed.layer]: seed.payload
                          for seed in route.seeds}
        seeds = {}
        for item in _resolved_state_abis(cls.source):
            if item.first_access is StateUseAccess.READ:
                abi = item.abi
                seeds[abi.state_ref] = (
                    route_by_state[abi.state_ref]
                    if abi.kind is StateKind.MOE_STATIC_ROUTE
                    else struct.pack("<e", 0.0625) * (abi.size_bytes // 2)
                )
        cls.seeds = seeds

    def test_public_program_io_accepts_real_linked_forward_and_scratch(self):
        contract = build_timing_program_io(
            self.source, hashlib.sha256(b"unit npup").hexdigest(),
            state_seed_overrides=self.seeds,
        )
        self.assertEqual(len(contract.initializations), 76)
        self.assertEqual(len(contract.output_probes), 1)
        self.assertEqual(contract, build_timing_program_io(
            self.source, hashlib.sha256(b"unit npup").hexdigest(),
            state_seed_overrides=self.seeds,
        ))

    def test_missing_state_seed_fails_closed(self):
        missing = dict(self.seeds)
        missing.pop(next(iter(missing)))
        with self.assertRaises(SchemaError):
            build_timing_program_io(
                self.source, hashlib.sha256(b"unit npup").hexdigest(),
                state_seed_overrides=missing,
            )


if __name__ == "__main__":
    unittest.main()
