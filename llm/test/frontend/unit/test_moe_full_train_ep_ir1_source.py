"""Typed source provenance and current official EP IR1 refusal."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_ir1_source import (
    build_moe_ep_placed_ir1_candidate,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeFullTrainEpIr1SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.phase=Fixture.phase
        cls.original_dense=Fixture.dense
        cls.sequence=Fixture.sequence
        cls.placement=Fixture.placement
        cls.context=Fixture.context
        cls.dense_manifest=Fixture.manifest
        cls.candidate=build_moe_ep_placed_ir1_candidate(
            cls.phase,original_dense=cls.original_dense,
            sequence=cls.sequence,placement=cls.placement,
            context=cls.context,dense_manifest=cls.dense_manifest,
        )

    def _check(self,source):
        source.validate_source_against(
            self.phase,self.original_dense,self.sequence,
            self.placement,self.context,self.dense_manifest,
        )

    def test_all_32_shared_moe_physical_candidates_and_16_ep_owners(self):
        self._check(self.candidate)
        self.assertEqual(len(self.candidate.physical_ir1.nodes),32)
        self.assertEqual(len(self.candidate.owner_proofs),16)
        proofs=[proof for proof in self.candidate.owner_proofs
                if proof.e2e_ep_rank==1]
        self.assertEqual(len(proofs),8)
        self.assertTrue(all(proof.physical_die==1 and proof.tp_shard==0
                            for proof in proofs))
        self.assertEqual(len(self.candidate.physical_ir1.
                             persistent_state_manifest.bindings),27)
        actual_rank0_die=next(placement.die_id for placement in
                    self.placement.physical_group.placements
                    if placement.rank==0)
        self.assertEqual(actual_rank0_die,0)
        # Existing IR1.validate's `shard_index==physical rank` would map these
        # eight real EP1 source parameters to die0: a second official blocker.
        self.assertTrue(all(proof.physical_die!=actual_rank0_die
                            for proof in proofs))

    def test_official_ir1_currently_rejects_new_physical_opcode(self):
        with self.assertRaises(KeyError) as failure:
            self.candidate.validate_official_ir1()
        self.assertIs(failure.exception.args[0],OpKind.MOE_EXPERT_FORWARD)

    def test_expert_ep1_cannot_be_relabelled_as_tp1_shard(self):
        proofs=list(self.candidate.owner_proofs)
        position=next(i for i,proof in enumerate(proofs)
                      if proof.e2e_ep_rank==1)
        proofs[position]=replace(proofs[position],tp_shard=1)
        with self.assertRaisesRegex(SchemaError,"source E2E EP tensor"):
            self._check(replace(self.candidate,owner_proofs=tuple(proofs)))

    def test_e2e_ep_owner_cannot_migrate_without_real_die_and_abi(self):
        proofs=list(self.candidate.owner_proofs)
        position=next(i for i,proof in enumerate(proofs)
                      if proof.e2e_ep_rank==1)
        proofs[position]=replace(proofs[position],e2e_ep_rank=0)
        with self.assertRaisesRegex(SchemaError,"source E2E EP tensor"):
            self._check(replace(self.candidate,owner_proofs=tuple(proofs)))

    def test_source_aggregate_64b_projection_slice_cannot_forge_0(self):
        proofs=list(self.candidate.owner_proofs)
        position=next(i for i,proof in enumerate(proofs)
                      if proof.source_aggregate_slice_offset==128)
        proofs[position]=replace(proofs[position],
                                 source_aggregate_slice_offset=0)
        with self.assertRaisesRegex(SchemaError,"aggregate StateABI"):
            self._check(replace(self.candidate,owner_proofs=tuple(proofs)))

    def test_one_forward_physical_node_without_source_moe_block_fails(self):
        candidate=self.candidate.physical_ir1
        nodes=candidate.nodes[:-1]
        modified=replace(candidate,nodes=nodes)
        with self.assertRaisesRegex(SchemaError,"every original shared/MoE"):
            self._check(replace(self.candidate,physical_ir1=modified))


if __name__=="__main__":
    unittest.main()
