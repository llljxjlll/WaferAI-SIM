"""Typed source provenance and genuine official EP IR1 acceptance."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_ir1_source import (
    build_moe_ep_placed_ir1_candidate,
    build_moe_ep_shared_reverse_ir1_candidate,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_ir0 import (
    build_moe_full_train_forward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_placement import (
    build_moe_full_train_ep_placement,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateIdentity, StateKind,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import NaiveProjectToIR2
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy, _ordinary_rank_local_view,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1, RankPlacement
from llm.frontend.wafer_frontend.schema.ir0 import OpKind, StateAccess


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

    def test_ep1_exact_moe_compute_carriers_enter_official_ir2(self):
        phase, sequence, placement, context = (
            build_single_die_moe_train_physical_source(Fixture)
        )
        candidate = build_moe_ep_placed_ir1_candidate(
            phase, original_dense=Fixture.dense, sequence=sequence,
            placement=placement, context=context, dense_manifest=Fixture.manifest,
        )
        placed = IR1.create(
            producer_pass="placement", **candidate.physical_ir1._semantic_key(),
        )
        partitioned = partition_ir1(placed)
        projected = NaiveProjectToIR2().run(
            partitioned, (), (), state_transfers=(),
        )
        projected.validate_against(partitioned, (), (), ())
        moe = [task for dag in projected.dags for task in dag.tasks
               if task.compute is not None
               and task.compute.op_kind in {
                   OpKind.MOE_ROUTER, OpKind.MOE_ROUTE_FREEZE,
                   OpKind.MOE_DISPATCH, OpKind.MOE_EXPERT_FORWARD,
                   OpKind.MOE_COMBINE,
               }]
        schedule = NaiveIntraDiePolicy().schedule(projected, partitioned)
        self.assertEqual(len(schedule.schedules), 1)
        self.assertEqual(len(moe), 10)
        self.assertEqual({kind: sum(task.compute.op_kind is kind for task in moe)
                          for kind in {task.compute.op_kind for task in moe}},
                         {kind: 2 for kind in {
                             OpKind.MOE_ROUTER, OpKind.MOE_ROUTE_FREEZE,
                             OpKind.MOE_DISPATCH, OpKind.MOE_EXPERT_FORWARD,
                             OpKind.MOE_COMBINE,
                         }})
        expert = next(task.compute for task in moe
                      if task.compute.op_kind is OpKind.MOE_EXPERT_FORWARD)
        self.assertEqual(tuple(operand.role for operand in expert.inputs),
                         ("expert_activation", "gate_weight", "up_weight", "down_weight"))
        with self.assertRaisesRegex(SchemaError, "operand roles/arity"):
            replace(expert, inputs=(replace(expert.inputs[0], role="forged"),
                                    *expert.inputs[1:])).validate("forged_moe_compute")
        with self.assertRaisesRegex(SchemaError, "MoE compute kind differs"):
            replace(expert, op_kind=OpKind.MOE_ROUTER).validate("forged_moe_compute")
        expert_task = next(task for task in moe if task.compute is expert)
        input_value = next(value for value in partitioned.values
                           if value.id == expert.inputs[0].value_id)
        group = partitioned.groups[0]
        forged_group = replace(group, logical_shape=(1, 2),
                               placements=(*group.placements,
                                           RankPlacement(1, 1, (0, 1))))
        with self.assertRaisesRegex(SchemaError, "one-axis physical group"):
            _ordinary_rank_local_view(
                expert_task, input_value,
                replace(partitioned, groups=(forged_group,)),
            )

    def test_ep2_both_steps_place_real_shared_reverse_and_preserve_owners(self):
        candidates = []
        for step in (0, 1):
            phase = self.phase if step == 0 else build_moe_full_train_forward_ir0(
                self.original_dense, self.sequence, step=step,
            )
            placement = self.placement if step == 0 else build_moe_full_train_ep_placement(
                phase, original_dense=self.original_dense,
                dense_manifest=self.dense_manifest, sequence=self.sequence,
                context=self.context,
            )
            candidate = build_moe_ep_shared_reverse_ir1_candidate(
                phase, original_dense=self.original_dense,
                sequence=self.sequence, placement=placement,
                context=self.context, dense_manifest=self.dense_manifest,
            )
            candidate.validate_source_against(
                phase, original_dense=self.original_dense,
                sequence=self.sequence, placement=placement,
                context=self.context, dense_manifest=self.dense_manifest,
            )
            self.assertEqual(len(candidate.physical_ir1.nodes), 38)
            self.assertEqual(len(candidate.forward.owner_proofs), 16)
            self.assertEqual(len(candidate.physical_ir1.persistent_state_manifest.bindings), 29)
            self.assertEqual({proof.physical_die for proof in
                              candidate.forward.owner_proofs}, {0, 1})
            declarations = {state.id: state for state in
                            candidate.physical_ir1.persistent_state_manifest.declarations}
            self.assertTrue(all(
                access.rank == declarations[access.state_ref].identity.ep_owner_rank
                for access in candidate.physical_ir1.state_accesses
                if declarations[access.state_ref].identity.ep_owner_rank is not None
            ))
            placed = IR1.create(
                producer_pass="placement", **candidate.physical_ir1._semantic_key(),
            )
            partitioned = partition_ir1(placed)
            projected = NaiveProjectToIR2().run(
                partitioned, (), (), state_transfers=(),
            )
            projected.validate_against(partitioned, (), (), ())
            self.assertEqual(len(projected.dags), 2)
            self.assertEqual(sum(len(dag.tasks) for dag in projected.dags), 106)
            candidates.append(candidate)
        self.assertNotEqual(candidates[0].physical_ir1.id,
                            candidates[1].physical_ir1.id)
        first = candidates[0]
        declarations = {state.id: state for state in
                        first.physical_ir1.persistent_state_manifest.declarations}
        remote = next(access for access in first.physical_ir1.state_accesses
                      if declarations[access.state_ref].identity.ep_owner_rank == 1)
        forged_access = StateAccess.create(
            node_ref=remote.node_ref, state_ref=remote.state_ref,
            mode=remote.mode, rank=0,
            read_offset=remote.read_offset, read_shape=remote.read_shape,
            write_offset=remote.write_offset, write_shape=remote.write_shape,
        )
        forged_ir1 = IR1.create(
            producer_pass=first.physical_ir1.producer_pass,
            **{**first.physical_ir1._semantic_key(),
               "state_accesses": tuple(
                   forged_access if item is remote else item
                   for item in first.physical_ir1.state_accesses)},
        )
        with self.assertRaisesRegex(SchemaError, "physical TP shard or EP owner"):
            forged_ir1.validate("forged_ep_rank0")
        forged = replace(first.physical_ir1.nodes[-1],
                         execution_group_ref="forged_group")
        with self.assertRaisesRegex(SchemaError, "preserve source gradients"):
            replace(first, physical_ir1=replace(
                first.physical_ir1,
                nodes=(*first.physical_ir1.nodes[:-1], forged),
            )).validate_source_against(
                self.phase, original_dense=self.original_dense,
                sequence=self.sequence, placement=self.placement,
                context=self.context, dense_manifest=self.dense_manifest,
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
                             persistent_state_manifest.bindings),29)
        actual_rank0_die=next(placement.die_id for placement in
                    self.placement.physical_group.placements
                    if placement.rank==0)
        self.assertEqual(actual_rank0_die,0)
        # TP shard zero is genuine on both physical EP owners; die1 requires
        # opt-in source-signed EP ownership, not a fake TP shard one.
        self.assertTrue(all(proof.physical_die!=actual_rank0_die
                            for proof in proofs))
        states={state.id:state for state in self.candidate.physical_ir1.
                persistent_state_manifest.declarations}
        self.assertTrue(all(states[proof.declaration_ref].identity.ep_owner_rank
                            ==proof.e2e_ep_rank for proof in proofs))

    def test_official_ir1_accepts_actual_ep_home_and_full_forward_source(self):
        self.candidate.validate_official_ir1()

    def test_legacy_dense_identity_preserves_id_when_ep_field_absent(self):
        shared=next(state for state in self.candidate.physical_ir1.
                    persistent_state_manifest.declarations
                    if state.identity.ep_owner_rank is None)
        identity=shared.identity
        self.assertEqual(identity, PersistentStateIdentity.create(
            kind=identity.kind,instance_ref=identity.instance_ref,
            mesh_ref=identity.mesh_ref,request_ref=identity.request_ref,
            layer_index=identity.layer_index,tensor_ref=identity.tensor_ref,
            shard_index=identity.shard_index,generation=identity.generation,
        ))

    def test_ep_rank_cannot_be_hidden_in_generation(self):
        expert=next(state for state in self.candidate.physical_ir1.
                    persistent_state_manifest.declarations
                    if state.identity.ep_owner_rank==1)
        identity=expert.identity
        with self.assertRaisesRegex(SchemaError,"EP placement owner"):
            PersistentStateIdentity.create(
                kind=StateKind.PARAMETER,instance_ref=identity.instance_ref,
                mesh_ref=identity.mesh_ref,request_ref=None,layer_index=None,
                tensor_ref=identity.tensor_ref,shard_index=0,generation=1,
                ep_owner_rank=1,
            )

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
