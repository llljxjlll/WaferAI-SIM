"""Real two-layer Dense/MoE shared HBM collision and EP rank-home tests."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    materialize_flexible_dense_train_forward,
)
from llm.frontend.wafer_frontend.passes.moe_compile_sequence import compile_moe_sequence
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.lowering.moe_full_training_namespace import (
    _HBM_REGION_SHIFT,_HBM_MOE_LAYER_STRIDE,
    namespace_moe_training_unit,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_ir0 import (
    build_moe_full_train_forward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_placement import (
    build_moe_full_train_ep_placement,
)
from llm.frontend.wafer_frontend.schema.persistent_state import HbmAddressSpace
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier,MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily,WorkloadRunRequest,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit.test_flexible_dense_train import _spec
from llm.test.frontend.unit.test_moe_compile_sequence import _request
from llm.test.frontend.unit.test_workload_materialization import _capability
from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import (
    _rank_local_fabric,
)


class MoeFullTrainEpPlacementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fabric=physical_fabric_from_data(minimal_hardware(2,1,sram_bytes=131072))
        cap=1<<25
        spaces=tuple(HbmAddressSpace.create(
            die_id=rank,base_address=rank*cap,size_bytes=cap,
            alignment_bytes=64,
        ) for rank in (0,1))
        base=_spec(1,1)
        spec=replace(
            base, model=replace(base.model,V=16,NH=2,KVH=2,DH=2,
                                rotary_dim=2,max_position_embeddings=32),
            workload=replace(base.workload,
                             train=replace(base.workload.train,seq_len=4)),
        )
        spec.validate("new_resident_32MiB_128KiB_dense")
        cls.spec=spec
        cls.forward=materialize_flexible_dense_train_forward(
            spec,RectMeshSpec(1,1),_rank_local_fabric(fabric),(spaces[0],),
            producer_pass="moe_ep_train_shared_dense_resident_32MiB_128KiB",
        )
        cls.dense=cls.forward.plan.forward_graph
        cls.manifest=cls.forward.linked_forward.manifest
        old_request=_request(WorkloadFamily.MOE_TRAINING)
        semantic=old_request._semantic_key()
        # The new public workload.case_id truthfully requests no SRAM spill on
        # the 128KiB profile; old resident 16MiB/64KiB case stays untouched.
        semantic["memory"]=replace(old_request.memory,allow_sram_spill=False)
        request=WorkloadRunRequest.create(**semantic)
        assert request.case_id!=old_request.case_id
        capacities=tuple(MemoryTierCapacity.create(
            tier=MemoryTier.HBM,location_ref=f"die:{rank}",
            base_address=0,capacity_bytes=cap,alignment_bytes=16,
        ) for rank in (0,1))
        preflight=materialize_workload_preflight(
            request,_capability(supported=True),capacities=capacities,
        )
        cls.sequence=compile_moe_sequence(
            preflight,source_rank_policy="rank0_shared_spine",
        )
        cls.old_workload_case_ref=old_request.case_id
        old_cap=1<<24
        old_spaces=tuple(HbmAddressSpace.create(
            die_id=rank,base_address=rank*old_cap,size_bytes=old_cap,
            alignment_bytes=64,
        ) for rank in (0,1))
        old_preflight=materialize_workload_preflight(
            old_request,_capability(supported=True),
            capacities=tuple(MemoryTierCapacity.create(
                tier=MemoryTier.HBM,location_ref=f"die:{rank}",
                base_address=0,capacity_bytes=old_cap,alignment_bytes=16,
            ) for rank in (0,1)),
        )
        cls.old_sequence=compile_moe_sequence(
            old_preflight,source_rank_policy="rank0_shared_spine",
        )
        old_fabric=physical_fabric_from_data(
            minimal_hardware(2,1,sram_bytes=65536),
        )
        cls.old_forward=materialize_flexible_dense_train_forward(
            spec,RectMeshSpec(1,1),_rank_local_fabric(old_fabric),
            (old_spaces[0],),
            producer_pass="old_moe_shared_dense_16MiB_64KiB_rejection",
        )
        cls.old_phase=build_moe_full_train_forward_ir0(
            cls.old_forward.plan.forward_graph,cls.old_sequence,
        )
        cls.old_context_64=PlacementContext.create(
            producer_pass="old_moe_train_16MiB_64KiB_rejection",
            fabric=old_fabric,placement=cls.old_forward.plan.source_experiment.placement,
            hbm_address_spaces=old_spaces,
        )
        cls.old_context_128=PlacementContext.create(
            producer_pass="old_moe_train_16MiB_128KiB_hbm_rejection",
            fabric=fabric,placement=cls.old_forward.plan.source_experiment.placement,
            hbm_address_spaces=old_spaces,
        )
        cls.context=PlacementContext.create(
            producer_pass="moe_ep_train_resident_32MiB_128KiB",fabric=fabric,
            placement=cls.forward.plan.source_experiment.placement,
            hbm_address_spaces=spaces,
        )
        cls.phase=build_moe_full_train_forward_ir0(cls.dense,cls.sequence)
        cls.placement=build_moe_full_train_ep_placement(
            cls.phase, original_dense=cls.dense,dense_manifest=cls.manifest,
            sequence=cls.sequence,context=cls.context,
        )

    def test_real_two_axis_group_routes_and_exact_model_homes(self):
        placement=self.placement
        placement.validate(self.phase,self.dense,self.manifest,
                           self.sequence,self.context)
        self.assertEqual(placement.physical_group.logical_shape,(1,2))
        self.assertEqual([(p.rank,p.die_id,p.logical_coord) for p in
                          placement.physical_group.placements],
                         [(0,0,(0,0)),(1,1,(0,1))])
        self.assertEqual(len(placement.physical_group.embedding.routes),2)
        self.assertEqual((len(placement.hbm_layout.shared),
                          len(placement.hbm_layout.ep)),(11,16))
        self.assertEqual(len(placement.persistent_state_manifest.declarations),27)
        self.assertEqual(len(placement.persistent_state_manifest.bindings),27)
        self.assertEqual({die:sum(binding.die_id==die for binding in
                          placement.persistent_state_manifest.bindings)
                          for die in (0,1)},{0:19,1:8})
        homes=placement.hbm_layout.ep
        for layer in (0,1):
            for die in (0,1):
                expert=sorted((home for home in homes if home.source_layer==layer
                               and home.die_id==die and ".expert" in
                               home.source_parameter_name),
                              key=lambda home:home.slice_offset)
                self.assertEqual([h.slice_offset for h in expert],[0,64,128])
                self.assertEqual([h.tensor_size for h in expert],[64,64,64])
                self.assertEqual({h.original_leaf_address for h in expert},{0})
                base=self.context.hbm_address_spaces[die].base_address
                self.assertEqual([h.physical_address-base for h in expert],
                                 [_HBM_REGION_SHIFT+layer*_HBM_MOE_LAYER_STRIDE,
                                  _HBM_REGION_SHIFT+layer*_HBM_MOE_LAYER_STRIDE+64,
                                  _HBM_REGION_SHIFT+layer*_HBM_MOE_LAYER_STRIDE+128])
                router=next(h for h in homes if h.source_layer==layer
                            and h.die_id==die and ".router.weight" in
                            h.source_parameter_name)
                self.assertEqual(router.physical_address-base,
                                 _HBM_REGION_SHIFT+layer*_HBM_MOE_LAYER_STRIDE+4096)
        self.assertEqual(self.context.hbm_address_spaces[1].base_address,1<<25)
        self.assertNotEqual(self.placement.source_workload_case_ref,
                            self.old_workload_case_ref)
        self.assertNotEqual(self.placement.physical_case_id,
                            self.placement.source_workload_case_ref)

    def test_actual_four_train_namespace_relocates_stateabi_and_hbm_symbols(self):
        for unit in self.sequence.units:
            named=namespace_moe_training_unit(
                unit,timeline_source_id=self.phase.graph.id,
                hbm_homes=self.context.hbm_address_spaces,
                sram_capacity_bytes=131072,
            )
            actual={(state.die_id,state.address,state.size_bytes)
                    for fragment in named.fragments for state in fragment.state_abi}
            expected={(home.die_id,home.physical_address,
                       home.original_leaf_size)
                      for home in self.placement.hbm_layout.ep
                      if home.source_layer==unit.layer and home.slice_offset==0}
            with self.subTest(step=unit.step,layer=unit.layer):
                self.assertEqual(len(actual),4)
                self.assertEqual(actual,expected)
                bindings={binding.state_abi_id for binding in named.state_bindings}
                self.assertTrue(bindings.issubset({abi.id
                    for fragment in named.fragments for abi in fragment.state_abi}))
                self.assertTrue(bindings)

    def test_original_16mib_64kib_case_rejects_real_moe_sram_region(self):
        with self.assertRaisesRegex(SchemaError,"physical SRAM region peak"):
            build_moe_full_train_ep_placement(
                self.old_phase,
                original_dense=self.old_forward.plan.forward_graph,
                dense_manifest=self.old_forward.linked_forward.manifest,
                sequence=self.old_sequence,context=self.old_context_64,
            )

    def test_original_16mib_hbm_rejects_existing_real_namespace_rebase(self):
        with self.assertRaisesRegex(SchemaError,"physical HBM"):
            build_moe_full_train_ep_placement(
                self.old_phase,
                original_dense=self.old_forward.plan.forward_graph,
                dense_manifest=self.old_forward.linked_forward.manifest,
                sequence=self.old_sequence,context=self.old_context_128,
            )

    def test_old_shared_dense_home_must_not_be_discarded(self):
        bad=replace(self.placement.hbm_layout,
                    shared=self.placement.hbm_layout.shared[1:])
        with self.assertRaisesRegex(SchemaError,"shared\\+EP exact"):
            bad.validate(self.phase,self.manifest,self.sequence)

    def test_layer1_expert_reusing_layer0_hbm_home_is_rejected(self):
        ep=list(self.placement.hbm_layout.ep)
        first=next(home for home in ep if home.source_layer==0
                   and home.die_id==0 and home.slice_offset==0
                   and ".expert" in home.source_parameter_name)
        index=next(i for i,home in enumerate(ep) if home.source_layer==1
                   and home.die_id==0 and home.slice_offset==0
                   and ".expert" in home.source_parameter_name)
        ep[index]=replace(ep[index],physical_address=first.physical_address)
        with self.assertRaisesRegex(SchemaError,"physical HBM homes overlap"):
            replace(self.placement.hbm_layout,ep=tuple(ep)).validate(
                self.phase,self.manifest,self.sequence,
            )

    def test_expert_projection_cannot_hide_missing_64b_slice(self):
        ep=list(self.placement.hbm_layout.ep)
        index=next(i for i,home in enumerate(ep) if home.slice_offset==64)
        ep[index]=replace(ep[index],slice_offset=0)
        with self.assertRaisesRegex(SchemaError,"three full expert projections"):
            replace(self.placement.hbm_layout,ep=tuple(ep)).validate(
                self.phase,self.manifest,self.sequence,
            )

    def test_owner_rank_and_production_abi_state_ref_are_exact(self):
        ep=list(self.placement.hbm_layout.ep)
        index=next(i for i,home in enumerate(ep) if home.die_id==1)
        ep[index]=replace(ep[index],original_abi_ref="other_layer_fake_state")
        with self.assertRaisesRegex(SchemaError,"leaf group parameter StateABI"):
            replace(self.placement.hbm_layout,ep=tuple(ep)).validate(
                self.phase,self.manifest,self.sequence,
            )

    def test_limited_hbm_capacity_rejects_router_and_experts(self):
        spaces=self.context.hbm_address_spaces
        tiny=HbmAddressSpace.create(die_id=1,
                                    base_address=spaces[1].base_address,
                                    size_bytes=4096,alignment_bytes=64)
        short=replace(self.context,
                      hbm_address_spaces=(spaces[0],tiny))
        # Its source digest is old; a public forged context must fail first.
        with self.assertRaisesRegex(SchemaError,"unstable artifact id"):
            build_moe_full_train_ep_placement(
                self.phase,original_dense=self.dense,
                dense_manifest=self.manifest,sequence=self.sequence,
                context=short,
            )
        signed=PlacementContext.create(
            producer_pass="moe_ep_train_source_homes",fabric=self.context.fabric,
            placement=self.context.placement,
            hbm_address_spaces=(spaces[0],tiny),
        )
        with self.assertRaisesRegex(SchemaError,"preflight HBM capacities"):
            build_moe_full_train_ep_placement(
                self.phase,original_dense=self.dense,
                dense_manifest=self.manifest,sequence=self.sequence,
                context=signed,
            )

    def test_source_fabric_route_cannot_be_dropped(self):
        group=self.placement.physical_group
        shorter=replace(group,embedding=replace(group.embedding,
                    routes=group.embedding.routes[:1]))
        with self.assertRaisesRegex(SchemaError,"rank.*routes.*fabric"):
            replace(self.placement,physical_group=shorter).validate(
                self.phase,self.dense,self.manifest,
                self.sequence,self.context,
            )


if __name__=="__main__":
    unittest.main()
