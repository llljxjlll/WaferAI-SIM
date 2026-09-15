"""Source-backed physical HBM home proposal for full two-layer EP2 training.

This does not relocate a linked fragment.  The executable linker must prove
its rewritten StateABI, state refs, LSU operands and segment versions match
these disjoint homes before any full-model TRAIN runtime claim is possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..lowering.moe_full_training_namespace import (
    _HBM_REGION_SHIFT,_HBM_MOE_LAYER_STRIDE,
)
from ..schema.artifact_manifest import LinkedProgramManifest
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.persistent_state import HbmAddressSpace
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase


@dataclass(frozen=True,slots=True)
class MoeFullTrainParameterHome:
    declaration_ref: str
    original_e2e_state_ref: str | None
    original_abi_ref: str
    source_layer: int | None
    source_parameter_name: str
    die_id: int
    original_leaf_address: int
    original_leaf_size: int
    physical_address: int
    tensor_size: int
    slice_offset: int
    source_space_ref: str

    def validate(self,*,space: HbmAddressSpace) -> None:
        if (self.die_id != space.die_id or self.source_space_ref != space.id
                or self.tensor_size < 1 or self.original_leaf_size < self.tensor_size
                or self.slice_offset < 0
                or self.slice_offset+self.tensor_size > self.original_leaf_size
                or self.physical_address < space.base_address
                or self.physical_address+self.tensor_size
                    > space.base_address+space.size_bytes
                or self.original_leaf_address < 0):
            raise SchemaError("parameter slice exceeds true source ABI or physical HBM",
                              path=f"moe_full_train_home[{self.declaration_ref}]")


@dataclass(frozen=True,slots=True)
class MoeFullTrainHbmLayout:
    source_ir0_ref: str
    dense_forward_manifest_ref: str
    source_sequence_ref: str
    source_spaces: tuple[HbmAddressSpace,...]
    shared: tuple[MoeFullTrainParameterHome,...]
    ep: tuple[MoeFullTrainParameterHome,...]

    def validate(self,phase: FullMoeForwardIr0Phase,
                 dense_manifest: LinkedProgramManifest,
                 sequence: MoeCompileSequence) -> None:
        if (self.source_ir0_ref != phase.graph.id
                or self.dense_forward_manifest_ref != dense_manifest.id
                or self.source_sequence_ref != sequence.id
                or len(self.shared) != 11 or len(self.ep) != 16
                or {home.declaration_ref for home in (*self.shared,*self.ep)}
                    != {state.id for state in phase.graph.persistent_states}):
            raise SchemaError("shared+EP exact StateDecl/production source map lost",
                              path="moe_full_train_hbm_layout.source")
        spaces = {space.die_id:space for space in self.source_spaces}
        if set(spaces) != {0,1} or any(space.die_id != die
                                    for die,space in spaces.items()):
            raise SchemaError("production EP2 requires exact two physical address spaces",
                              path="moe_full_train_hbm_layout.spaces")
        states = {state.id:state for state in phase.graph.persistent_states}
        for home in (*self.shared,*self.ep):
            home.validate(space=spaces[home.die_id])
            state = states[home.declaration_ref]
            if (state.tensor_bytes != home.tensor_size
                    or state.identity.tensor_ref != home.source_parameter_name):
                raise SchemaError("source parameter tensor work shrunk or renamed",
                                  path=f"moe_full_train_hbm_layout.state[{home.declaration_ref}]")
        originals = dict(phase.shared_source_state_refs)
        dense_abi = {abi.state_ref:abi for fragment in dense_manifest.fragments
                     for abi in fragment.state_abi if abi.state_ref in originals}
        if set(dense_abi) != set(originals):
            raise SchemaError("production Dense shared StateABI incomplete",
                              path="moe_full_train_hbm_layout.shared")
        for home in self.shared:
            original = next(old for old,new in originals.items()
                            if new == home.declaration_ref)
            abi = dense_abi[original]
            if (home.original_abi_ref != abi.state_ref
                    or home.die_id != abi.die_id
                    or home.physical_address != abi.address
                    or home.original_leaf_address != abi.address
                    or home.original_leaf_size != abi.size_bytes):
                raise SchemaError("shared Dense home no longer matches production ABI",
                                  path=f"moe_full_train_hbm_layout.shared[{original}]")
        owners = {owner.source_state_decl_ref:owner
                  for owner in phase.ep_state_owners}
        for home in self.ep:
            owner = owners[home.declaration_ref]
            if (home.original_e2e_state_ref != owner.source_e2e_state_version0_ref
                    or home.source_parameter_name !=
                       states[home.declaration_ref].identity.tensor_ref
                    or home.die_id != owner.ep_owner):
                raise SchemaError("real E2E EP tensor owner lost in HBM placement",
                                  path=f"moe_full_train_hbm_layout.ep[{home.declaration_ref}]")
            unit = next(unit for unit in sequence.units
                        if (unit.step,unit.layer) == (0,home.source_layer))
            group = next(group for group in unit.parameter_bindings
                         if owner.source_e2e_parameter_name in group.parameter_refs)
            source = [abi for fragment in unit.linked_manifest.fragments
                      for abi in fragment.state_abi
                      if abi.state_ref == home.original_abi_ref
                      and abi.die_id == home.die_id]
            if (len(source) != 1
                    or home.original_abi_ref not in group.production_parameter_state_refs
                    or home.original_leaf_address != source[0].address
                    or home.original_leaf_size != source[0].size_bytes):
                raise SchemaError("leaf group parameter StateABI identity/extent drifted",
                                  path=f"moe_full_train_hbm_layout.ep[{home.declaration_ref}]")
            if group.expert is None:
                if home.slice_offset != 0 or home.original_leaf_size != 16:
                    raise SchemaError("one full router replica tensor must be 16B",
                                      path=f"moe_full_train_hbm_layout.ep[{home.declaration_ref}]")
            else:
                projection = owner.source_e2e_parameter_name.rsplit('.',2)[-2]
                order = {"gate":0,"up":1,"down":2}
                if (home.slice_offset != order[projection]*64
                        or home.original_leaf_size != 192
                        or home.tensor_size != 64):
                    raise SchemaError("all three full expert projections require exact 192B ABI",
                                      path=f"moe_full_train_hbm_layout.ep[{home.declaration_ref}]")
        by_die: dict[int,list[MoeFullTrainParameterHome]] = {0:[],1:[]}
        for home in (*self.shared,*self.ep):
            by_die[home.die_id].append(home)
        for die,homes in by_die.items():
            ordered = sorted(homes,key=lambda home:home.physical_address)
            for before,after in zip(ordered,ordered[1:]):
                if before.physical_address+before.tensor_size > after.physical_address:
                    raise SchemaError("Dense/layer/expert/router physical HBM homes overlap",
                                      path=f"moe_full_train_hbm_layout.die{die}")


def build_moe_full_train_hbm_layout(
    phase: FullMoeForwardIr0Phase, *, original_dense,
    dense_manifest: LinkedProgramManifest,
    sequence: MoeCompileSequence,
    spaces: tuple[HbmAddressSpace,...],
) -> MoeFullTrainHbmLayout:
    """Plan each source owner onto disjoint physical addresses, never rewrite ABI."""
    phase.validate_against(original_dense,sequence)
    dense_manifest.validate()
    for space in spaces:
        space.validate()
    catalog = {space.die_id:space for space in spaces}
    if set(catalog) != {0,1} or len(spaces) != 2:
        raise SchemaError("true EP2 placement needs exactly two address spaces",
                          path="moe_full_train_hbm_layout.spaces")
    states = {state.id:state for state in phase.graph.persistent_states}
    originals = dict(phase.shared_source_state_refs)
    abi_by_ref = {abi.state_ref:abi for fragment in dense_manifest.fragments
                  for abi in fragment.state_abi if abi.state_ref in originals}
    if set(abi_by_ref) != set(originals):
        raise SchemaError("Dense shared parameter physical StateABI missing",
                          path="moe_full_train_hbm_layout.dense")
    shared = tuple(MoeFullTrainParameterHome(
        new,None,old,None,states[new].identity.tensor_ref,abi_by_ref[old].die_id,
        abi_by_ref[old].address,abi_by_ref[old].size_bytes,
        abi_by_ref[old].address,states[new].tensor_bytes,0,
        catalog[abi_by_ref[old].die_id].id,
    ) for old,new in phase.shared_source_state_refs)
    ep = []
    for owner in phase.ep_state_owners:
        state = states[owner.source_state_decl_ref]
        origin = next(origin for origin in
                      sequence.materialization.logical_graph.state_versions
                      if origin.id == owner.source_e2e_state_version0_ref)
        layer = origin.layer
        unit = next(unit for unit in sequence.units
                    if (unit.step,unit.layer)==(0,layer))
        group = next(group for group in unit.parameter_bindings
                     if owner.source_e2e_parameter_name in group.parameter_refs)
        matched = [abi for fragment in unit.linked_manifest.fragments
                   for abi in fragment.state_abi
                   if abi.state_ref in group.production_parameter_state_refs
                   and abi.die_id==owner.ep_owner]
        if len(matched) != 1:
            raise SchemaError("true E2E expert/router ABI per Die missing or duplicated",
                              path=f"moe_full_train_hbm_layout.layer{layer}")
        abi = matched[0]
        if group.expert is None:
            offset = 0
            local = _HBM_REGION_SHIFT+layer*_HBM_MOE_LAYER_STRIDE+abi.address
        else:
            projection = owner.source_e2e_parameter_name.rsplit('.',2)[-2]
            offset = {"gate":0,"up":1,"down":2}[projection]*64
            local = _HBM_REGION_SHIFT+layer*_HBM_MOE_LAYER_STRIDE+abi.address
        ep.append(MoeFullTrainParameterHome(
            state.id,origin.id,abi.state_ref,layer,
            state.identity.tensor_ref,owner.ep_owner,abi.address,abi.size_bytes,
            catalog[owner.ep_owner].base_address+local+offset,
            state.tensor_bytes,offset,catalog[owner.ep_owner].id,
        ))
    result = MoeFullTrainHbmLayout(
        phase.graph.id,dense_manifest.id,sequence.id,spaces,shared,tuple(ep),
    )
    result.validate(phase,dense_manifest,sequence)
    return result


__all__ = ["MoeFullTrainParameterHome","MoeFullTrainHbmLayout",
           "build_moe_full_train_hbm_layout"]
