import json
from collections import Counter
from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily, WorkloadRunRequest, WorkloadMemoryPolicy, WorkloadMemoryMode
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTierCapacity,MemoryTier
from llm.frontend.wafer_frontend.passes.workload_materialization import materialize_workload_preflight
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import compile_moe_full_model_inference_sequence
from llm.frontend.wafer_frontend.passes.moe_inference_paged_compile_sequence import relink_moe_inference_paged_segment,unique_state_abis
from llm.test.frontend.unit.test_moe_compile_sequence import _request,_manifest
from llm.test.frontend.unit.test_moe_full_model_compile_sequence import _legacy_template
from llm.test.frontend.unit.test_workload_materialization import _capability
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
for rows,columns in [(1,4),(4,1),(2,2)]:
 name=f'{rows}x{columns}'; n=rows*columns
 r=_request(WorkloadFamily.MOE_INFERENCE,rows=rows,columns=columns)
 h=tuple(MemoryTierCapacity.create(tier=MemoryTier.HBM,location_ref=f'die:{i}',base_address=i<<30,capacity_bytes=1024,alignment_bytes=16) for i in range(n))
 ext=MemoryTierCapacity.create(tier=MemoryTier.EXTERNAL,location_ref='host:0',base_address=0,capacity_bytes=8192,alignment_bytes=16)
 try:
  materialize_workload_preflight(r,_capability(supported=True),capacities=h)
  print(name,'RESIDENT UNEXPECTED PASS',flush=True)
 except SchemaError as e: print(name,'resident',e.code,str(e),flush=True)
 off=WorkloadRunRequest.create(family=r.family,model=r.model,steps=r.steps,mesh=r.mesh,parallel=r.parallel,memory=WorkloadMemoryPolicy(mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,external_tier_ref='host:0'),optimizer=r.optimizer,execution=r.execution)
 try:
  manifest=materialize_workload_preflight(off,_capability(supported=True),capacities=(ext,*h))
  print(name,'offload PASS',manifest.id,flush=True)
 except Exception as e:
  print(name,'offload FAIL',type(e).__name__,str(e),flush=True);continue
 source=_manifest(WorkloadFamily.MOE_INFERENCE,rows=rows,columns=columns)
 f=physical_fabric_from_data(minimal_hardware(columns,rows,sram_bytes=65536))
 try:
  seq=compile_moe_full_model_inference_sequence(source,_legacy_template(),f,hbm_address_spaces=valid_hbm_address_spaces(f))
  print(name,'COMPILE PASS',len(seq.segments),flush=True)
 except Exception as e:
  print(name,'COMPILE FAIL',type(e).__name__,str(e),flush=True);continue
 for step,seg in enumerate(seq.segments):
  m=seg.executable_manifest; abis=unique_state_abis(m)
  print(name,'step',step,'frags',len(m.fragments),'cores',[(x.runtime_core_id,len(x.records)) for x in m.core_streams],'bindings',len(m.state_operand_bindings),'abis',len(abis),'state',[(i,dict(Counter((a.kind.value,a.size_bytes) for a in abis if a.die_id==i))) for i in range(n)],flush=True)
  try:
   relink_moe_inference_paged_segment(m,step)
   print(name,'RELINK PASS',flush=True)
  except Exception as e:print(name,'RELINK FAIL',type(e).__name__,str(e),flush=True)
