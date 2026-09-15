"""Real H16/L2 Dense AdamW source-to-linked coverage and fused slices."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.passes.dense_adamw_compile_sequence import (
    compile_dense_adamw_step,
)
from llm.frontend.wafer_frontend.passes.dense_training_compile_sequence import (
    compile_dense_training_sequence,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data, physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode, SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryTier, MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadOptimizerKind, WorkloadOptimizerSpec, WorkloadRunRequest,
)
from llm.test.frontend.integration.run_dense_training_sequence_runtime_canary import (
    _offload_sequence,
)
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec
from llm.test.frontend.unit.test_workload_materialization import _capability


def _adamw_case():
    base = _offload_sequence().materialization.request
    raw = _hardware(1, 1)
    fabric = physical_fabric_from_data(raw)
    spaces = hbm_address_spaces_from_data(raw)
    legacy = _spec(1, 1)
    legacy = replace(
        legacy,
        model=replace(
            legacy.model, V=2, H=16, I=1, L=2, NH=1,
            KVH=1, DH=16, rotary_dim=16,
        ),
    )
    model = replace(base.model, hidden_size=16, head_dim=16)
    optimizer = WorkloadOptimizerSpec(
        WorkloadOptimizerKind.ADAMW, 0.001, weight_decay=0.01,
        beta1=0.9, beta2=0.999, epsilon=1e-8,
    )
    capacity = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM, location_ref=f"die:{space.die_id}",
            base_address=space.base_address,
            capacity_bytes=space.size_bytes,
            alignment_bytes=space.alignment_bytes,
        ) for space in spaces
    )
    def request(opt):
        return WorkloadRunRequest.create(
            family=base.family, model=model, steps=base.steps,
            mesh=base.mesh, parallel=base.parallel, optimizer=opt,
        )
    adamw = materialize_workload_preflight(
        request(optimizer), _capability(supported=True), capacities=capacity,
    )
    sgd = materialize_workload_preflight(
        request(WorkloadOptimizerSpec(WorkloadOptimizerKind.SGD, 0.001)),
        _capability(supported=True), capacities=capacity,
    )
    physical = compile_dense_training_sequence(
        sgd, legacy, fabric, hbm_address_spaces=spaces,
    ).segments[0].linked_program
    return adamw, physical


class DenseAdamwCompileSequenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adamw, cls.physical = _adamw_case()

    def test_real_source_17_updates_with_68_optimizer_states(self):
        linked = compile_dense_adamw_step(self.adamw, self.physical, 0)
        fragment = linked.manifest.fragments[0]
        records = fragment.core_streams[0].records
        self.assertEqual(sum(r.opcode is RecordOpcode.ADAMW_UPDATE for r in records), 17)
        self.assertEqual(sum(r.opcode is RecordOpcode.SGD_UPDATE for r in records), 0)
        self.assertEqual(sum(r.opcode is RecordOpcode.LSU_LOAD for r in records), 83)
        self.assertEqual(sum(r.opcode is RecordOpcode.LSU_STORE for r in records), 83)
        self.assertEqual(len(fragment.state_abi), 83)
        self.assertLess(
            max(a.region_offset_bytes + a.size_bytes for a in fragment.buffer_abi),
            65536,
        )

    def test_gate_up_slices_no_overlap_and_step_versions(self):
        for step_index in (0, 1):
            linked = compile_dense_adamw_step(self.adamw, self.physical, step_index)
            fragment = linked.manifest.fragments[0]
            updates = {
                record.source_global_action_id: (index, record)
                for index, record in enumerate(fragment.core_streams[0].records)
                if record.opcode is RecordOpcode.ADAMW_UPDATE
            }
            logical = {
                op.id: op for op in self.adamw.logical_graph.operations
                if op.kind.value == "adamw_update" and op.step == step_index
            }
            self.assertEqual(set(updates), set(logical))
            self.assertEqual(
                {record.operands[17].literal_value for _index, record in updates.values()},
                {step_index + 1},
            )
            relocs = {
                (r.record_index, r.operand_id): r
                for r in fragment.core_streams[0].address_relocations
            }
            for layer in (0, 1):
                pairs = tuple(
                    (index, record) for action, (index, record) in updates.items()
                    if logical[action].parameter_ref in (
                        f"layer.{layer}.mlp_gate.weight",
                        f"layer.{layer}.mlp_up.weight",
                    )
                )
                self.assertEqual(len(pairs), 2)
                weights = tuple(
                    (relocs[(index, SemanticOperandId.COMPUTE_INPUT_ADDRESS)].symbol_ref,
                     relocs[(index, SemanticOperandId.COMPUTE_INPUT_ADDRESS)].addend,
                     record.operands[16].literal_value * 2)
                    for index, record in pairs
                )
                self.assertEqual(weights[0][0], weights[1][0])
                intervals = sorted((offset, offset + length) for _ref, offset, length in weights)
                self.assertEqual(intervals, [(0, 32), (32, 64)])


if __name__ == "__main__":
    unittest.main()
