from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import unittest

from test_train_forward_global_action import _global_action

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import lower_train as public_lower_train
from llm.frontend.wafer_frontend.passes.train_lower_program import lower_train
from llm.frontend.wafer_frontend.schema import (
    TrainLoweredProgram as PublicTrainLoweredProgram,
    TrainLoweredReplica as PublicTrainLoweredReplica,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    RecordOpcode,
    RegionManifest,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragment
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.train_n6 import (
    TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
    TrainLoweredProgram,
    TrainLoweredReplica,
)


_RECORD_COUNTS = Counter(
    {
        RecordOpcode.SRAM_ALLOC_AT: 134,
        RecordOpcode.SRAM_FREE: 134,
        RecordOpcode.SRAM_BIND: 60,
        RecordOpcode.LSU_LOAD: 30,
        RecordOpcode.MATMUL: 26,
        RecordOpcode.DTE_SEND: 16,
        RecordOpcode.DTE_RECV: 16,
        RecordOpcode.DTE_WAIT: 16,
        RecordOpcode.RMSNORM: 10,
        RecordOpcode.LOCAL_REDUCE: 8,
        RecordOpcode.RESIDUAL: 8,
        RecordOpcode.DTE_ISSUE: 8,
        RecordOpcode.EVENT_WAIT: 8,
        RecordOpcode.EVENT_SET: 8,
        RecordOpcode.ROPE_QK_EXACT: 4,
        RecordOpcode.SWIGLU: 4,
        RecordOpcode.ATTENTION_EXACT: 4,
        RecordOpcode.EMBEDDING_LOOKUP: 2,
        RecordOpcode.CROSS_ENTROPY_FORWARD: 2,
    }
)


class TrainForwardN6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.source = _global_action()
        cls.result = lower_train(cls.source)

    def test_dp2_tp2_exact_fragment_record_relocation_and_ce_goldens(
        self,
    ) -> None:
        result = self.result
        self.assertIs(public_lower_train, lower_train)
        self.assertIs(PublicTrainLoweredProgram, TrainLoweredProgram)
        self.assertIs(PublicTrainLoweredReplica, TrainLoweredReplica)
        result.validate_against(self.source)
        self.assertEqual(
            result.schema_version,
            "wafer_frontend.train_lowered_program/v1alpha2",
        )
        self.assertEqual(
            result.schema_version,
            TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
        )
        self.assertEqual(result.dp_degree, 2)
        self.assertEqual(len(result.replicas), 2)

        for replica in result.replicas:
            leaves = tuple(
                _leaf_fragment(fragment) for fragment in replica.fragments
            )
            records = tuple(
                record
                for leaf in leaves
                for stream in leaf.core_streams
                for record in stream.records
            )
            self.assertEqual(
                (
                    len(replica.lowering_context.global_dag.actions),
                    len(replica.fragments),
                    Counter(type(item) for item in replica.fragments),
                    Counter(leaf.kind for leaf in leaves),
                    len(records),
                    sum(
                        len(stream.address_relocations)
                        for leaf in leaves
                        for stream in leaf.core_streams
                    ),
                    sum(
                        len(stream.runtime_relocations)
                        for leaf in leaves
                        for stream in leaf.core_streams
                    ),
                    sum(len(leaf.buffer_abi) for leaf in leaves),
                    sum(len(leaf.state_abi) for leaf in leaves),
                ),
                (
                    154,
                    86,
                    Counter({CommandFragment: 78, RegionManifest: 8}),
                    Counter(
                        {
                            FragmentKind.COARSE: 44,
                            FragmentKind.STATE_IO: 30,
                            FragmentKind.ISA_REGION: 8,
                            FragmentKind.STANDALONE_COLLECTIVE: 4,
                        }
                    ),
                    498,
                    826,
                    144,
                    222,
                    30,
                ),
            )
            self.assertEqual(
                Counter(record.opcode for record in records),
                _RECORD_COUNTS,
            )
            self.assertEqual(
                tuple(leaf.id for leaf in leaves),
                tuple(sorted(leaf.id for leaf in leaves)),
            )
            claimed = tuple(
                action_id
                for leaf in leaves
                for action_id in leaf.claimed_action_ids
            )
            self.assertEqual(len(claimed), 154)
            self.assertEqual(
                set(claimed),
                {action.id for action in replica.lowering_context.global_dag.actions},
            )

            ce_records = []
            for leaf in leaves:
                for stream in leaf.core_streams:
                    for record_index, record in enumerate(stream.records):
                        if record.opcode is not RecordOpcode.CROSS_ENTROPY_FORWARD:
                            continue
                        ce_records.append(record)
                        relocations = tuple(
                            relocation
                            for relocation in stream.address_relocations
                            if relocation.record_index == record_index
                        )
                        self.assertEqual(len(relocations), 3)
                        self.assertEqual(
                            {item.operand_id for item in relocations},
                            {
                                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                            },
                        )
                        self.assertFalse(
                            any(
                                item.record_index == record_index
                                for item in stream.runtime_relocations
                            )
                        )
            self.assertEqual(len(ce_records), 2)
            self.assertEqual(
                tuple(
                    (operand.name, operand.literal_value)
                    for operand in ce_records[0].operands
                ),
                (
                    ("logits_datatype", 1),
                    ("label_datatype", 2),
                    ("loss_datatype", 3),
                    ("reduction", 0),
                    ("logits_address", None),
                    ("labels_address", None),
                    ("loss_address", None),
                    ("logical_rows", 8),
                    ("rank_rows", 4),
                    ("tp_degree", 2),
                    ("vocab_size", 32),
                ),
            )

    def test_strict_serde_stable_id_and_version_boundaries(self) -> None:
        result = self.result
        encoded = canonical_json(result)
        decoded = loads_dataclass(
            TrainLoweredProgram,
            encoded,
            path="train_lowered",
        )
        self.assertEqual(decoded, result)
        decoded.validate_against(self.source)
        self.assertEqual(
            TrainLoweredProgram.create(
                source=self.source,
                replicas=result.replicas,
            ),
            result,
        )

        raw = json.loads(encoded)
        raw["unexpected"] = True
        with self.assertRaises(SchemaError):
            loads_dataclass(
                TrainLoweredProgram,
                canonical_json(raw),
                path="train_lowered",
            )
        del raw["unexpected"]
        del raw["source_global_action_carrier_id"]
        with self.assertRaises(SchemaError):
            loads_dataclass(
                TrainLoweredProgram,
                canonical_json(raw),
                path="train_lowered",
            )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                result,
                schema_version=(
                    "wafer_frontend.train_lowered_program/v1alpha0"
                ),
            ).validate()

    def test_replica_order_context_and_fragment_coverage_fail_closed(
        self,
    ) -> None:
        result = self.result
        with self.assertRaisesRegex(SchemaError, "canonical DP order"):
            TrainLoweredProgram.create(
                source=self.source,
                replicas=(result.replicas[1], result.replicas[0]),
            )
        with self.assertRaisesRegex(SchemaError, "cover every executable"):
            replace(
                result.replicas[0],
                fragments=result.replicas[0].fragments[:-1],
            ).validate()
        with self.assertRaisesRegex(SchemaError, "replica context"):
            TrainLoweredReplica.create(
                source=self.source.replicas[1],
                lowering_context=result.replicas[0].lowering_context,
                fragments=result.replicas[0].fragments,
            )
        with self.assertRaises(SchemaError):
            replace(
                result.replicas[0],
                lowering_context=replace(
                    result.replicas[0].lowering_context,
                    global_dag=replace(
                        result.replicas[0].lowering_context.global_dag,
                        source_schedule_set_id="wrong",
                    ),
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "TrainGlobalAction"):
            lower_train(self.source.replicas[0])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
