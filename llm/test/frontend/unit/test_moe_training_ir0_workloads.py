"""Source-bound INT32 embedding indices and physical FP32 parameter gradients."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.moe_training_ir0_workloads import (
    EmbeddingTableWgradWorkload, NormGammaWgradWorkload,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    EffectKind, LogicalNode, NodeEffects, OpKind, OpPhase,
)
from llm.frontend.wafer_frontend.passes.moe_train_source_indices import (
    derive_forward_embedding_index_trace,
    require_embedding_trace_matches_source_input,
)
from llm.frontend.wafer_frontend.schema.moe_train_token_input_case import (
    MoeTrainTokenInputCase,
)
from llm.frontend.wafer_frontend.passes.moe_train_token_program_io import (
    build_source_bound_moe_train_token_program_io,
)
from llm.frontend.wafer_frontend.passes.full_training_ce_seed_program_io import (
    build_bounded_seeded_ce_program_io,
)
from llm.test.frontend.integration.run_bounded_dense_seeded_ce_canary import (
    build_case,
)
from llm.test.frontend.unit.test_full_training_timeline_linker import (
    FullTrainingTimelineLinkerTest,
)
from llm.test.frontend.unit.test_full_dense_training_ce_ir0 import _spec
from llm.frontend.wafer_frontend.passes.train_forward import (
    build_train_forward_ir0,
)


class MoeTrainingIr0WorkloadsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_case()
        # Unit source-only fixture; this SHA does not denote a finalized
        # artifact and the test never invokes finalizer/ProgramIO resolver.
        cls.io = build_bounded_seeded_ce_program_io(
            cls.case.profile, cls.case.manifest, cls.case.graft,
            "0" * 64,
        )
        cls.moe = FullTrainingTimelineLinkerTest.moe.materialization
        cls.token_case = MoeTrainTokenInputCase.create(
            cls.moe, tokens_by_step=((3, 3, 5, 7), (3, 3, 5, 7)),
        )

    def test_source_signed_full_v_repeat_updates_io_then_native_decodes(self) -> None:
        updated = build_source_bound_moe_train_token_program_io(
            self.case.manifest, self.io, self.moe,
            self.token_case, step=0, token_value_id="T0.token_ids",
        )
        observed = derive_forward_embedding_index_trace(
            self.case.manifest, updated, token_value_id="T0.token_ids",
            rank_rows=4, vocab_size=16, vocab_start=0,
            vocab_rows=16, hidden_size=4,
        )
        require_embedding_trace_matches_source_input(
            observed, self.moe, self.token_case, step=0,
        )
        self.assertEqual(observed.active_ids, (3, 3, 5, 7))
        self.assertEqual(observed.index_trace[:4], (3, 3, 5, 7))
        self.assertEqual(observed.index_trace[4:], (0,) * 12)
        self.assertEqual(len(updated.initializations), len(self.io.initializations))
        self.assertNotEqual(observed.token_seed_blob_sha256,
                            derive_forward_embedding_index_trace(
                                self.case.manifest, self.io,
                                token_value_id="T0.token_ids", rank_rows=4,
                                vocab_size=16, vocab_start=0,
                                vocab_rows=16, hidden_size=4,
                            ).token_seed_blob_sha256)

    def test_old_io_all_zeros_cannot_impersonate_nonzero_source_case(self) -> None:
        old = derive_forward_embedding_index_trace(
            self.case.manifest, self.io, token_value_id="T0.token_ids",
            rank_rows=4, vocab_size=16, vocab_start=0,
            vocab_rows=16, hidden_size=4,
        )
        with self.assertRaisesRegex(SchemaError, "source-signed case"):
            require_embedding_trace_matches_source_input(
                old, self.moe, self.token_case, step=0,
            )

    def test_changed_public_request_binding_or_vocab_id_fails(self) -> None:
        with self.assertRaisesRegex(SchemaError, "original MoE request"):
            replace(self.token_case, source_case_id="wrong.case").validate_against(
                self.moe,
            )
        with self.assertRaisesRegex(SchemaError, "legal full-V"):
            MoeTrainTokenInputCase.create(
                self.moe, tokens_by_step=((3, 3, 17, 7), (3, 3, 5, 7)),
            )

    def test_actual_borrowed_source_blob_not_fabricated_token_trace(self) -> None:
        source = derive_forward_embedding_index_trace(
            self.case.manifest, self.io, token_value_id="T0.token_ids",
            rank_rows=4, vocab_size=16, vocab_start=0,
            vocab_rows=16, hidden_size=4,
        )
        self.assertEqual(source.active_ids, (0, 0, 0, 0))
        self.assertEqual(source.index_trace, (0,) * 16)
        self.assertEqual(len(source.token_seed_blob_sha256), 64)
        self.assertEqual(source.source_linked_manifest_id, self.case.manifest.id)
        workload = EmbeddingTableWgradWorkload(
            4, 4, 1, 16, 0, 16, 4, source.index_trace,
        )
        workload.validate()
        self.assertEqual((workload.input_index_bytes, workload.upstream_bytes,
                          workload.table_tile_bytes,
                          workload.physical_gradient_bytes),
                         (16, 32, 128, 256))

    def test_different_seed_value_or_vocab_tile_cannot_claim_same_source(self) -> None:
        with self.assertRaisesRegex(SchemaError, "one real borrowed INT32"):
            derive_forward_embedding_index_trace(
                self.case.manifest, self.io, token_value_id="wrong.tokens",
                rank_rows=4, vocab_size=16, vocab_start=0,
                vocab_rows=16, hidden_size=4,
            )
        with self.assertRaisesRegex(SchemaError, "physical index trace"):
            derive_forward_embedding_index_trace(
                self.case.manifest, self.io, token_value_id="T0.token_ids",
                rank_rows=4, vocab_size=16, vocab_start=8,
                vocab_rows=8, hidden_size=4,
            )

    def test_embedding_trace_requires_actual_valid_indices_and_zero_suffix(self) -> None:
        valid = EmbeddingTableWgradWorkload(
            4, 4, 1, 16, 0, 16, 4, (0,) * 16,
        )
        for changed in (
            replace(valid, index_trace=(17, *((0,) * 15))),
            replace(valid, index_trace=(*((0,) * 4), 1, *((0,) * 11))),
            replace(valid, rank_rows=17),
            replace(valid, logical_rows=8),
        ):
            with self.assertRaisesRegex(SchemaError, "physical index trace"):
                changed.validate()

    def test_public_native_op_rejects_dx_alias_or_fake_phase(self) -> None:
        source = build_train_forward_ir0(_spec(1, 1))
        embedding = next(node for node in source.nodes
                         if node.kind is OpKind.EMBEDDING)
        native = replace(
            embedding, id="source_embedding_wgrad", kind=OpKind.EMBEDDING_TABLE_WGRAD,
            phase=OpPhase.WGRAD, impl_ref="embedding_table_wgrad_timing",
            inputs=(*embedding.inputs, "source_backward_hidden"),
            outputs=("source_table_gradient_fp32",),
            workload=EmbeddingTableWgradWorkload(
                4, 4, 1, 16, 0, 16, 4, (3, 3, 5, 7, *((0,) * 12)),
            ),
            effects=NodeEffects(EffectKind.PURE, None, None),
        )
        native.validate("native_embedding")
        for bad in (
            replace(native, phase=OpPhase.DGRAD),
            replace(native, impl_ref="embedding_lookup"),
            replace(native, inputs=embedding.inputs),
        ):
            with self.assertRaisesRegex(SchemaError, "exact operands"):
                bad.validate("native_embedding")
        with self.assertRaisesRegex(SchemaError, "requires one of"):
            replace(native, kind=OpKind.EMBEDDING).validate("native_embedding")

    def test_norm_gamma_wgrad_shape_and_fp32_bytes_not_fp16_dx(self) -> None:
        norm = NormGammaWgradWorkload(4, 4, 1, 4, mode=0)
        norm.validate()
        self.assertEqual((norm.forward_bytes, norm.upstream_bytes,
                          norm.physical_gradient_bytes), (32, 32, 16))
        with self.assertRaisesRegex(SchemaError, "FP16→FP32"):
            replace(norm, output_dtype=norm.input_dtype).validate()
        with self.assertRaisesRegex(SchemaError, "mode invalid"):
            replace(norm, mode=2).validate()
        with self.assertRaisesRegex(SchemaError, "mode invalid"):
            replace(norm, logical_rows=8).validate()


if __name__ == "__main__":
    unittest.main()
