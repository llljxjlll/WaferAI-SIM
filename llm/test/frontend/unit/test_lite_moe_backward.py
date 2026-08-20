from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_moe_backward import (
    build_lite_moe_backward_contract,
    build_lite_moe_backward_oracle,
    build_lite_moe_backward_overlay,
    validate_lite_moe_backward_overlay,
)
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.lite_moe_backward import (
    LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION,
    S3_LITE_MOE_BACKWARD_CASE_ID,
    LiteMoeBackwardContract,
    LiteMoeBackwardCoverage,
    LiteMoeBackwardOverlay,
    LiteMoeRemoteGradDte,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    HbmBinding,
    PersistentStateAccess,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.test.frontend.integration.lite_moe_cases import (
    build_lite_moe_execution_case,
)


class LiteMoeBackwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_lite_moe_execution_case()
        cls.trace = cls.case.source.moe_spec.trace
        cls.overlay = build_lite_moe_backward_overlay(
            cls.case.n4,
            cls.case.projection,
            cls.case.schedule,
            cls.case.global_dag,
            cls.case.n6_intent,
            cls.trace,
        )

    def _validate(self, overlay: LiteMoeBackwardOverlay) -> None:
        validate_lite_moe_backward_overlay(
            overlay,
            self.case.n4,
            self.case.projection,
            self.case.schedule,
            self.case.global_dag,
            self.case.n6_intent,
            self.trace,
        )

    def test_exact_contract_oracle_and_overlay_counts(self) -> None:
        result = self.overlay
        self.assertEqual(result.contract.case_id, S3_LITE_MOE_BACKWARD_CASE_ID)
        self.assertEqual(result.contract.expert_histogram, (2, 2, 2, 2))
        self.assertEqual(
            (
                result.oracle.remote_grad_count,
                result.oracle.remote_grad_bytes_total,
                result.oracle.token_wgrad_count,
                result.oracle.token_wgrad_bytes_total,
                result.oracle.expert_reduce_count,
                result.oracle.sgd_store_count,
            ),
            (4, 128, 8, 16384, 4, 4),
        )
        self.assertEqual(
            tuple(item.token_index for item in result.remote_grad_dtes),
            (1, 3, 4, 6),
        )
        self.assertEqual(
            tuple((item.source_die_id, item.destination_die_id) for item in result.remote_grad_dtes),
            ((1, 0), (1, 0), (0, 1), (0, 1)),
        )
        self.assertEqual(
            tuple(
                (
                    item.declaration.identity.kind.value,
                    item.declaration.access.value,
                    item.binding.die_id,
                    item.binding.size_bytes,
                )
                for item in result.trainable_down_states
            ),
            (
                ("trainable_parameter", "read_write", 0, 1024),
                ("trainable_parameter", "read_write", 0, 1024),
                ("trainable_parameter", "read_write", 1, 1024),
                ("trainable_parameter", "read_write", 1, 1024),
            ),
        )
        self.assertEqual(
            tuple((item.expert_index, item.offset_bytes) for item in result.token_wgrads),
            ((0, 0), (0, 2048), (1, 0), (1, 2048), (2, 0), (2, 2048), (3, 0), (3, 2048)),
        )
        self.assertTrue(all(item.input_span_bytes == 4096 for item in result.expert_reduces))
        self.assertTrue(all(item.output_alias_ref == item.contribution_refs[0] for item in result.expert_reduces))
        self.assertTrue(
            all(
                (item.weight_read_bytes, item.gradient_read_bytes, item.state_store_bytes)
                == (1024, 2048, 1024)
                for item in result.sgd_stores
            )
        )
        self._validate(result)

    def test_strict_serde_stable_id_and_determinism(self) -> None:
        rebuilt = build_lite_moe_backward_overlay(
            self.case.n4,
            self.case.projection,
            self.case.schedule,
            self.case.global_dag,
            self.case.n6_intent,
            self.trace,
        )
        self.assertEqual(rebuilt, self.overlay)
        self.assertEqual(
            loads_dataclass(
                LiteMoeBackwardOverlay,
                canonical_json(self.overlay),
                path="overlay",
            ),
            self.overlay,
        )
        raw = json.loads(canonical_json(self.overlay))
        raw["unexpected"] = 1
        with self.assertRaises(SchemaError):
            loads_dataclass(LiteMoeBackwardOverlay, json.dumps(raw), path="overlay")
        raw = json.loads(canonical_json(self.overlay))
        raw["schema_version"] = "wafer_frontend.s3_lite_moe_backward_overlay/v0"
        with self.assertRaises(SchemaError):
            loads_dataclass(LiteMoeBackwardOverlay, json.dumps(raw), path="overlay")

    def test_contract_and_oracle_fail_closed(self) -> None:
        contract = build_lite_moe_backward_contract(self.trace)
        oracle = build_lite_moe_backward_oracle(contract)
        with self.assertRaises(SchemaError):
            replace(
                contract,
                coverage=LiteMoeBackwardCoverage.FULL_EXPERT,
            ).validate()
        with self.assertRaises(SchemaError):
            replace(contract, expert_histogram=(1, 3, 2, 2)).validate()
        with self.assertRaises(SchemaError):
            replace(oracle, token_wgrad_bytes_total=16383).validate_against(contract)

    def test_remote_and_alias_lineage_fail_closed(self) -> None:
        reordered = self.overlay.remote_grad_dtes[::-1]
        with self.assertRaises(SchemaError):
            replace(self.overlay, remote_grad_dtes=reordered).validate()
        remote = self.overlay.remote_grad_dtes[0]
        semantic = {
            name: getattr(remote, name)
            for name in remote.__dataclass_fields__
            if name != "id"
        }
        semantic["forward_flow_ref"] = self.overlay.remote_grad_dtes[1].forward_flow_ref
        wrong_remote = LiteMoeRemoteGradDte(
            stable_artifact_id(
                "s3_lite_moe_remote_grad_dte",
                semantic,
                schema_version=LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        with self.assertRaises(SchemaError):
            LiteMoeBackwardOverlay.create(
                **{
                    **self.overlay._semantic(),
                    "remote_grad_dtes": (
                        (wrong_remote,) + self.overlay.remote_grad_dtes[1:]
                    ),
                }
            )
        with self.assertRaises(SchemaError):
            replace(
                self.overlay,
                token_wgrads=(
                    replace(self.overlay.token_wgrads[1], deps=()),
                    *self.overlay.token_wgrads[1:],
                ),
            ).validate()
        with self.assertRaises(SchemaError):
            replace(
                self.overlay.expert_reduces[0],
                output_alias_ref=self.overlay.expert_reduces[0].contribution_refs[1],
            ).validate("reduce")

    def test_source_provenance_is_recomputed(self) -> None:
        tampered = LiteMoeBackwardOverlay.create(
            **{
                **self.overlay._semantic(),
                "source_global_id": "lite_moe_global_tampered",
            }
        )
        with self.assertRaises(SchemaError):
            self._validate(tampered)

    def test_trainable_permission_address_and_owner_tamper(self) -> None:
        state = self.overlay.trainable_down_states[0]
        with self.assertRaises(SchemaError):
            replace(
                state,
                declaration=replace(
                    state.declaration,
                    access=PersistentStateAccess.READ_ONLY,
                ),
            ).validate("state")
        wrong_binding = HbmBinding.create(
            state_ref=state.declaration.id,
            die_id=state.binding.die_id,
            address=state.binding.address + 64,
            size_bytes=state.binding.size_bytes,
        )
        with self.assertRaises(SchemaError):
            replace(
                self.overlay,
                trainable_down_states=(
                    replace(state, binding=wrong_binding),
                    *self.overlay.trainable_down_states[1:],
                ),
            ).validate()
        with self.assertRaises(SchemaError):
            replace(state, home_die_id=1).validate("state")


if __name__ == "__main__":
    unittest.main()
