"""Focused physical AdamW five-role ABI and real bounded HBM tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.test.frontend.integration.dense_adamw_native_role_observer import (
    observe_native_adamw_role_values,
)

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_adamw_compile_sequence import compile_dense_adamw_step
from llm.frontend.wafer_frontend.passes.dense_adamw_mid_program_residency import (
    assign_dense_adamw_bounded_slots, derive_dense_adamw_mid_program_residency,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_paged_compile_sequence import (
    compile_dense_adamw_paged_step,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_paged_runtime import (
    build_dense_adamw_paged_runtime,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_physical_authority import (
    prove_physical_adamw_authority_and_capacity,
)
from llm.test.frontend.integration.run_dense_adamw_dma_component_canary import (
    build_source_dma_program,
)


class DenseAdamwPhysicalAuthorityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resident, cls.physical, cls.window, cls.program, _ = build_source_dma_program()
        cls.original = tuple(
            compile_dense_adamw_step(cls.window.materialization, cls.physical, step)
            for step in (0, 1)
        )
        schedule = derive_dense_adamw_mid_program_residency(cls.window, cls.original)
        slots = assign_dense_adamw_bounded_slots(cls.window, schedule)
        linked = tuple(
            compile_dense_adamw_paged_step(cls.window, cls.physical, step, slots)
            for step in (0, 1)
        )
        cls.paged = build_dense_adamw_paged_runtime(
            cls.window, schedule, slots, linked, cls.program,
        )

    def witness(self, paged=None):
        return prove_physical_adamw_authority_and_capacity(
            self.resident, self.window, self.original[0],
            paged or self.paged, self.program,
        )

    def test_real_83_abi_m_v_and_original_pinned_capacity_rejection(self):
        witness = self.witness()
        roles = {item.kind: item for item in witness.roles}
        self.assertEqual(len(witness.states), 83)
        self.assertEqual(len({item.linked_state_abi_id for item in witness.states}), 83)
        self.assertEqual(sum(item.size_bytes for item in witness.states), 32100)
        self.assertEqual(
            {name: (item.state_count, item.size_bytes)
             for name, item in roles.items()},
            {
                "trainable_parameter": (15, 4576),
                "optimizer_master": (17, 9152),
                "optimizer_moment1": (17, 9152),
                "optimizer_moment2": (17, 9152),
                "optimizer_step": (17, 68),
            },
        )
        self.assertEqual(witness.hbm_capacity_bytes, 36864)
        self.assertEqual(witness.workspace_bytes, 9248)
        self.assertEqual(witness.resident_rejection_code, "memory_capacity_exceeded")
        self.assertIn("capacity exceeded", witness.resident_rejection)
        self.assertTrue(all(
            item.seed_digest != "0" * 64 for item in witness.states
        ))
        self.assertNotEqual(roles["optimizer_moment1"].seed_digest,
                            roles["optimizer_moment2"].seed_digest)

    def test_native_external_m_v_byte_probe_rejects_forged_value(self):
        witness = self.witness()
        stdout = []
        for version in (0, 1, 2):
            if version:
                stdout.append(f"[DENSE_ADAMW_PAGED_DMA_EVENT] index={version * 166 - 1} step={version-1}")
            for item in witness.roles:
                stdout.append(
                    f"[DENSE_ADAMW_EXTERNAL_ROLE_VALUE] role={item.kind} "
                    f"version={version} bytes={item.size_bytes} "
                    f"digest={item.seed_digest} state_count={item.state_count} "
                    "pending=0 functional=0 pass=1"
                )
            stdout.append(f"[DENSE_TRAINING_SEQUENCE_STATE] version={version}")
            if version:
                stdout.append(f"[DENSE_ADAMW_SEQUENCE_STEP] index={version - 1}")
        stdout.extend(("[DENSE_ADAMW_PAGED_DMA_DRAIN]", "[SIM_RESULT]"))
        real = "\n".join(stdout)
        actual = observe_native_adamw_role_values(real, witness)
        self.assertEqual(actual["versions"], (0, 1, 2))
        self.assertFalse(actual["numerical_optimizer_correctness_verified"])
        fake = real.replace(
            next(item.seed_digest for item in witness.roles
                 if item.kind == "optimizer_moment2"),
            "0" * 64, 1,
        )
        with self.assertRaises(RuntimeError):
            observe_native_adamw_role_values(fake, witness)

    def test_wrong_m_role_is_rejected_even_if_abi_count_unchanged(self):
        changed = list(self.paged.state_spans)
        index = next(i for i, item in enumerate(changed)
                     if item.kind == "optimizer_moment1")
        changed[index] = replace(changed[index], kind="optimizer_moment2")
        forged = replace(self.paged, state_spans=tuple(changed))
        with self.assertRaises(SchemaError):
            self.witness(forged)


if __name__ == "__main__":
    unittest.main()
