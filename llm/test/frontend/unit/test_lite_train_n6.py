from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.lifecycle import (
    add_fixed_sram_lifecycle,
)
from llm.frontend.wafer_frontend.passes.lite_train_link_program import (
    link_s2_lite_train,
)
from llm.frontend.wafer_frontend.passes.lite_train_lower_program import (
    lower_s2_lite_train,
)
from llm.frontend.wafer_frontend.passes.lower_program import (
    _resolve_dependencies,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    _resolved_abis,
    _resolved_state_abis,
    _train_label_seed_overrides,
    _validate_lite_train_state_update,
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.lite_train_n6 import (
    S2_LITE_TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
    S2_LITE_TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
    S2LiteTrainLinkedProgram,
    S2LiteTrainLoweredProgram,
    s2_lite_train_lowering_context,
)
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragment
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateLifetime,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramHbmTarget,
    ProgramSramTarget,
    _manifest_allocations,
)

from test_lite_train_production_chain import _build_chain


class S2LiteTrainN6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _build_chain()[-1]
        cls.lowered = lower_s2_lite_train(cls.source)
        cls.linked = link_s2_lite_train(cls.lowered)

    def test_production_lower_link_exact_counts_and_state_contract(self) -> None:
        leaves = tuple(_leaf_fragment(item) for item in self.lowered.fragments)
        streams = tuple(
            stream for leaf in leaves for stream in leaf.core_streams
        )
        self.assertEqual(
            (
                len(leaves),
                sum(len(stream.records) for stream in streams),
                sum(len(stream.runtime_relocations) for stream in streams),
                sum(len(stream.address_relocations) for stream in streams),
                sum(len(leaf.buffer_abi) for leaf in leaves),
                sum(len(leaf.state_abi) for leaf in leaves),
            ),
            (46, 169, 0, 324, 99, 17),
        )
        self.assertEqual(
            Counter(
                record.opcode
                for stream in streams
                for record in stream.records
            )[RecordOpcode.CROSS_ENTROPY_BACKWARD],
            1,
        )
        self.assertEqual(
            Counter(
                record.opcode
                for stream in streams
                for record in stream.records
            )[RecordOpcode.SGD_UPDATE],
            1,
        )
        trainable = tuple(
            state
            for leaf in leaves
            for state in leaf.state_abi
            if state.kind is StateKind.TRAINABLE_PARAMETER
        )
        self.assertEqual(len(trainable), 3)
        self.assertEqual(len({item.state_ref for item in trainable}), 1)
        self.assertEqual(len({item.hbm_binding_ref for item in trainable}), 1)
        self.assertTrue(
            all(
                item.lifetime is PersistentStateLifetime.PERSISTENT
                and item.access is PersistentStateAccess.READ_WRITE
                for item in trainable
            )
        )

        manifest = self.linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.core_streams),
                sum(len(stream.records) for stream in manifest.core_streams),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.input_digests),
            ),
            (46, 1, 169, 1, 110, 307, 17, 50),
        )
        self.assertEqual(lower_s2_lite_train(self.source), self.lowered)
        self.assertEqual(link_s2_lite_train(self.lowered), self.linked)
        self.assertTrue(self.lowered.id.startswith("s2_lite_train_lowered_program_"))
        self.assertTrue(self.linked.id.startswith("s2_lite_train_linked_program_"))
        self.assertTrue(manifest.id.startswith("linked_program_manifest_"))

    def test_strict_roundtrip_versions_stable_ids_and_tamper(self) -> None:
        self.assertEqual(
            S2_LITE_TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_train_lowered_program/v1alpha1",
        )
        self.assertEqual(
            S2_LITE_TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_train_linked_program/v1alpha1",
        )
        self.assertEqual(
            loads_dataclass(
                S2LiteTrainLoweredProgram,
                canonical_json(self.lowered),
                path="s2_lite_train_lowered_program",
            ),
            self.lowered,
        )
        self.assertEqual(
            loads_dataclass(
                S2LiteTrainLinkedProgram,
                canonical_json(self.linked),
                path="s2_lite_train_linked_program",
            ),
            self.linked,
        )
        with self.subTest("old-lowered-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    self.lowered,
                    schema_version=(
                        "wafer_frontend.s2_lite_train_lowered_program/v1alpha0"
                    ),
                ).validate()
        with self.subTest("old-linked-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    self.linked,
                    schema_version=(
                        "wafer_frontend.s2_lite_train_linked_program/v1alpha0"
                    ),
                ).validate()
        with self.subTest("noncanonical-fragment-order"):
            with self.assertRaisesRegex(SchemaError, "canonical leaf-id order"):
                replace(
                    self.lowered,
                    fragments=tuple(reversed(self.lowered.fragments)),
                ).validate()
        with self.subTest("missing-fragment"):
            with self.assertRaises(SchemaError):
                S2LiteTrainLoweredProgram.create(
                    source=self.source,
                    lowering_context=self.lowered.lowering_context,
                    fragments=self.lowered.fragments[:-1],
                )
        with self.subTest("forged-linked-manifest-provenance"):
            with self.assertRaises(SchemaError):
                replace(
                    self.linked,
                    manifest=replace(
                        self.linked.manifest,
                        source_global_dag_id="global_action_dag_forged",
                    ),
                ).validate()

    def test_lifecycle_accepts_exact_alias_and_rejects_bad_root_graphs(self) -> None:
        context = s2_lite_train_lowering_context(self.source)
        action = next(
            item
            for item in context.global_dag.actions
            if ".sgd_update" in item.source.task_id
        )
        raw = _resolve_dependencies(None, None, None, None, None).coarse.lower(
            action,
            context,
        )
        decorated = add_fixed_sram_lifecycle(raw, context)
        decorated.validate_against(context.global_dag)
        root = next(
            item
            for item in raw.buffer_abi
            if item.ownership is not BufferOwnership.ALIASED
            and any(
                alias.alias_of == item.binding_id
                for alias in raw.buffer_abi
                if alias.ownership is BufferOwnership.ALIASED
            )
        )
        alias = next(
            item
            for item in raw.buffer_abi
            if item.ownership is BufferOwnership.ALIASED
        )

        def tamper(*members):
            return CommandFragment.create(
                producer_pass=raw.producer_pass,
                **{
                    **raw._semantic_key(),
                    "buffer_abi": tuple(members),
                },
            )

        other = tuple(item for item in raw.buffer_abi if item is not alias)
        cases = {
            "multiple-root": tamper(
                *other,
                replace(alias, ownership=BufferOwnership.OWNED, alias_of=None),
            ),
            "dangling": tamper(
                *other,
                replace(alias, alias_of="buffer_binding_missing"),
            ),
            "cross-geometry": tamper(
                *other,
                replace(alias, region_offset_bytes=alias.region_offset_bytes + 64),
            ),
            "root-does-not-cover-alias": tamper(
                *tuple(item for item in raw.buffer_abi if item is not root),
                replace(root, lifetime_start=alias.lifetime_start + 1),
            ),
        }
        chained_root = replace(
            alias,
            id="buffer_abi_chain_root",
            binding_id="buffer_binding_chain_root",
            alias_of=root.binding_id,
        )
        cases["chained"] = tamper(
            *other,
            replace(alias, alias_of=chained_root.binding_id),
            chained_root,
        )
        for name, fragment in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(
                    SchemaError,
                    "canonical root|directly reuse",
                ):
                    add_fixed_sram_lifecycle(
                        fragment,
                        context,
                        validate=False,
                    )

    def test_program_io_exact_labels_loss_gradient_state_and_terminal(self) -> None:
        seeds, expected = build_deterministic_timing_state_overrides(self.linked)
        self.assertEqual((len(seeds), len(expected)), (15, 0))
        contract = build_timing_program_io(
            self.linked,
            "ab" * 32,
            state_seed_overrides=seeds,
            state_expected_overrides=expected,
        )
        self.assertEqual(
            (
                len(contract.blobs),
                len(contract.initializations),
                len(contract.output_probes),
                sum(
                    type(item.target) is ProgramSramTarget
                    for item in contract.initializations
                ),
                sum(
                    type(item.target) is ProgramHbmTarget
                    for item in contract.initializations
                ),
            ),
            (23, 62, 1, 47, 15),
        )
        probe = contract.output_probes[0]
        self.assertIs(type(probe.target), ProgramSramTarget)
        self.assertEqual(
            (probe.target.value_id, probe.target.dtype.value, probe.length_bytes),
            ("T0.loss", "fp32", 32),
        )
        blobs = {blob.id: blob.payload() for blob in contract.blobs}
        sram_by_value = {
            item.target.value_id: item
            for item in contract.initializations
            if type(item.target) is ProgramSramTarget
        }
        label = sram_by_value["T0.labels"]
        self.assertEqual(
            blobs[label.blob_ref],
            b"".join(index.to_bytes(4, "little") for index in range(8)),
        )
        loss_gradient = sram_by_value["T0.loss_gradient"]
        self.assertEqual(blobs[loss_gradient.blob_ref], bytes(32))
        self.assertNotIn("T0.lm_head.weight.updated", sram_by_value)

        automatic = {
            label.target.buffer_abi_id: blobs[label.blob_ref],
            loss_gradient.target.buffer_abi_id: blobs[loss_gradient.blob_ref],
        }
        self.assertEqual(
            build_timing_program_io(
                self.linked,
                "ab" * 32,
                sram_seed_overrides=automatic,
                state_seed_overrides=seeds,
            ),
            contract,
        )
        with self.subTest("conflicting-label"):
            with self.assertRaisesRegex(SchemaError, "label override conflicts"):
                build_timing_program_io(
                    self.linked,
                    "ab" * 32,
                    sram_seed_overrides={
                        label.target.buffer_abi_id: b"\xff" + blobs[label.blob_ref][1:]
                    },
                    state_seed_overrides=seeds,
                )
        trainable_ref = next(
            state_ref
            for state_ref in seeds
            if any(
                state.kind is StateKind.TRAINABLE_PARAMETER
                and state.state_ref == state_ref
                for fragment in self.lowered.fragments
                for state in _leaf_fragment(fragment).state_abi
            )
        )
        with self.subTest("numeric-updated-weight"):
            with self.assertRaisesRegex(SchemaError, "does not claim numeric"):
                build_timing_program_io(
                    self.linked,
                    "ab" * 32,
                    state_seed_overrides=seeds,
                    state_expected_overrides={trainable_ref: seeds[trainable_ref]},
                )

    def test_program_io_state_order_label_use_and_alias_fail_closed(self) -> None:
        resolved = _resolved_abis(self.linked)
        resolved_state = _resolved_state_abis(self.linked)
        trainable_index = next(
            index
            for index, item in enumerate(resolved_state)
            if item.abi.kind is StateKind.TRAINABLE_PARAMETER
        )
        trainable = resolved_state[trainable_index]
        bad_state = replace(
            trainable,
            uses=(trainable.uses[0], trainable.uses[2], trainable.uses[1]),
        )
        with self.assertRaisesRegex(SchemaError, "two loads before SGD"):
            _validate_lite_train_state_update(
                self.linked,
                resolved,
                (
                    *resolved_state[:trainable_index],
                    bad_state,
                    *resolved_state[trainable_index + 1 :],
                ),
            )

        labels = tuple(
            (index, item)
            for index, item in enumerate(resolved)
            if item.abi.value_id == "T0.labels"
        )
        self.assertEqual(len(labels), 1)
        label_index, label = labels[0]
        with self.assertRaisesRegex(SchemaError, "label BufferABI/use contract"):
            _train_label_seed_overrides(
                self.linked,
                (
                    *resolved[:label_index],
                    replace(label, uses=label.uses[:1]),
                    *resolved[label_index + 1 :],
                ),
            )

        manifest = self.linked.manifest
        fragment_index = next(
            index
            for index, item in enumerate(manifest.fragments)
            if any(
                abi.ownership is BufferOwnership.ALIASED
                for abi in _leaf_fragment(item).buffer_abi
            )
        )
        fragment = _leaf_fragment(manifest.fragments[fragment_index])
        alias = next(
            abi
            for abi in fragment.buffer_abi
            if abi.ownership is BufferOwnership.ALIASED
        )
        root = next(
            abi for abi in fragment.buffer_abi if abi.binding_id == alias.alias_of
        )

        def tampered_manifest(*abis):
            tampered_fragment = replace(fragment, buffer_abi=tuple(abis))
            return replace(
                manifest,
                fragments=(
                    *manifest.fragments[:fragment_index],
                    tampered_fragment,
                    *manifest.fragments[fragment_index + 1 :],
                ),
            )

        other = tuple(abi for abi in fragment.buffer_abi if abi is not alias)
        chained = replace(
            alias,
            id="buffer_abi_chain",
            binding_id="buffer_binding_chain",
            alias_of=root.binding_id,
        )
        cases = {
            "multiple-root": tampered_manifest(
                *other,
                replace(alias, ownership=BufferOwnership.OWNED, alias_of=None),
            ),
            "dangling": tampered_manifest(
                *other,
                replace(alias, alias_of="buffer_binding_missing"),
            ),
            "chained": tampered_manifest(
                *other,
                replace(alias, alias_of=chained.binding_id),
                chained,
            ),
            "cross-geometry": tampered_manifest(
                *other,
                replace(alias, size_bytes=alias.size_bytes + 64),
            ),
        }
        for name, tampered in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(
                    SchemaError,
                    "canonical non-alias root|directly reuse",
                ):
                    _manifest_allocations(tampered, "linked_program_manifest")

        allocation_fragment_index = next(
            index
            for index, item in enumerate(manifest.fragments)
            if any(
                record.opcode is RecordOpcode.SRAM_ALLOC_AT
                and any(
                    binding.fragment_id == _leaf_fragment(item).id
                    and binding.logical_core == stream.logical_core
                    and binding.fragment_record_index == record_index
                    and root.id in binding.buffer_abi_ids
                    for binding in manifest.address_operand_bindings
                )
                for stream in _leaf_fragment(item).core_streams
                for record_index, record in enumerate(stream.records)
            )
        )
        allocation_fragment = _leaf_fragment(
            manifest.fragments[allocation_fragment_index]
        )
        stream = allocation_fragment.core_streams[0]
        alloc_index = next(
            index
            for index, record in enumerate(stream.records)
            if record.opcode is RecordOpcode.SRAM_ALLOC_AT
            and any(
                binding.fragment_id == allocation_fragment.id
                and binding.logical_core == stream.logical_core
                and binding.fragment_record_index == index
                and root.id in binding.buffer_abi_ids
                for binding in manifest.address_operand_bindings
            )
        )
        new_index = len(stream.records)
        alloc_relocations = tuple(
            relocation
            for relocation in stream.address_relocations
            if relocation.record_index == alloc_index
        )
        extra_stream = replace(
            stream,
            records=(*stream.records, stream.records[alloc_index]),
            address_relocations=(
                *stream.address_relocations,
                *(
                    replace(relocation, record_index=new_index)
                    for relocation in alloc_relocations
                ),
            ),
        )
        extra_fragment = replace(
            allocation_fragment,
            core_streams=(extra_stream,),
        )
        source_bindings = tuple(
            binding
            for binding in manifest.address_operand_bindings
            if binding.fragment_id == allocation_fragment.id
            and binding.logical_core == stream.logical_core
            and binding.fragment_record_index == alloc_index
        )
        extra_bindings = tuple(
            replace(
                binding,
                fragment_record_index=new_index,
                buffer_abi_ids=(alias.id,),
                tensor_slices=(alias.tensor_slice,),
            )
            for binding in source_bindings
        )
        extra_alloc_manifest = replace(
            manifest,
            fragments=(
                *manifest.fragments[:allocation_fragment_index],
                extra_fragment,
                *manifest.fragments[allocation_fragment_index + 1 :],
            ),
            address_operand_bindings=(
                *manifest.address_operand_bindings,
                *extra_bindings,
            ),
        )
        with self.subTest("extra-alias-allocation"):
            with self.assertRaisesRegex(
                SchemaError,
                "each core/label and BufferABI requires exactly one",
            ):
                _manifest_allocations(
                    extra_alloc_manifest,
                    "linked_program_manifest",
                )


if __name__ == "__main__":
    unittest.main()
