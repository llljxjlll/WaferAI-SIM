from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema import (
    HbmAddressSpace,
    HbmBinding,
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    PersistentStateManifest,
    StateKind,
    canonical_state_staging_value_id,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)


def _identity(
    kind: StateKind,
    *,
    shard: int = 0,
    generation: int = 0,
) -> PersistentStateIdentity:
    if kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
        return PersistentStateIdentity.create(
            kind=kind,
            instance_ref="P0",
            mesh_ref="P0.tp",
            request_ref="request_slot_0",
            layer_index=1,
            tensor_ref=None,
            shard_index=shard,
            generation=generation,
        )
    return PersistentStateIdentity.create(
        kind=kind,
        instance_ref="P0",
        mesh_ref="P0.tp",
        request_ref=None,
        layer_index=None,
        tensor_ref="layer0.qkv.weight",
        shard_index=shard,
        generation=generation,
    )


def _decl(
    kind: StateKind,
    *,
    shard: int = 0,
    shape: tuple[int, ...] = (2, 2, 4),
) -> PersistentStateDecl:
    access = {
        StateKind.PARAMETER: PersistentStateAccess.READ_ONLY,
        StateKind.KV_KEY: PersistentStateAccess.READ_WRITE,
        StateKind.KV_VALUE: PersistentStateAccess.READ_WRITE,
        StateKind.OPTIMIZER_RESERVED: PersistentStateAccess.RESERVED,
    }[kind]
    return PersistentStateDecl.create(
        identity=_identity(kind, shard=shard),
        shape=shape,
        dtype=DType.FP16,
        layout="dense_row_major",
        lifetime=(
            PersistentStateLifetime.STEP
            if kind is StateKind.OPTIMIZER_RESERVED
            else PersistentStateLifetime.PERSISTENT
        ),
        access=access,
    )


def _manifest() -> PersistentStateManifest:
    key = _decl(StateKind.KV_KEY)
    value = _decl(StateKind.KV_VALUE)
    space = HbmAddressSpace.create(
        die_id=0,
        base_address=0,
        size_bytes=1 << 20,
        alignment_bytes=64,
    )
    return PersistentStateManifest.create(
        address_spaces=(space,),
        declarations=(value, key),
        bindings=(
            HbmBinding.create(
                state_ref=value.id,
                die_id=0,
                address=0x2040,
                size_bytes=value.tensor_bytes,
            ),
            HbmBinding.create(
                state_ref=key.id,
                die_id=0,
                address=0x2000,
                size_bytes=key.tensor_bytes,
            ),
        ),
    )


class PersistentStateSchemaTest(unittest.TestCase):
    def test_round_trip_stable_identity_and_manifest(self) -> None:
        manifest = _manifest()
        manifest.validate()
        decoded = loads_dataclass(
            PersistentStateManifest,
            canonical_json(manifest),
            path="persistent_state_manifest",
        )
        self.assertEqual(decoded, manifest)
        self.assertEqual(canonical_digest(decoded), canonical_digest(manifest))
        self.assertEqual(
            tuple(item.identity.kind for item in manifest.declarations),
            tuple(
                item.identity.kind
                for item in sorted(
                    manifest.declarations,
                    key=lambda item: (item.identity.id, item.id),
                )
            ),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            manifest.bindings[0].address = 0  # type: ignore[misc]

    def test_physical_placement_is_not_part_of_logical_state_identity(self) -> None:
        declaration = _decl(StateKind.PARAMETER, shape=(8, 8))
        space = HbmAddressSpace.create(
            die_id=0,
            base_address=0,
            size_bytes=1 << 20,
            alignment_bytes=64,
        )
        left = PersistentStateManifest.create(
            address_spaces=(space,),
            declarations=(declaration,),
            bindings=(
                HbmBinding.create(
                    state_ref=declaration.id,
                    die_id=0,
                    address=0x1000,
                    size_bytes=declaration.tensor_bytes,
                ),
            ),
        )
        right = PersistentStateManifest.create(
            address_spaces=(space,),
            declarations=(declaration,),
            bindings=(
                HbmBinding.create(
                    state_ref=declaration.id,
                    die_id=0,
                    address=0x1100,
                    size_bytes=declaration.tensor_bytes,
                ),
            ),
        )
        self.assertEqual(left.declarations, right.declarations)
        self.assertNotEqual(left.bindings, right.bindings)
        self.assertNotEqual(left.id, right.id)

    def test_kind_lifetime_and_access_combinations_fail_closed(self) -> None:
        kv = _identity(StateKind.KV_KEY)
        with self.assertRaisesRegex(SchemaError, "KV state must"):
            PersistentStateDecl.create(
                identity=kv,
                shape=(2, 2, 4),
                dtype=DType.FP16,
                layout="dense_row_major",
                lifetime=PersistentStateLifetime.STEP,
                access=PersistentStateAccess.READ_WRITE,
            )
        parameter = _identity(StateKind.PARAMETER)
        with self.assertRaisesRegex(SchemaError, "parameter must"):
            PersistentStateDecl.create(
                identity=parameter,
                shape=(8, 8),
                dtype=DType.FP16,
                layout="dense_row_major",
                lifetime=PersistentStateLifetime.PERSISTENT,
                access=PersistentStateAccess.READ_WRITE,
            )
        optimizer = _identity(StateKind.OPTIMIZER_RESERVED)
        with self.assertRaisesRegex(SchemaError, "cannot grant DMA"):
            PersistentStateDecl.create(
                identity=optimizer,
                shape=(8, 8),
                dtype=DType.FP32,
                layout="dense_row_major",
                lifetime=PersistentStateLifetime.STEP,
                access=PersistentStateAccess.READ_WRITE,
            )
        with self.assertRaisesRegex(SchemaError, "requires request_ref"):
            PersistentStateIdentity.create(
                kind=StateKind.KV_VALUE,
                instance_ref="P0",
                mesh_ref="P0.tp",
                request_ref=None,
                layer_index=0,
                tensor_ref=None,
                shard_index=0,
                generation=0,
            )

    def test_tensor_bytes_and_stable_ids_are_exact(self) -> None:
        declaration = _decl(StateKind.KV_KEY)
        self.assertEqual(declaration.tensor_bytes, 32)
        with self.assertRaisesRegex(SchemaError, "product"):
            replace(declaration, tensor_bytes=31).validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(declaration.identity, generation=1).validate()

    def test_state_staging_id_depends_only_on_valid_access_identity(self) -> None:
        left = canonical_state_staging_value_id("state_access.alpha")
        self.assertEqual(
            canonical_state_staging_value_id("state_access.alpha"), left
        )
        self.assertNotEqual(
            canonical_state_staging_value_id("state_access.beta"), left
        )
        with self.assertRaisesRegex(SchemaError, "non-empty"):
            canonical_state_staging_value_id("")

    def test_address_space_alignment_bounds_and_overflow(self) -> None:
        valid = HbmAddressSpace.create(
            die_id=0,
            base_address=0,
            size_bytes=1 << 20,
            alignment_bytes=64,
        )
        valid.validate()
        for kwargs in (
            dict(die_id=0, base_address=1, size_bytes=64, alignment_bytes=64),
            dict(die_id=0, base_address=0, size_bytes=63, alignment_bytes=64),
            dict(die_id=0, base_address=0, size_bytes=64, alignment_bytes=3),
            dict(
                die_id=0,
                base_address=(1 << 64) - 32,
                size_bytes=64,
                alignment_bytes=32,
            ),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(SchemaError):
                HbmAddressSpace.create(**kwargs)

    def test_manifest_rejects_wrong_home_crossing_overlap_and_missing(self) -> None:
        manifest = _manifest()
        declaration = manifest.declarations[0]
        binding = next(
            item for item in manifest.bindings if item.state_ref == declaration.id
        )
        with self.assertRaisesRegex(SchemaError, "alignment"):
            PersistentStateManifest.create(
                address_spaces=manifest.address_spaces,
                declarations=(declaration,),
                bindings=(
                    HbmBinding.create(
                        state_ref=declaration.id,
                        die_id=0,
                        address=0x2020,
                        size_bytes=declaration.tensor_bytes,
                    ),
                ),
            )
        tiny_space = HbmAddressSpace.create(
            die_id=0,
            base_address=0,
            size_bytes=0x2000,
            alignment_bytes=64,
        )
        with self.assertRaisesRegex(SchemaError, "home address space"):
            PersistentStateManifest.create(
                address_spaces=(tiny_space,),
                declarations=(declaration,),
                bindings=(binding,),
            )
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            PersistentStateManifest.create(
                address_spaces=manifest.address_spaces,
                declarations=manifest.declarations,
                bindings=manifest.bindings[:1],
            )
        with self.assertRaisesRegex(SchemaError, "must not overlap"):
            replace(
                manifest,
                address_spaces=(
                    manifest.address_spaces[0],
                    HbmAddressSpace.create(
                        die_id=1,
                        base_address=0x80000,
                        size_bytes=1 << 20,
                        alignment_bytes=64,
                    ),
                ),
            ).validate()

    def test_manifest_rejects_duplicate_or_tampered_identity(self) -> None:
        manifest = _manifest()
        duplicate_identity = PersistentStateDecl.create(
            identity=manifest.declarations[0].identity,
            shape=(1, 2, 4),
            dtype=DType.FP16,
            layout="dense_row_major",
            lifetime=PersistentStateLifetime.PERSISTENT,
            access=PersistentStateAccess.READ_WRITE,
        )
        with self.assertRaisesRegex(SchemaError, "duplicate logical state identity"):
            PersistentStateManifest.create(
                address_spaces=manifest.address_spaces,
                declarations=(
                    manifest.declarations[0],
                    duplicate_identity,
                ),
                bindings=(),
            )
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(manifest, id="persistent_state_manifest_forged").validate()


if __name__ == "__main__":
    unittest.main()
