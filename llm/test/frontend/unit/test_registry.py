from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import (
    RegistryError,
    SchemaError,
    StageNotImplementedError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.policies.registry import (
    POLICY_SELECTION_SCHEMA_VERSION,
    PolicyRegistry,
    PolicySelection,
    RegistrationState,
    RegistryKind,
    default_registry,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass


def _activate(
    registry: PolicyRegistry,
    kind: RegistryKind,
    name: str,
    factory: object,
    *,
    configuration: object = (),
) -> None:
    registry.activate(
        kind,
        name,
        factory,  # type: ignore[arg-type]
        implementation_id=f"test.{kind.value}.{name}",
        implementation_schema_version=f"test.{kind.value}.{name}/v1",
        capability_ids=("test.capability",),
        configuration=configuration,
    )


class RegistryTest(unittest.TestCase):
    def test_default_registry_activates_only_naive_production_policies(self) -> None:
        registry = default_registry()
        expected_active_types = (
            (RegistryKind.INTER_DIE, "naive", NaiveInterDiePolicy),
            (
                RegistryKind.STANDALONE_COLLECTIVE,
                "direct_all_gather",
                DirectAllGatherPolicy,
            ),
            (RegistryKind.INTRA_DIE, "naive", NaiveIntraDiePolicy),
        )
        for kind, name, implementation_type in expected_active_types:
            with self.subTest(kind=kind, name=name):
                registration = registry.registration(kind, name)
                self.assertIs(registration.state, RegistrationState.ACTIVE)
                self.assertTrue(registration.interface_version.endswith("/v1"))
                self.assertTrue(registration.implementation_id)
                self.assertTrue(registration.implementation_schema_version)
                self.assertEqual(
                    registration.capability_ids,
                    ("s1.gemm_collective.naive",),
                )
                self.assertIsInstance(registry.create(kind, name), implementation_type)

        for kind, name, stage in (
            (RegistryKind.INTER_DIE, "swizzle_topo", "O1"),
            (RegistryKind.INTRA_DIE, "optimized", "O2"),
        ):
            with self.subTest(kind=kind, name=name):
                registration = registry.registration(kind, name)
                self.assertIs(registration.state, RegistrationState.DECLARED)
                self.assertEqual(registration.available_stage, stage)
                self.assertIsNone(registration.implementation_id)
                self.assertEqual(registration.capability_ids, ())
                with self.assertRaises(StageNotImplementedError):
                    registry.instantiate(kind, name)

    def test_unsupported_name_never_falls_back(self) -> None:
        registry = default_registry()
        with self.assertRaises(UnsupportedFeatureError):
            registry.instantiate(RegistryKind.INTER_DIE, "typo")
        self.assertIs(
            registry.registration(RegistryKind.INTER_DIE, "naive").state,
            RegistrationState.ACTIVE,
        )

    def test_activation_creation_and_selection(self) -> None:
        registry = PolicyRegistry()
        registration = registry.declare(
            RegistryKind.INTER_DIE,
            "naive",
            interface="InterDiePolicy",
            available_stage="N4",
        )

        class Implementation:
            def plan(self) -> None:
                return None

        marker = Implementation()
        _activate(
            registry,
            RegistryKind.INTER_DIE,
            "naive",
            lambda: marker,
            configuration={"alpha": 1, "beta": 2},
        )
        resolved = registry.instantiate(RegistryKind.INTER_DIE, "naive")
        self.assertIs(resolved.implementation, marker)
        self.assertEqual(resolved.selection.name, "naive")
        self.assertEqual(
            resolved.selection.interface_version,
            registration.interface_version,
        )
        resolved.selection.validate()

    def test_declare_enforces_kind_strings_and_exact_interface(self) -> None:
        invalid = (
            ("inter_die", "naive", "InterDiePolicy", "N4"),
            (RegistryKind.INTER_DIE, "", "InterDiePolicy", "N4"),
            (RegistryKind.INTER_DIE, "   ", "InterDiePolicy", "N4"),
            (RegistryKind.INTER_DIE, 7, "InterDiePolicy", "N4"),
            (RegistryKind.INTER_DIE, "naive", "IntraDiePolicy", "N4"),
            (RegistryKind.INTER_DIE, "naive", "InterDiePolicy", ""),
        )
        for kind, name, interface, stage in invalid:
            with self.subTest(kind=kind, name=name, interface=interface, stage=stage):
                with self.assertRaises(RegistryError):
                    PolicyRegistry().declare(
                        kind,
                        name,
                        interface=interface,
                        available_stage=stage,
                    )

    def test_all_kinds_freeze_exact_versioned_interface_contract(self) -> None:
        expected = {
            RegistryKind.FUSION_PARTITION: ("FusionPartition", "run"),
            RegistryKind.INTER_DIE: ("InterDiePolicy", "plan"),
            RegistryKind.STANDALONE_COLLECTIVE: (
                "StandaloneCollectivePolicy",
                "plan",
            ),
            RegistryKind.INTRA_DIE: ("IntraDiePolicy", "schedule"),
            RegistryKind.COARSE_LOWERING: ("CoarseLowering", "lower"),
            RegistryKind.STANDALONE_LOWERING: (
                "StandaloneCollectiveLowering",
                "lower",
            ),
            RegistryKind.FUSED_LOWERING: ("IsaRegionLowering", "lower"),
            RegistryKind.MANIFEST_LINKER: ("ManifestLinker", "link"),
        }
        registry = PolicyRegistry()
        for index, (kind, (interface, method)) in enumerate(expected.items()):
            registration = registry.declare(
                kind,
                f"implementation_{index}",
                interface=interface,
                available_stage="N-test",
            )
            self.assertEqual(registration.interface, interface)
            self.assertTrue(registration.interface_version.endswith("/v1"))
            implementation_type = type(
                f"Implementation{index}",
                (),
                {method: lambda self: None},
            )
            _activate(
                registry,
                kind,
                registration.name,
                implementation_type,
            )
            implementation = registry.create(kind, registration.name)
            self.assertTrue(callable(getattr(implementation, method)))

    def test_activate_requires_zero_argument_factory_without_instantiating(self) -> None:
        registry = PolicyRegistry()
        registry.declare(
            RegistryKind.INTER_DIE,
            "naive",
            interface="InterDiePolicy",
            available_stage="N4",
        )
        with self.assertRaises(RegistryError):
            _activate(
                registry,
                RegistryKind.INTER_DIE,
                "naive",
                lambda required: required,
            )

        calls: list[str] = []

        class Implementation:
            def plan(self) -> None:
                return None

        def factory() -> Implementation:
            calls.append("called")
            return Implementation()

        _activate(registry, RegistryKind.INTER_DIE, "naive", factory)
        self.assertEqual(calls, [])
        registry.instantiate(RegistryKind.INTER_DIE, "naive")
        self.assertEqual(calls, ["called"])

    def test_activate_requires_complete_canonical_metadata(self) -> None:
        invalid_metadata = (
            ("", "test/v1", ("test.capability",)),
            ("test.impl", "", ("test.capability",)),
            ("test.impl", "test/v1", ()),
            ("test.impl", "test/v1", ("z", "a")),
            ("test.impl", "test/v1", ("a", "a")),
        )
        for index, (implementation_id, version, capabilities) in enumerate(
            invalid_metadata
        ):
            with self.subTest(index=index):
                registry = PolicyRegistry()
                registry.declare(
                    RegistryKind.INTER_DIE,
                    "naive",
                    interface="InterDiePolicy",
                    available_stage="N4",
                )
                with self.assertRaises(RegistryError):
                    registry.activate(
                        RegistryKind.INTER_DIE,
                        "naive",
                        type("Implementation", (), {"plan": lambda self: None}),
                        implementation_id=implementation_id,
                        implementation_schema_version=version,
                        capability_ids=capabilities,
                    )

    def test_create_rejects_missing_or_non_callable_required_method(self) -> None:
        for implementation in (object(), type("Bad", (), {"plan": 7})()):
            with self.subTest(implementation=implementation):
                registry = PolicyRegistry()
                registry.declare(
                    RegistryKind.INTER_DIE,
                    "bad",
                    interface="InterDiePolicy",
                    available_stage="N4",
                )
                _activate(
                    registry,
                    RegistryKind.INTER_DIE,
                    "bad",
                    lambda: implementation,
                )
                with self.assertRaisesRegex(RegistryError, "callable plan"):
                    registry.create(RegistryKind.INTER_DIE, "bad")

    def test_create_wraps_factory_failure_as_registry_error(self) -> None:
        registry = PolicyRegistry()
        registry.declare(
            RegistryKind.INTER_DIE,
            "bad",
            interface="InterDiePolicy",
            available_stage="N4",
        )

        def broken_factory() -> object:
            raise ValueError("broken")

        _activate(registry, RegistryKind.INTER_DIE, "bad", broken_factory)
        with self.assertRaisesRegex(RegistryError, "factory failed"):
            registry.create(RegistryKind.INTER_DIE, "bad")

    def test_selection_is_stable_strict_and_frozen(self) -> None:
        first = default_registry().instantiate(RegistryKind.INTER_DIE, "naive")
        second = default_registry().instantiate(RegistryKind.INTER_DIE, "naive")
        self.assertEqual(first.selection, second.selection)
        self.assertIsNot(first.implementation, second.implementation)
        encoded = canonical_json(first.selection)
        decoded = loads_dataclass(PolicySelection, encoded)
        self.assertEqual(decoded, first.selection)
        self.assertEqual(
            decoded.schema_version,
            POLICY_SELECTION_SCHEMA_VERSION,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decoded.name = "forged"  # type: ignore[misc]
        with self.assertRaisesRegex(SchemaError, "unstable selection id"):
            replace(decoded, implementation_id="forged").validate()

    def test_configuration_digest_is_canonical(self) -> None:
        def selection(configuration: object) -> PolicySelection:
            registry = PolicyRegistry()
            registry.declare(
                RegistryKind.INTER_DIE,
                "naive",
                interface="InterDiePolicy",
                available_stage="N4",
            )
            implementation_type = type(
                "Implementation",
                (),
                {"plan": lambda self: None},
            )
            _activate(
                registry,
                RegistryKind.INTER_DIE,
                "naive",
                implementation_type,
                configuration=configuration,
            )
            return registry.instantiate(
                RegistryKind.INTER_DIE, "naive"
            ).selection

        self.assertEqual(
            selection({"a": 1, "b": 2}),
            selection({"b": 2, "a": 1}),
        )

    def test_duplicate_and_undeclared_activation_fail(self) -> None:
        registry = PolicyRegistry()
        registry.declare(
            RegistryKind.INTER_DIE,
            "naive",
            interface="InterDiePolicy",
            available_stage="N4",
        )
        with self.assertRaises(RegistryError):
            registry.declare(
                RegistryKind.INTER_DIE,
                "naive",
                interface="InterDiePolicy",
                available_stage="N4",
            )
        with self.assertRaises(RegistryError):
            _activate(registry, RegistryKind.INTRA_DIE, "naive", object)


if __name__ == "__main__":
    unittest.main()
