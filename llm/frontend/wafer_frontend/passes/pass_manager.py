"""Deterministic pass-order state machine for the frontend pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from ..errors import InputMutationError, PassOrderError, SchemaError
from ..schema.common import stable_artifact_id
from ..schema.policy import PolicySelection, RegistryKind
from ..schema.serde import canonical_digest


PIPELINE_SCHEMA_VERSION = "wafer_frontend.pipeline/v1alpha4"

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class PipelinePhase(str, Enum):
    SPEC_VALIDATED = "spec_validated"
    IR0_READY = "ir0_ready"
    LOGICAL_EXPANDED = "logical_expanded"
    IR1_PLACED = "ir1_placed"
    FUSION_PARTITIONED = "fusion_partitioned"
    INTERDIE_PLANNED = "interdie_planned"
    IR2_PROJECTED = "ir2_projected"
    INTRADIE_REFINED = "intradie_refined"
    INTRADIE_SCHEDULED = "intradie_scheduled"
    GLOBAL_DAG_BUILT = "global_dag_built"
    LOWERED = "lowered"
    MANIFEST_LINKED = "manifest_linked"
    ARTIFACT_FINALIZED = "artifact_finalized"
    SIMULATED = "simulated"


@dataclass(frozen=True, slots=True)
class PassSpec:
    name: str
    input_phase: PipelinePhase
    output_phase: PipelinePhase


PASS_SPECS = (
    PassSpec("build_ir0", PipelinePhase.SPEC_VALIDATED, PipelinePhase.IR0_READY),
    PassSpec("logical_expand", PipelinePhase.IR0_READY, PipelinePhase.LOGICAL_EXPANDED),
    PassSpec("placement", PipelinePhase.LOGICAL_EXPANDED, PipelinePhase.IR1_PLACED),
    PassSpec(
        "fusion_partition", PipelinePhase.IR1_PLACED, PipelinePhase.FUSION_PARTITIONED
    ),
    PassSpec(
        "inter_die_plan",
        PipelinePhase.FUSION_PARTITIONED,
        PipelinePhase.INTERDIE_PLANNED,
    ),
    PassSpec(
        "project_to_ir2", PipelinePhase.INTERDIE_PLANNED, PipelinePhase.IR2_PROJECTED
    ),
    PassSpec(
        "intra_die_refine",
        PipelinePhase.IR2_PROJECTED,
        PipelinePhase.INTRADIE_REFINED,
    ),
    PassSpec(
        "intra_die_schedule",
        PipelinePhase.INTRADIE_REFINED,
        PipelinePhase.INTRADIE_SCHEDULED,
    ),
    PassSpec(
        "global_action_dag",
        PipelinePhase.INTRADIE_SCHEDULED,
        PipelinePhase.GLOBAL_DAG_BUILT,
    ),
    PassSpec("lowering", PipelinePhase.GLOBAL_DAG_BUILT, PipelinePhase.LOWERED),
    PassSpec(
        "manifest_link", PipelinePhase.LOWERED, PipelinePhase.MANIFEST_LINKED
    ),
    PassSpec(
        "artifact_finalize",
        PipelinePhase.MANIFEST_LINKED,
        PipelinePhase.ARTIFACT_FINALIZED,
    ),
    PassSpec("simulation", PipelinePhase.ARTIFACT_FINALIZED, PipelinePhase.SIMULATED),
)

_PASS_BY_NAME = {spec.name: spec for spec in PASS_SPECS}
_PASS_BY_INPUT = {spec.input_phase: spec for spec in PASS_SPECS}
_CONTEXT_REQUIRED_PASSES = frozenset(
    {
        "placement",
        "fusion_partition",
        "inter_die_plan",
        "project_to_ir2",
        "intra_die_refine",
        "intra_die_schedule",
    }
)
_POLICY_KINDS_BY_PASS = {
    "inter_die_plan": (
        RegistryKind.INTER_DIE,
        RegistryKind.STANDALONE_COLLECTIVE,
    ),
    "intra_die_schedule": (RegistryKind.INTRA_DIE,),
}


@dataclass(frozen=True, slots=True)
class PassReceipt:
    pass_name: str
    input_phase: PipelinePhase
    output_phase: PipelinePhase
    input_digest: str
    context_digest: str | None
    output_digest: str
    policy_selections: tuple[PolicySelection, ...] = ()

    def validate(self, path: str = "pass_receipt") -> None:
        spec = _PASS_BY_NAME.get(self.pass_name)
        if spec is None:
            raise SchemaError("references an unknown pass", path=f"{path}.pass_name")
        if (
            self.input_phase is not spec.input_phase
            or self.output_phase is not spec.output_phase
        ):
            raise SchemaError(
                "receipt phases disagree with the fixed pass transition",
                path=path,
            )
        if self.pass_name in _CONTEXT_REQUIRED_PASSES and self.context_digest is None:
            raise SchemaError(
                "pass requires a recorded context digest",
                path=f"{path}.context_digest",
            )
        for field_name in ("input_digest", "output_digest"):
            digest = getattr(self, field_name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SchemaError(
                    "must be a lowercase SHA-256 digest",
                    path=f"{path}.{field_name}",
                )
        if self.context_digest is not None:
            digest = self.context_digest
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SchemaError(
                    "must be null or a lowercase SHA-256 digest",
                    path=f"{path}.context_digest",
                )
        if type(self.policy_selections) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.policy_selections",
            )
        for index, selection in enumerate(self.policy_selections):
            if type(selection) is not PolicySelection:
                raise SchemaError(
                    "must be a PolicySelection",
                    path=f"{path}.policy_selections[{index}]",
                )
            selection.validate(f"{path}.policy_selections[{index}]")
        expected_kinds = _POLICY_KINDS_BY_PASS.get(self.pass_name, ())
        if (
            tuple(selection.kind for selection in self.policy_selections)
            != expected_kinds
        ):
            raise SchemaError(
                "policy selections do not match the pass's exact kind order",
                path=f"{path}.policy_selections",
            )


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    schema_version: str
    producer_pass: str
    id: str
    phase: PipelinePhase
    receipts: tuple[PassReceipt, ...]

    def validate(self, path: str = "pipeline_snapshot") -> None:
        if self.schema_version != PIPELINE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "pass_manager":
            raise SchemaError("must be 'pass_manager'", path=f"{path}.producer_pass")
        if type(self.phase) is not PipelinePhase:
            raise SchemaError("must be a PipelinePhase", path=f"{path}.phase")
        if type(self.receipts) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.receipts")
        current_phase = PipelinePhase.SPEC_VALIDATED
        previous_output_digest: str | None = None
        for index, receipt in enumerate(self.receipts):
            receipt_path = f"{path}.receipts[{index}]"
            receipt.validate(receipt_path)
            if receipt.input_phase is not current_phase:
                raise SchemaError(
                    "receipt input phase is not continuous",
                    path=f"{receipt_path}.input_phase",
                )
            if (
                previous_output_digest is not None
                and receipt.input_digest != previous_output_digest
            ):
                raise SchemaError(
                    "receipt input digest does not equal the previous output digest",
                    path=f"{receipt_path}.input_digest",
                )
            current_phase = receipt.output_phase
            previous_output_digest = receipt.output_digest
        if self.phase is not current_phase:
            raise SchemaError(
                "snapshot phase does not equal the receipt chain output",
                path=f"{path}.phase",
            )
        expected_id = stable_artifact_id(
            "pipeline",
            {"phase": self.phase.value, "receipts": self.receipts},
            schema_version=PIPELINE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class PipelineDescription:
    schema_version: str
    initial_phase: PipelinePhase
    passes: tuple[PassSpec, ...]


def pipeline_description() -> PipelineDescription:
    return PipelineDescription(
        schema_version=PIPELINE_SCHEMA_VERSION,
        initial_phase=PipelinePhase.SPEC_VALIDATED,
        passes=PASS_SPECS,
    )


def _make_snapshot(
    phase: PipelinePhase, receipts: tuple[PassReceipt, ...]
) -> PipelineSnapshot:
    snapshot_id = stable_artifact_id(
        "pipeline",
        {"phase": phase.value, "receipts": receipts},
        schema_version=PIPELINE_SCHEMA_VERSION,
    )
    snapshot = PipelineSnapshot(
        schema_version=PIPELINE_SCHEMA_VERSION,
        producer_pass="pass_manager",
        id=snapshot_id,
        phase=phase,
        receipts=receipts,
    )
    snapshot.validate()
    return snapshot


class PassManager:
    """Run exactly one legal successor pass at a time."""

    def __init__(self, initial_phase: PipelinePhase = PipelinePhase.SPEC_VALIDATED) -> None:
        if initial_phase is not PipelinePhase.SPEC_VALIDATED:
            raise PassOrderError(
                "a new pipeline must start at 'spec_validated'; validated resume is not implemented",
                path="pipeline.initial_phase",
            )
        self._snapshot = _make_snapshot(PipelinePhase.SPEC_VALIDATED, ())
        self._running = False
        self._reentrant_violation = False

    @property
    def snapshot(self) -> PipelineSnapshot:
        return self._snapshot

    @property
    def next_pass(self) -> PassSpec | None:
        return _PASS_BY_INPUT.get(self._snapshot.phase)

    def run_pass(
        self,
        pass_name: str,
        input_artifact: InputT,
        transform: Callable[..., OutputT],
        *,
        context: object | None = None,
        policy_selections: tuple[PolicySelection, ...] = (),
    ) -> OutputT:
        if self._running:
            self._reentrant_violation = True
            raise PassOrderError(
                "the same PassManager cannot run a pass reentrantly",
                path="pipeline.pass",
            )
        self._snapshot.validate("pipeline.snapshot")
        if pass_name not in _PASS_BY_NAME:
            raise PassOrderError(f"unknown pass {pass_name!r}", path="pipeline.pass")
        spec = _PASS_BY_NAME[pass_name]
        expected = self.next_pass
        if expected is None or spec != expected:
            expected_name = expected.name if expected is not None else "<pipeline complete>"
            raise PassOrderError(
                f"cannot run {pass_name!r} from phase {self._snapshot.phase.value!r}; "
                f"expected {expected_name!r}",
                path="pipeline.pass",
            )
        if pass_name in _CONTEXT_REQUIRED_PASSES and context is None:
            raise PassOrderError(
                f"pass {pass_name!r} requires an explicit immutable context",
                path="pipeline.context",
            )
        expected_policy_kinds = _POLICY_KINDS_BY_PASS.get(pass_name, ())
        if type(policy_selections) is not tuple:
            raise PassOrderError(
                "policy_selections must be an immutable tuple",
                path="pipeline.policy_selections",
            )
        for index, selection in enumerate(policy_selections):
            if type(selection) is not PolicySelection:
                raise PassOrderError(
                    "policy selection must be a PolicySelection",
                    path=f"pipeline.policy_selections[{index}]",
                )
            selection.validate(f"pipeline.policy_selections[{index}]")
        if (
            tuple(selection.kind for selection in policy_selections)
            != expected_policy_kinds
        ):
            raise PassOrderError(
                "policy selections do not match the pass's exact kind order",
                path="pipeline.policy_selections",
            )

        input_digest = canonical_digest(input_artifact)
        context_digest = None if context is None else canonical_digest(context)
        if (
            self._snapshot.receipts
            and input_digest != self._snapshot.receipts[-1].output_digest
        ):
            raise PassOrderError(
                "input artifact digest does not equal the previous pass output digest",
                path="pipeline.input",
            )

        # Phase-specific artifact/source-id contracts remain the responsibility
        # of future typed producer bundles.  This N1 boundary guarantees that a
        # pass consumes the exact canonical output of its predecessor.
        entry_snapshot = self._snapshot
        committed = False
        self._running = True
        self._reentrant_violation = False

        def ensure_inputs_unchanged(*, before_failure: bool = False) -> None:
            suffix = " before failing" if before_failure else ""
            try:
                current_digest = canonical_digest(input_artifact)
            except SchemaError as error:
                raise InputMutationError(
                    f"pass {pass_name!r} mutated its input{suffix}; "
                    "the input is no longer canonically encodable",
                    path="pipeline.input",
                ) from error
            if current_digest != input_digest:
                raise InputMutationError(
                    f"pass {pass_name!r} mutated its input{suffix}",
                    path="pipeline.input",
                )
            if context is None:
                return
            try:
                current_context_digest = canonical_digest(context)
            except SchemaError as error:
                raise InputMutationError(
                    f"pass {pass_name!r} mutated its context{suffix}; "
                    "the context is no longer canonically encodable",
                    path="pipeline.context",
                ) from error
            if current_context_digest != context_digest:
                raise InputMutationError(
                    f"pass {pass_name!r} mutated its context{suffix}",
                    path="pipeline.context",
                )

        try:
            try:
                output = (
                    transform(input_artifact)
                    if context is None
                    else transform(input_artifact, context)
                )
                ensure_inputs_unchanged()
                if self._reentrant_violation:
                    raise PassOrderError(
                        "a reentrant pass attempt invalidated the transaction",
                        path="pipeline.pass",
                    )
                validator = getattr(output, "validate", None)
                if validator is not None:
                    if not callable(validator):
                        raise SchemaError(
                            "validate attribute must be callable",
                            path="pipeline.output.validate",
                        )
                    validator("pipeline.output")
                ensure_inputs_unchanged()
                output_digest = canonical_digest(output)
                if self._reentrant_violation:
                    raise PassOrderError(
                        "a reentrant pass attempt invalidated the transaction",
                        path="pipeline.pass",
                    )
            except InputMutationError:
                raise
            except Exception as error:
                try:
                    ensure_inputs_unchanged(before_failure=True)
                except InputMutationError as mutation_error:
                    raise mutation_error from error
                raise

            receipt = PassReceipt(
                pass_name=pass_name,
                input_phase=spec.input_phase,
                output_phase=spec.output_phase,
                input_digest=input_digest,
                context_digest=context_digest,
                output_digest=output_digest,
                policy_selections=policy_selections,
            )
            candidate = _make_snapshot(
                spec.output_phase, entry_snapshot.receipts + (receipt,)
            )
            self._snapshot = candidate
            committed = True
            return output
        finally:
            if not committed:
                self._snapshot = entry_snapshot
            self._running = False
            self._reentrant_violation = False
