"""Deterministic lowering for strict HBM-backed state DMA actions."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    AddressRelocation,
    CommandFragment,
    CoreFragmentStream,
    FragmentKind,
    ProgramSymbol,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    SemanticOperandId,
    StateABI,
)
from ..schema.common import stable_artifact_id
from ..schema.global_action import GlobalAction
from ..schema.ir2 import (
    BufferUseRole,
    RegionLowering,
    SemanticTaskKind,
)
from .context import LoweringContext
from .coarse import _buffer_abi, _program_symbol, _view_addend_for_use


_PRODUCER_PASS = "state_dma_lowering"


def _hbm_program_symbol(hbm_binding_ref: str) -> ProgramSymbol:
    semantic = {
        "hbm_binding_ref": hbm_binding_ref,
        "kind": int(ProgramSymbolKind.ABSOLUTE_ADDRESS),
    }
    return ProgramSymbol(
        stable_artifact_id(
            "program_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        ProgramSymbolKind.ABSOLUTE_ADDRESS,
        hbm_binding_ref,
    )


class NaiveStateDmaLowering:
    """Lower one STRICT_STATE_IO action without changing schedule decisions."""

    def __init__(self, *, validate_output: bool = True) -> None:
        self._validate_output = validate_output
        self._validated_contexts: list[LoweringContext] = []

    def _validate_context_once(self, context: LoweringContext) -> None:
        if not any(previous is context for previous in self._validated_contexts):
            context.validate()
            self._validated_contexts.append(context)

    def lower(
        self,
        action: GlobalAction,
        context: LoweringContext,
    ) -> CommandFragment:
        if type(context) is not LoweringContext:
            raise SchemaError("must be a LoweringContext", path="context")
        if type(action) is not GlobalAction:
            raise SchemaError("must be a GlobalAction", path="action")
        self._validate_context_once(context)
        source = next(
            (
                candidate
                for candidate in context.global_dag.actions
                if candidate.id == action.id
            ),
            None,
        )
        if source is None or source != action:
            raise SchemaError(
                "action must exactly equal one action in the lowering context",
                path="action",
            )
        if (
            action.task_kind
            not in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT)
            or action.lowering is not RegionLowering.STRICT_STATE_IO
            or action.logical_core is None
            or action.dma is None
            or len(action.state_uses) != 1
            or len(action.buffer_uses) != 1
        ):
            raise SchemaError(
                "state DMA lowering requires one STRICT_STATE_IO DMA action",
                path="action",
            )

        schedule = next(
            (
                candidate
                for candidate in context.schedule_set.schedules
                if candidate.id == action.source.schedule_id
            ),
            None,
        )
        if schedule is None:
            raise SchemaError(
                "action references an unknown schedule",
                path="action.source.schedule_id",
            )
        local_use = action.buffer_uses[0]
        local_binding = next(
            (
                binding
                for binding in schedule.buffer_bindings
                if binding.id == local_use.binding_id
            ),
            None,
        )
        if local_binding is None:
            raise SchemaError(
                "state DMA references an unknown local buffer binding",
                path="action.buffer_uses[0].binding_id",
            )

        manifest = context.ir1.persistent_state_manifest
        if manifest is None:
            raise SchemaError(
                "state DMA requires an IR1 persistent-state manifest",
                path="context.ir1.persistent_state_manifest",
            )
        state_use = action.state_uses[0]
        hbm_binding = next(
            (
                binding
                for binding in manifest.bindings
                if binding.id == state_use.hbm_binding_ref
            ),
            None,
        )
        if hbm_binding is None:
            raise SchemaError(
                "state DMA references an unknown HBM binding",
                path="action.state_uses[0].hbm_binding_ref",
            )
        declaration = next(
            (
                candidate
                for candidate in manifest.declarations
                if candidate.id == hbm_binding.state_ref
            ),
            None,
        )
        if declaration is None:
            raise SchemaError(
                "HBM binding references an unknown state declaration",
                path="context.ir1.persistent_state_manifest.bindings",
            )
        home = next(
            (
                space
                for space in manifest.address_spaces
                if space.die_id == hbm_binding.die_id
            ),
            None,
        )
        if home is None:
            raise SchemaError(
                "HBM binding has no home address space",
                path="context.ir1.persistent_state_manifest.address_spaces",
            )

        is_load = action.task_kind is SemanticTaskKind.DMA_IN
        expected_role = (
            BufferUseRole.DMA_DESTINATION
            if is_load
            else BufferUseRole.DMA_SOURCE
        )
        local_addend = _view_addend_for_use(
            action,
            local_binding,
            expected_role,
            0,
            path="action.buffer_uses",
        )
        local_symbol = _program_symbol(
            schedule_id=schedule.id,
            binding=local_binding,
            kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
        )
        hbm_symbol = _hbm_program_symbol(hbm_binding.id)
        local_operand_id = (
            SemanticOperandId.DESTINATION_ADDRESS
            if is_load
            else SemanticOperandId.SOURCE_ADDRESS
        )
        record = RelocatableRecord(
            action.id,
            RecordOpcode.LSU_LOAD if is_load else RecordOpcode.LSU_STORE,
            (
                RecordOperand.address(
                    "hbm_address",
                    SemanticOperandId.HBM_ADDRESS,
                    hbm_symbol.id,
                ),
                RecordOperand.literal("size_bytes", action.bytes),
                RecordOperand.address(
                    "destination_address" if is_load else "source_address",
                    local_operand_id,
                    local_symbol.id,
                ),
            ),
        )
        stream = CoreFragmentStream(
            action.logical_core,
            (record,),
            (),
            (
                AddressRelocation(
                    0,
                    local_operand_id,
                    ProgramSymbolKind.ABSOLUTE_ADDRESS,
                    local_symbol.id,
                    local_addend,
                ),
                AddressRelocation(
                    0,
                    SemanticOperandId.HBM_ADDRESS,
                    ProgramSymbolKind.ABSOLUTE_ADDRESS,
                    hbm_symbol.id,
                    action.dma.state_offset_bytes,
                ),
            ),
        )
        state_abi = StateABI.create(
            state_ref=declaration.id,
            hbm_binding_ref=hbm_binding.id,
            kind=declaration.identity.kind,
            lifetime=declaration.lifetime,
            access=declaration.access,
            shape=declaration.shape,
            dtype=declaration.dtype,
            layout=declaration.layout,
            die_id=hbm_binding.die_id,
            address=hbm_binding.address,
            size_bytes=hbm_binding.size_bytes,
            alignment_bytes=home.alignment_bytes,
        )
        fragment = CommandFragment.create(
            producer_pass=_PRODUCER_PASS,
            source_global_dag_id=context.global_dag.id,
            kind=FragmentKind.STATE_IO,
            claimed_action_ids=(action.id,),
            core_streams=(stream,),
            runtime_symbols=(),
            program_symbols=tuple(
                sorted((hbm_symbol, local_symbol), key=lambda symbol: symbol.id)
            ),
            buffer_abi=(
                _buffer_abi(schedule.id, local_binding, action.logical_core),
            ),
            state_abi=(state_abi,),
        )
        if self._validate_output:
            fragment.validate_against(context.global_dag)
        return fragment
