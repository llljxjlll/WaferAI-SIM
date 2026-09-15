"""Bounded real L2 forward/native seeded-CE carrier for provenance auditing.

This product preserves every original Dense forward record and appends only
its physically allocated independent dLoss/native CE backward.  It never
certifies a full training IR1/schedule: a caller must independently run
`validate_against` on new production IR1/projection/schedule artifacts before
offering this carrier as complete training evidence.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LinkedProgramManifest, ManifestInputDigest, ManifestInputKind,
    RuntimeSymbolKind,
)
from ..schema.global_action import GlobalActionDAG
from ..schema.serde import canonical_digest
from .full_training_ce_tape_graft import CeGraftedPhysicalForward
from .moe_full_model_linker import _interfaces


def build_bounded_seeded_ce_physical_manifest(
    source_forward: LinkedProgramManifest,
    *,
    graft: CeGraftedPhysicalForward,
    new_global_dag: GlobalActionDAG,
) -> LinkedProgramManifest:
    """Build a schema-valid physical carrier without altering source tasks.

    Production `validate_against(IR1, projection, schedule, global_dag)`
    still rejects the CE backward task until the real multi-layer full TRAIN
    source introduces that new task.  This function does not manufacture a
    source IR1/projection/schedule that hides the rejected task.
    """
    source_forward.validate("bounded_seeded_ce_forward_source")
    new_global_dag.validate("bounded_seeded_ce_global_source")
    if (graft.source_forward_manifest_id != source_forward.id
            or graft.global_dag_id != new_global_dag.id
            or graft.loss_gradient_seed is None
            or graft.loss_gradient_abi_id is None
            or graft.retained_forward_loss_free is None):
        raise SchemaError("bounded CE must originate in source L2 and real dLoss graft",
                          path="graft")
    original_cores = {stream.logical_core for stream in source_forward.core_streams}
    if {stream.logical_core for stream in graft.core_streams} != original_cores:
        raise SchemaError("seeded CE graft changed actual physical core inventory",
                          path="graft.core_streams")
    if not any(graft.retained_forward_loss_free in stream.records
               for stream in graft.core_streams):
        raise SchemaError("original forward loss FREE disappeared",
                          path="graft.retained_forward_loss_free")
    source_runtime_starts = tuple(definition for definition
                                  in source_forward.runtime_symbol_definitions
                                  if definition.symbol.kind is
                                  RuntimeSymbolKind.START_TAG)
    if not source_runtime_starts:
        raise SchemaError("source real L2 lacks START envelope", path="source_forward")
    old_inputs = tuple(input_digest for input_digest in
                       source_forward.input_digests if input_digest.kind not in (
                           ManifestInputKind.GLOBAL_ACTION_DAG,
                           ManifestInputKind.COMMAND_FRAGMENT,
                       ))
    inputs = tuple(sorted((
        *old_inputs,
        ManifestInputDigest(ManifestInputKind.GLOBAL_ACTION_DAG,
                            new_global_dag.id, new_global_dag.schema_version,
                            canonical_digest(new_global_dag)),
        *(ManifestInputDigest(ManifestInputKind.COMMAND_FRAGMENT,
                              fragment.id, fragment.schema_version,
                              canonical_digest(fragment))
          for fragment in graft.fragments),
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    args = source_forward._semantic_key()
    args.update({
        "source_global_dag_id": new_global_dag.id,
        "input_digests": inputs,
        "fragments": graft.fragments,
        "fragment_interfaces": _interfaces(graft.fragments),
        "core_streams": graft.core_streams,
        "runtime_symbol_definitions": tuple(sorted((
            *source_runtime_starts, *graft.runtime_definitions,
        ), key=lambda definition: definition.symbol.id)),
        "program_symbol_definitions": graft.program_definitions,
        "address_operand_bindings": tuple(sorted(
            graft.address_bindings,
            key=lambda item: (item.logical_core.die_id,
                              item.logical_core.local_core_id,
                              item.fragment_id, item.fragment_record_index,
                              int(item.operand_id)),
        )),
        "state_operand_bindings": tuple(sorted(
            graft.state_bindings,
            key=lambda item: (item.logical_core.die_id,
                              item.logical_core.local_core_id,
                              item.fragment_id, item.fragment_record_index,
                              int(item.operand_id)),
        )),
    })
    result = LinkedProgramManifest.create(
        producer_pass="bounded_dense_seeded_ce_physical_manifest",
        **args,
    )
    result.validate("bounded_dense_seeded_ce_physical_manifest")
    return result


__all__ = ["build_bounded_seeded_ce_physical_manifest"]
