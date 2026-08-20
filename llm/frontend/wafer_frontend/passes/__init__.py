"""Frontend pass orchestration contracts."""

from .build_ir0 import build_ir0, build_ir0_template_for_profile
from .capability_manifest import (
    build_capability_manifest,
    build_s1_s3_case_matrix,
    build_stage1a_capability_manifest,
    build_stage1a_case_matrix,
    build_stage2_capability_manifest,
    build_stage2_case_matrix,
    build_stage3_capability_manifest,
    build_stage3_case_matrix,
    build_stage4_capability_manifest,
    build_stage0_capability_manifest,
)
from .logical_expand import logical_expand
from .train_forward import build_train_forward_ir0
from .train_forward_oracle import build_train_forward_oracle
from .lite_train import build_s2_lite_lm_head_train_oracle
from .lite_train_graph import build_s2_lite_lm_head_train_ir0
from .lite_train_link_program import link_s2_lite_train
from .lite_train_lower_program import lower_s2_lite_train
from .lite_train_dp2 import (
    build_s2_lite_dp2_rooted_ar_global_action,
    build_s2_lite_dp2_rooted_ar_source,
)
from .lite_train_rooted_ar_n6 import build_s2_lite_rooted_ar_n6_intent
from .lite_train_rooted_ar_lower_program import lower_s2_lite_rooted_ar
from .lite_train_rooted_ar_link_program import link_s2_lite_rooted_ar
from .lite_moe import build_lite_moe_oracle
from .lite_moe_graph import build_lite_moe_ir0_adapter
from .lite_moe_n4 import (
    build_lite_moe_n4,
    place_lite_moe_adapter,
    validate_lite_moe_n4,
    validate_lite_moe_placement,
)
from .lite_moe_execution import (
    build_lite_moe_global,
    project_lite_moe,
    schedule_lite_moe,
    validate_lite_moe_global,
    validate_lite_moe_projection,
    validate_lite_moe_schedule,
)
from .lite_moe_n6 import build_lite_moe_n6_intent, validate_lite_moe_n6_intent
from .lite_moe_lower_program import lower_lite_moe_n6
from .lite_moe_link_program import link_lite_moe_n6
from .train_global_action import (
    build_s2_lite_train_global_action,
    build_train_global_action,
)
from .train_lower_program import lower_train
from .train_link_program import link_train
from .lower_program import lower_bundle, lower_profile, lower_stage4
from .link_program import link_bundle, link_profile, link_stage4
from .load_fabric import (
    hbm_address_spaces_from_data,
    load_physical_fabric_and_hbm_address_spaces,
    load_physical_fabric,
    physical_fabric_from_data,
    validate_identity_mapping_text,
)
from .group_registry import build_group_registry, validate_group_against
from .global_action import build_global_action_dag
from .global_action_dag import (
    build_global_bundle,
    build_global_profile,
    build_stage4_global_action,
)
from .fusion_partition import (
    partition_bundle,
    partition_ir1,
    partition_stage4,
    partition_train_forward,
)
from .intra_die_schedule import (
    schedule_bundle,
    schedule_profile,
    schedule_stage4,
    schedule_train_forward,
)
from .inter_die_plan import (
    plan_bundle,
    plan_profile,
    plan_stage4,
    plan_train_forward,
)
from .placement import (
    place_bundle,
    place_ir0,
    place_train_forward_ir0,
    place_stage4_carrier,
    place_stage4_ir0,
    validate_placement_against,
)
from .pass_manager import PassManager, PipelinePhase
from .project_to_ir2 import (
    project_bundle,
    project_profile,
    project_stage4,
    project_train_forward,
)
from .program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from .stage2_dense_forward_oracle import build_stage2_dense_forward_oracle
from .stage3_profile_selection import select_stage3_profile
from .stage3_dense_inference_oracle import build_stage3_dense_inference_oracle
from .stage4_pd import build_stage4_pd_oracle, build_stage4_pd_plan
from .stage4_pd_case_matrix import build_stage4_pd_case_matrix
from .stage4_logical_expand import (
    build_stage4_fused_ir0,
    build_stage4_separated_ir0,
)
from .stage4_segmented_state_transfer import (
    build_stage4_segmented_state_transfers,
    validate_stage4_segmented_state_transfers,
)
from .stage4_state_transfer import (
    build_stage4_planned_state_transfers,
    build_stage4_state_transfers,
    validate_stage4_planned_state_transfers,
    validate_stage4_state_transfers,
)
from .validate_fusion import FusionSemanticValidator
from .validate_ir0 import DenseIR0Validator
from .validate_logical_bundle import DenseLogicalBundleValidator

__all__ = [
    "build_ir0",
    "build_ir0_template_for_profile",
    "build_capability_manifest",
    "build_s1_s3_case_matrix",
    "build_stage1a_capability_manifest",
    "build_stage1a_case_matrix",
    "build_stage2_capability_manifest",
    "build_stage2_case_matrix",
    "build_stage3_capability_manifest",
    "build_stage3_case_matrix",
    "build_stage4_capability_manifest",
    "build_stage0_capability_manifest",
    "logical_expand",
    "build_train_forward_ir0",
    "build_train_forward_oracle",
    "build_s2_lite_lm_head_train_oracle",
    "build_s2_lite_lm_head_train_ir0",
    "link_s2_lite_train",
    "lower_s2_lite_train",
    "build_s2_lite_dp2_rooted_ar_global_action",
    "build_s2_lite_dp2_rooted_ar_source",
    "build_s2_lite_rooted_ar_n6_intent",
    "lower_s2_lite_rooted_ar",
    "link_s2_lite_rooted_ar",
    "build_lite_moe_oracle",
    "build_lite_moe_ir0_adapter",
    "build_lite_moe_n4",
    "place_lite_moe_adapter",
    "validate_lite_moe_n4",
    "validate_lite_moe_placement",
    "build_lite_moe_global",
    "project_lite_moe",
    "schedule_lite_moe",
    "validate_lite_moe_global",
    "validate_lite_moe_projection",
    "validate_lite_moe_schedule",
    "build_lite_moe_n6_intent",
    "validate_lite_moe_n6_intent",
    "lower_lite_moe_n6",
    "link_lite_moe_n6",
    "build_train_global_action",
    "build_s2_lite_train_global_action",
    "lower_train",
    "link_train",
    "lower_bundle",
    "lower_profile",
    "lower_stage4",
    "link_bundle",
    "link_profile",
    "link_stage4",
    "hbm_address_spaces_from_data",
    "load_physical_fabric_and_hbm_address_spaces",
    "load_physical_fabric",
    "physical_fabric_from_data",
    "validate_identity_mapping_text",
    "build_group_registry",
    "validate_group_against",
    "build_global_action_dag",
    "build_global_bundle",
    "build_global_profile",
    "build_stage4_global_action",
    "partition_bundle",
    "partition_ir1",
    "partition_stage4",
    "partition_train_forward",
    "plan_bundle",
    "plan_profile",
    "plan_stage4",
    "plan_train_forward",
    "project_bundle",
    "project_profile",
    "project_stage4",
    "project_train_forward",
    "build_deterministic_timing_state_overrides",
    "build_timing_program_io",
    "build_stage2_dense_forward_oracle",
    "select_stage3_profile",
    "build_stage3_dense_inference_oracle",
    "build_stage4_pd_oracle",
    "build_stage4_pd_plan",
    "build_stage4_pd_case_matrix",
    "build_stage4_fused_ir0",
    "build_stage4_separated_ir0",
    "build_stage4_segmented_state_transfers",
    "validate_stage4_segmented_state_transfers",
    "build_stage4_planned_state_transfers",
    "build_stage4_state_transfers",
    "validate_stage4_planned_state_transfers",
    "validate_stage4_state_transfers",
    "schedule_bundle",
    "schedule_profile",
    "schedule_stage4",
    "schedule_train_forward",
    "place_bundle",
    "place_ir0",
    "place_train_forward_ir0",
    "place_stage4_carrier",
    "place_stage4_ir0",
    "validate_placement_against",
    "DenseIR0Validator",
    "DenseLogicalBundleValidator",
    "FusionSemanticValidator",
    "PassManager",
    "PipelinePhase",
]
