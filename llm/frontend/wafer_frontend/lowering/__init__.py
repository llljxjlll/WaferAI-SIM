"""Frozen lowering/linking interfaces."""

from .context import LoweringContext
from .coarse import NaiveCoarseLowering
from .isa_region import NaiveIsaRegionLowering
from .linker import NaiveManifestLinker
from .lifecycle import add_fixed_sram_lifecycle
from .lite_moe import (
    lower_lite_moe_compute_unit,
    lower_lite_moe_dte_unit,
    lower_lite_moe_n6_intent,
    lower_lite_moe_state_load_unit,
    validate_lite_moe_fragment,
)
from .lite_moe_backward import (
    lower_lite_moe_backward,
    validate_lite_moe_backward_fragments,
)
from .lite_moe_backward_linker import link_lite_moe_backward_manifest
from .lite_train_dp4_linker import link_s2_lite_dp4_tree_ar_manifest
from .standalone import NaiveStandaloneCollectiveLowering
from .state import NaiveStateDmaLowering
from .state_transfer import NaiveStateTransferLowering
from .interfaces import (
    CoarseLowering,
    IsaRegionLowering,
    ManifestLinker,
    StateDmaLowering,
    StateTransferLowering,
    StandaloneCollectiveLowering,
)

__all__ = [
    "LoweringContext",
    "NaiveCoarseLowering",
    "NaiveIsaRegionLowering",
    "NaiveManifestLinker",
    "NaiveStandaloneCollectiveLowering",
    "NaiveStateDmaLowering",
    "NaiveStateTransferLowering",
    "add_fixed_sram_lifecycle",
    "lower_lite_moe_compute_unit",
    "lower_lite_moe_dte_unit",
    "lower_lite_moe_n6_intent",
    "lower_lite_moe_state_load_unit",
    "validate_lite_moe_fragment",
    "lower_lite_moe_backward",
    "validate_lite_moe_backward_fragments",
    "link_lite_moe_backward_manifest",
    "link_s2_lite_dp4_tree_ar_manifest",
    "CoarseLowering",
    "IsaRegionLowering",
    "ManifestLinker",
    "StateDmaLowering",
    "StateTransferLowering",
    "StandaloneCollectiveLowering",
]
from .lite_moe_dp4 import (
    build_lite_moe_dp4_tape_buffer_abis,
    lower_lite_moe_dp4_compute_unit,
    lower_lite_moe_dp4_dte_unit,
    lower_lite_moe_dp4_infer,
    lower_lite_moe_dp4_state_load_unit,
    lower_lite_moe_dp4_tape_copy,
    lower_lite_moe_dp4_train_forward_tapes,
    rebase_lite_moe_dp4_infer_fragments,
    validate_lite_moe_dp4_infer_fragment,
)
from .lite_moe_dp4_backward import (
    lower_lite_moe_dp4_backward,
    validate_lite_moe_dp4_backward_fragments,
)
from .lite_moe_dp4_linker import (
    link_lite_moe_dp4_infer_manifest,
    link_lite_moe_dp4_train_forward_manifest,
)
from .lite_moe_dp4_backward_linker import link_lite_moe_dp4_backward_manifest
from .lite_moe_dp4 import __all__ as _lite_moe_dp4_all
from .lite_moe_dp4_backward import __all__ as _lite_moe_dp4_backward_all
from .lite_moe_dp4_linker import __all__ as _lite_moe_dp4_linker_all
from .lite_moe_dp4_backward_linker import __all__ as _lite_moe_dp4_backward_linker_all

__all__ += [
    *_lite_moe_dp4_all,
    *_lite_moe_dp4_backward_all,
    *_lite_moe_dp4_linker_all,
    *_lite_moe_dp4_backward_linker_all,
]
from .swizzle import *
from .swizzle import __all__ as _swizzle_all
from .swizzle_abi import *
from .swizzle_abi import __all__ as _swizzle_abi_all
from .swizzle_linker import *
from .swizzle_linker import __all__ as _swizzle_linker_all
from .swizzle_standard import *
from .swizzle_standard import __all__ as _swizzle_standard_all

__all__ += [
    name
    for name in (
        *_swizzle_all,
        *_swizzle_abi_all,
        *_swizzle_linker_all,
        *_swizzle_standard_all,
    )
    if name not in __all__
]
from .swizzle_unfused import *
from .swizzle_unfused import __all__ as _swizzle_unfused_all
from .swizzle_unfused_standard import *
from .swizzle_unfused_standard import __all__ as _swizzle_unfused_standard_all

__all__ = list(dict.fromkeys((
    *__all__,
    *_swizzle_unfused_all,
    *_swizzle_unfused_standard_all,
)))
from .swizzle_meshslice_standard import *
from .swizzle_meshslice_standard import __all__ as _swizzle_meshslice_standard_all

__all__ = list(dict.fromkeys((
    *__all__,
    *_swizzle_meshslice_standard_all,
)))
from .moe_swizzle_calibration_standard import (
    build_moe_swizzle_calibration_standard_linked_program,
)

__all__ = list(dict.fromkeys((
    *__all__,
    "build_moe_swizzle_calibration_standard_linked_program",
)))
