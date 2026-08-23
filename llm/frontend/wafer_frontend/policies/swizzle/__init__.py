"""Pure deterministic building blocks for the Swizzle inter-die policy."""


from .semantics import *
from .topology import *
from .problem import *
from .chunking import *
from .wang_1d import *
from .meshslice_2d import *
from .enumerate import *
from .cost import *
from .calibration import *
from .decide import *
from .moe_problem import *
from .moe_unfused import *
from .moe_direct_xy import *
from .moe_comet_mesh import *
from .moe_cost import *
from .moe_problem import __all__ as _moe_problem_all
from .moe_unfused import __all__ as _moe_unfused_all
from .moe_direct_xy import __all__ as _moe_direct_xy_all
from .moe_comet_mesh import __all__ as _moe_comet_mesh_all
from .moe_cost import __all__ as _moe_cost_all
from .materialize import *
from .materialize_ir1 import *
from .materialize import __all__ as _materialize_all
from .materialize_ir1 import __all__ as _materialize_ir1_all
from .semantics import __all__ as _semantics_all
from .topology import __all__ as _topology_all
from .problem import __all__ as _problem_all
from .chunking import __all__ as _chunking_all
from .wang_1d import __all__ as _wang_all
from .meshslice_2d import __all__ as _meshslice_all
from .enumerate import __all__ as _enumerate_all
from .cost import __all__ as _cost_all
from .calibration import __all__ as _calibration_all
from .decide import __all__ as _decide_all

__all__ = list(dict.fromkeys((
    *_semantics_all,
    *_topology_all,
    *_problem_all,
    *_chunking_all,
    *_wang_all,
    *_meshslice_all,
    *_enumerate_all,
    *_cost_all,
    *_calibration_all,
    *_decide_all,
    *_moe_problem_all,
    *_moe_unfused_all,
    *_moe_direct_xy_all,
    *_moe_comet_mesh_all,
    *_moe_cost_all,
    *_materialize_all,
    *_materialize_ir1_all,
)))
