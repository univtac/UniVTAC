from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None


import warp as wp

from isaaclab.utils import configclass

wp.init()


from ..uipc_object import UipcObjectCfg
from .uipc_deformable_object import UipcDeformableObject

if TYPE_CHECKING:
    pass


@configclass
class UipcDeformableObjectCfg(UipcObjectCfg):
    class_type: type = UipcDeformableObject

    # contact_model:

    @configclass
    class StableNeoHookeanCfg:
        # class_type = StableNeoHookean
        youngs_modulus: float = 0.01
        """
        in [MPa]
        """

        poisson_rate: float = 0.49
        """ Poission Rate

        Has to be < 0.5.
        """

    constitution_cfg: StableNeoHookeanCfg = StableNeoHookeanCfg()

    debug_deformation_vis = False
