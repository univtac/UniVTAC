from .envs import UipcInteractiveScene, UipcRLEnv
from .objects import (
    UipcConstraint,
    UipcConstraintCfg,
    UipcDeformableObject,
    UipcDeformableObjectCfg,
    UipcDeformableObjectData,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcRigidObject,
    UipcRigidObjectCfg,
    UipcRigidObjectData,
)
from .sim import UipcSim, UipcSimCfg

# The UI/debug-draw extension is not enabled by Isaac Lab's headless rendering
# experience. Core UIPC imports must remain usable without it.
try:
    from .ui_extension import *  # noqa: F403
except ImportError:
    pass
