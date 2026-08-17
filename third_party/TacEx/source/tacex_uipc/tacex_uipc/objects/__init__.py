# Export the base object before concrete bodies. Deformable bodies import
# tacex_assets, whose GelSight FEM config refers back to UipcObject during
# package initialization.
from .uipc_object import UipcObject, UipcObjectCfg
from .constraints import UipcConstraint, UipcConstraintCfg, UipcIsaacAttachments, UipcIsaacAttachmentsCfg
from .deformable import UipcDeformableObject, UipcDeformableObjectCfg, UipcDeformableObjectData
from .rigid import UipcRigidObject, UipcRigidObjectCfg, UipcRigidObjectData
