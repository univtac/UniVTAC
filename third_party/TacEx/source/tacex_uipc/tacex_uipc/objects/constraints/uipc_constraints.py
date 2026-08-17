from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from uipc import Animation, builtin, view
from uipc.constitution import SoftPositionConstraint, SoftTransformConstraint

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from ..uipc_object import UipcObject


@configclass
class UipcConstraintCfg:
    """Configuration for a whole-body UIPC target constraint."""

    constraint_strength_ratio: float = 100.0
    """Constraint stiffness relative to the object's mass/stiffness scale."""

    constraint_type: type | None = None
    """Either :class:`SoftTransformConstraint` or :class:`SoftPositionConstraint`."""


class UipcConstraint:
    """A switchable whole-body target used by UniVTAC actors.

    The constitution must be applied before the mesh is inserted into the UIPC
    scene, while its animation is evaluated after insertion.  Keeping both
    operations here prevents callers from accidentally creating a constraint
    that exists in the constitution table but never receives targets.
    """

    cfg: UipcConstraintCfg

    def __init__(self, cfg: UipcConstraintCfg, uipc_object: UipcObject) -> None:
        cfg.validate()
        if cfg.constraint_type not in (SoftTransformConstraint, SoftPositionConstraint):
            raise ValueError(
                "constraint_type must be SoftTransformConstraint or SoftPositionConstraint, "
                f"got {cfg.constraint_type!r}"
            )

        self.cfg = cfg.copy()
        self.uipc_object = uipc_object
        self._active = False
        self._target: np.ndarray | None = None

        constraint = self.cfg.constraint_type()
        if self.cfg.constraint_type is SoftTransformConstraint:
            strength = np.array(
                [self.cfg.constraint_strength_ratio, self.cfg.constraint_strength_ratio], dtype=np.float64
            )
        else:
            strength = self.cfg.constraint_strength_ratio
        constraint.apply_to(self.uipc_object.uipc_meshes[0], strength)

        self._create_animation()

    @property
    def active(self) -> bool:
        return self._active

    def set_target(self, target: np.ndarray) -> None:
        """Enable the constraint and replace its target."""
        self._target = np.asarray(target, dtype=np.float64).copy()
        self._active = True

    def disable(self) -> None:
        """Disable the constraint without discarding its last target."""
        self._active = False

    def _create_animation(self) -> None:
        animator = self.uipc_object.uipc_sim.scene.animator()
        animator.insert(self.uipc_object.uipc_scene_objects[0], self._animate)

    def _animate(self, info: Animation.UpdateInfo) -> None:
        geo_slots = info.geo_slots()
        if not geo_slots:
            return

        geo = geo_slots[0].geometry()
        if self.cfg.constraint_type is SoftTransformConstraint:
            constrained = view(geo.instances().find(builtin.is_constrained))
            constrained[:] = int(self._active)
            if self._active and self._target is not None:
                aim = view(geo.instances().find(builtin.aim_transform))
                aim[:] = self._target.reshape(aim.shape)
        else:
            constrained = view(geo.vertices().find(builtin.is_constrained))
            constrained[:] = int(self._active)
            if self._active and self._target is not None:
                aim = view(geo.vertices().find(builtin.aim_position))
                aim[:] = self._target.reshape(aim.shape)
