from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal

import numpy as np
import omni.kit.commands
import omni.usd
import torch
from pxr import Sdf, UsdGeom, UsdShade
from uipc.constitution import SoftPositionConstraint, SoftTransformConstraint
from uipc.geometry import extract_surface

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from tacex_uipc import (
    UipcConstraintCfg,
    UipcDeformableObject,
    UipcDeformableObjectCfg,
    UipcRigidObject,
    UipcRigidObjectCfg,
)

from .._global import OBJECTS_ROOT, TEXTURES_ROOT
from .transforms import Pose, estimate_rigid_transform

if TYPE_CHECKING:
    from .._base_task import BaseTask


BodyType = Literal["rigid", "deformable"]
MotionType = Literal["dynamic", "kinematic"]


@configclass
class ActorCfg(AssetBaseCfg):
    """UniVTAC task actor configuration.

    ``body_type`` selects the UIPC constitution family. ``motion_type`` is an
    independent task-level semantic: dynamic actors respond to forces, while
    kinematic rigid actors keep the pose written by the task and still
    participate in contact.
    """

    class_type: type | None = None
    name: str = "actor"
    asset: str | None = None
    body_type: BodyType = "rigid"
    motion_type: MotionType = "dynamic"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    extents: tuple[float, float, float] = (0.1, 0.1, 0.1)
    mass_density: float = 1e3
    mesh_cfg: object | None = None
    usd_mesh_prim_name: str | None = None
    constitution_cfg: object | None = None
    constraint_strength_ratio: float = 100.0

    target_points: list = []
    contact_points: list = []
    functional_points: list = []
    orientation_points: list = []


class Actor:
    """Task-facing wrapper over a rigid or deformable UIPC object."""

    cfg: ActorCfg

    def __init__(self, task: BaseTask, cfg: ActorCfg):
        if cfg.body_type not in ("rigid", "deformable"):
            raise ValueError(f"Unsupported actor body_type: {cfg.body_type!r}")
        if cfg.motion_type not in ("dynamic", "kinematic"):
            raise ValueError(f"Unsupported actor motion_type: {cfg.motion_type!r}")
        if cfg.body_type != "rigid" and cfg.motion_type == "kinematic":
            raise ValueError("Kinematic motion is currently supported only for rigid actors.")

        self.task = task
        self.cfg = cfg
        self.body_type: BodyType = cfg.body_type
        self.motion_type: MotionType = cfg.motion_type
        self.init_pose = Pose(cfg.init_state.pos, cfg.init_state.rot)
        self._initial_transform = self.init_pose.to_transformation_matrix()
        self._pending_hard_pose: Pose | None = None
        self.next_pts: np.ndarray | None = None
        self.next_mat: np.ndarray | None = None

        common = dict(
            prim_path=cfg.prim_path,
            spawn=cfg.spawn,
            init_state=cfg.init_state,
            collision_group=cfg.collision_group,
            debug_vis=cfg.debug_vis,
            mesh_cfg=cfg.mesh_cfg,
            mass_density=cfg.mass_density,
            usd_mesh_prim_name=cfg.usd_mesh_prim_name,
        )
        if self.body_type == "rigid":
            is_kinematic = self.motion_type == "kinematic"
            constitution_cfg = (
                UipcRigidObjectCfg.AffineBodyConstitutionCfg()
                if cfg.constitution_cfg is None
                else deepcopy(cfg.constitution_cfg)
            )
            # Motion semantics belong to ActorCfg. Preserve custom material
            # parameters while preventing a second, conflicting source of truth.
            constitution_cfg.kinematic = is_kinematic
            body_cfg = UipcRigidObjectCfg(
                **common,
                constitution_cfg=constitution_cfg,
                constraint_cfg=UipcConstraintCfg(
                    constraint_type=SoftTransformConstraint,
                    constraint_strength_ratio=cfg.constraint_strength_ratio,
                ),
            )
            self.body: UipcRigidObject | UipcDeformableObject = UipcRigidObject(body_cfg, task.uipc_sim)
        else:
            constitution_cfg = cfg.constitution_cfg or UipcDeformableObjectCfg.StableNeoHookeanCfg()
            body_cfg = UipcDeformableObjectCfg(
                **common,
                constitution_cfg=constitution_cfg,
                constraint_cfg=UipcConstraintCfg(
                    constraint_type=SoftPositionConstraint,
                    constraint_strength_ratio=cfg.constraint_strength_ratio,
                ),
            )
            self.body = UipcDeformableObject(body_cfg, task.uipc_sim)

        surface = extract_surface(self.body.uipc_meshes[0]).positions().view().reshape(-1, 3).copy()
        nodal = self.body.uipc_meshes[0].positions().view().reshape(-1, 3).copy()
        if self.body_type == "rigid":
            self._initial_surface_vertices = self._transform_points(surface, self._initial_transform)
            self._initial_nodal_vertices = nodal
        else:
            self._initial_surface_vertices = surface
            self._initial_nodal_vertices = nodal

        # The scene owns the wrapper so it updates exactly once through the
        # normal UipcInteractiveScene path.
        task.scene.uipc_objects[cfg.name] = self

    @classmethod
    def from_usd_file(
        cls,
        task: BaseTask,
        name: str,
        asset_path: str | Path,
        pose: Pose,
        body_type: BodyType = "rigid",
        constitution_cfg=None,
        density: float = 1e3,
        *,
        motion_type: MotionType = "dynamic",
    ) -> Actor:
        asset_path = Path(asset_path)
        if not asset_path.is_absolute():
            asset_path = OBJECTS_ROOT / asset_path
        asset_path = asset_path.absolute()

        cfg = ActorCfg(
            name=name,
            asset=str(asset_path),
            body_type=body_type,
            motion_type=motion_type,
            prim_path=f"/World/envs/env_.*/{name}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=pose.p, rot=pose.q),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset_path),
                mass_props=sim_utils.MassPropertiesCfg(density=density),
            ),
            constitution_cfg=constitution_cfg,
            mass_density=density,
        )
        return cls(task, cfg)

    @staticmethod
    def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        return points @ transform[:3, :3].T + transform[:3, 3]

    def _target_vertices(self, pose: Pose, points: np.ndarray) -> np.ndarray:
        delta = pose.to_transformation_matrix() @ np.linalg.inv(self._initial_transform)
        return self._transform_points(points, delta)

    def set_pose(self, pose: Pose, soft: bool = False) -> None:
        """Set an actor pose using a hard state write or a soft target.

        Args:
            pose: Desired world-frame pose.
            soft: If false, restore the initial/rest mesh at ``pose``, clear all
                velocities and disable the target constraint. If true, retain
                dynamics and drive the actor using its UIPC target constraint.
        """
        target_transform = pose.to_transformation_matrix()
        self.next_mat = target_transform
        self.next_pts = self._target_vertices(pose, self._initial_surface_vertices)

        if soft:
            if self.motion_type == "kinematic":
                raise ValueError(
                    "Kinematic actors cannot use a soft pose target; use a hard "
                    "set_pose call to reposition them."
                )
            self._pending_hard_pose = None
            if self.body_type == "rigid":
                self.body.constraint.set_target(target_transform)
            else:
                self.body.constraint.set_target(self._target_vertices(pose, self._initial_nodal_vertices))
            return

        self.body.constraint.disable()
        if not self.body.is_initialized:
            self._pending_hard_pose = pose.clone()
            return
        self._write_hard_pose(pose)

    def _write_hard_pose(self, pose: Pose) -> None:
        if self.body_type == "rigid":
            pose_tensor = pose.totensor(dtype=torch.float64, device=self.body.device).reshape(1, 7)
            self.body.write_pose_to_sim(pose_tensor, zero_velocity=True)
        else:
            positions = self._target_vertices(pose, self._initial_nodal_vertices)
            positions = torch.as_tensor(positions, dtype=torch.float64, device=self.body.device).unsqueeze(0)
            state = torch.cat((positions, torch.zeros_like(positions)), dim=-1)
            self.body.write_nodal_state_to_sim(state)
        self._pending_hard_pose = None

    def remove_animate(self) -> None:
        """Compatibility alias: stop any active soft pose target."""
        self.body.constraint.disable()

    def update(self, dt: float) -> None:
        if self._pending_hard_pose is not None and self.body.is_initialized:
            self._write_hard_pose(self._pending_hard_pose)
        self.body.update(dt)

    def get_pose(self, type: Literal["pose", "matrix"] = "pose"):
        if self.body_type == "rigid":
            mat = np.asarray(self.body.geo_slot_list[0].geometry().transforms().view()).reshape(-1, 4, 4)[0]
        else:
            mat = estimate_rigid_transform(self._initial_surface_vertices, self.vertices)
        return mat if type == "matrix" else Pose.from_matrix(mat)

    @property
    def vertices(self) -> np.ndarray:
        all_points = self.task.uipc_sim.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
        offsets = self.task.uipc_sim._surf_vertex_offsets
        return all_points[offsets[self.body.obj_id - 1] : offsets[self.body.obj_id]]

    @property
    def points(self):
        return {
            "contact": self.cfg.contact_points,
            "target": self.cfg.target_points,
            "functional": self.cfg.functional_points,
            "orientation": self.cfg.orientation_points,
        }

    def get_point(
        self,
        type: Literal["contact", "target", "functional", "orientation"],
        idx: int,
        ret: Literal["pose", "matrix"] = "pose",
    ):
        points = self.points[type]
        if idx >= len(points):
            raise IndexError(f"Index {idx} out of range for {type} points.")
        world_matrix = self.get_pose("matrix") @ points[idx]
        return world_matrix if ret == "matrix" else Pose.from_matrix(world_matrix)

    def iter_point(
        self,
        type: Literal["contact", "target", "functional", "orientation"],
        ret: Literal["pose", "matrix"] = "pose",
    ) -> Generator:
        for idx in range(len(self.points[type])):
            yield self.get_point(type, idx, ret)

    def register_point(
        self, pose: Pose, type: Literal["contact", "target", "functional", "orientation"]
    ) -> int:
        local_matrix = np.linalg.inv(self.get_pose("matrix")) @ pose.to_transformation_matrix()
        self.points[type].append(local_matrix)
        return len(self.points[type]) - 1

    def set_texture(self, mdl_path: str, rng=None) -> None:
        self._set_texture(str(self.body._prim_view.prims[0].GetPath()), mdl_path, rng=rng)

    @staticmethod
    def _set_texture(prim_path: str, mdl_path: str, rng=None) -> None:
        rng = np.random if rng is None else rng

        def find_mesh(prim):
            if prim.GetTypeName() == "Mesh":
                return prim
            for child in prim.GetChildren():
                mesh_prim = find_mesh(child)
                if mesh_prim is not None:
                    return mesh_prim
            return None

        if mdl_path == "random":
            mdl_path = rng.choice(list(TEXTURES_ROOT.glob("*.mdl")))
        mdl_path = Path(mdl_path)

        stage = omni.usd.get_context().get_stage()
        mesh = UsdGeom.Mesh(find_mesh(stage.GetPrimAtPath(prim_path)))
        material_path = f"/World/envs/env_0/ground_plate/Looks/{mdl_path.stem}"
        if not stage.GetPrimAtPath(material_path).IsValid():
            success, _ = omni.kit.commands.execute(
                "CreateMdlMaterialPrimCommand",
                mtl_url=str(mdl_path),
                mtl_name=mdl_path.stem,
                mtl_path=material_path,
            )
            if not success:
                return

        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        shader = UsdShade.Shader(material.GetPrim().GetChild("Shader"))
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


class ActorManager:
    def __init__(self, task: BaseTask):
        self.task = task
        self.actors: dict[str, Actor] = {}

    def add_from_usd_file(
        self,
        name: str,
        asset_path: str,
        pose: Pose,
        body_type: BodyType = "rigid",
        constitution_cfg=None,
        density: float = 1e3,
        *,
        motion_type: MotionType = "dynamic",
    ) -> Actor:
        actor = Actor.from_usd_file(
            self.task,
            name,
            asset_path,
            pose,
            body_type=body_type,
            motion_type=motion_type,
            constitution_cfg=constitution_cfg,
            density=density,
        )
        self.actors[actor.cfg.name] = actor
        return actor

    def _reset_idx(self, rng=None) -> None:
        for actor in self.actors.values():
            if self.task.cfg.random_texture:
                actor.set_texture("random", rng=rng)

    def update(self, dt: float) -> None:
        # Actors are registered in UipcInteractiveScene and updated there.
        pass

    def remove_animate(self) -> None:
        for actor in self.actors.values():
            actor.remove_animate()

    def get_observations(self) -> dict[str, torch.Tensor]:
        return {name: actor.get_pose().totensor() for name, actor in self.actors.items()}
