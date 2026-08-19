import torch
from typing import TYPE_CHECKING, Literal
from envs.utils import data
import numpy as np
from tacex import GelSightSensor, GelSightSensorCfg
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gf225.gf225_cfg import GF225Cfg
from tacex.simulation_approaches.fem_based import ManiSkillSimulatorCfg
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg

from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg

from tacex_uipc import (
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcDeformableObject,
    UipcDeformableObjectCfg,
)

from ..utils.transforms import *

if TYPE_CHECKING:
    from .._base_task import BaseTask
    from tacex_uipc.sim import UipcIsaacAttachmentsCfg, UipcSim
    from tacex_uipc import UipcInteractiveScene

@configclass
class TactileCfg:
    name: str = 'tactile_sensor'
    sensor_cfg = None
    gelpad_cfg: UipcDeformableObjectCfg = None
    gelpad_attachment_cfg: UipcIsaacAttachmentsCfg = None

def create_gelsight_mini_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    name: str = "tactile_sensor",
    resolution = (320, 240),
    update_period = 1/120,
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
    optical_backend: Literal["taxim", "pix2pix"] = "taxim",
):
    from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_PIX2PIX_CFG, GELSIGHT_MINI_TAXIM_CFG
    from tacex.simulation_approaches.fem_based.sim.gelpad_info import CONSTRAIN_PTS

    if optical_backend == "taxim":
        sensor_cfg = GELSIGHT_MINI_TAXIM_CFG.copy()
    elif optical_backend == "pix2pix":
        sensor_cfg = GELSIGHT_MINI_PIX2PIX_CFG.copy()
    else:
        raise ValueError(f"Unknown GelSight optical backend: {optical_backend!r}")

    sensor_cfg = sensor_cfg.replace(
        prim_path=prim_path,
        sensor_camera_cfg=sensor_cfg.SensorCameraCfg(
            prim_name="Camera",
            resolution=resolution,
            update_period=update_period,
            data_types=["depth", "rgb"],
            clipping_range=(0.024, 0.034),
            update_latest_camera_pose=True,
        ),
        device="cuda",
        debug_vis=False,  # for rendering sensor output in the gui
        update_period=1/120,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            device="cuda",
            tactile_img_res=resolution,
            marker_shape=(9, 7),
            marker_interval=(2.40625, 2.45833),
            sub_marker_num=0,
            marker_radius=6,
            camera_to_surface=0.0283,
            real_size=(0.0266, 0.0209),
            sensor_type='gsmini',
        ),
        data_types=data_type,
    )
    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = (
        sensor_cfg.marker_motion_sim_cfg.marker_shape[0]
        * sensor_cfg.marker_motion_sim_cfg.marker_shape[1]
    )
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        tactile_img_res=resolution,
        device="cuda",
    )
    if optical_backend == "taxim":
        sensor_cfg.optical_sim_cfg.with_shadow = False
        sensor_cfg.compute_indentation_depth_class = "optical_sim"
    else:
        sensor_cfg.compute_indentation_depth_class = "default"

    cfg = TactileCfg(
        name=name,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcDeformableObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            isaaclab_rigid_body_prim_path=gelpad_prim_path.rsplit("/", 1)[0],
            attachment_point_indices=CONSTRAIN_PTS["gsmini"].tolist(),
            debug_vis=False,
        ),
    )
    return cfg

def create_gf225_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    data_type: list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    resolution = (480, 480)  # GF225 resolution
    update_period = 1/120
    
    sensor_cfg = GF225Cfg(
        prim_path=prim_path,
        sensor_camera_cfg=GF225Cfg.SensorCameraCfg(
            prim_name="Camera",
            resolution=resolution,
            update_period=update_period,
            data_types=["depth"],
            clipping_range=(0.02, 0.029),
        ),
        device="cuda",
        debug_vis=False,
        update_period=1/120,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            sub_marker_num=0,
            marker_radius=8,
            marker_shape=(9, 9),
            marker_interval=(2.0, 2.0),
            camera_to_surface=0.029,
            real_size = (0.0235, 0.0250),
            sensor_type='gf225',
        ),
        data_types=data_type
    )
    
    from tacex.simulation_approaches.mlp_fots import MLPFOTSSimulatorCfg
    from tacex_assets import TACEX_ASSETS_DATA_DIR

    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 81
    sensor_cfg.optical_sim_cfg = MLPFOTSSimulatorCfg(
        calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GF225/calibs/480x480",
        tactile_img_res=resolution,
        device="cuda",
    )
    
    cfg = TactileCfg(
        name=name,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcDeformableObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            isaaclab_rigid_body_prim_path=gelpad_attachment_prim_path or gelpad_prim_path.rsplit("/", 1)[0],
            debug_vis=False,
        ),
    )
    return cfg

def create_xensews_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    resolution = (320, 240),
    update_period = 1/120,
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    from tacex_assets.sensors.xensews.xensews_cfg import XenseWSCfg

    sensor_cfg = XenseWSCfg(
        prim_path=prim_path,
        sensor_camera_cfg=XenseWSCfg.SensorCameraCfg(
            prim_name="Camera",
            update_period=update_period,
            resolution=resolution,
            data_types=["depth", "rgb"],
            clipping_range=(0.01, 0.03),
        ),
        device="cuda",
        debug_vis=False,  # for rendering sensor output in the gui
        update_period=update_period,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            marker_shape=(20, 11),
            marker_interval=(1.15, 1.2),
            sub_marker_num=0,
            marker_radius=2,
            camera_to_surface=0.0245,
            real_size=(0.0293, 0.0175),
            sensor_type='xensews',
        ),
        data_types=data_type
    )
    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 220
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        tactile_img_res=resolution,
        device="cuda",
    )

    cfg = TactileCfg(
        name=name,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcDeformableObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            isaaclab_rigid_body_prim_path=gelpad_attachment_prim_path or gelpad_prim_path.rsplit("/", 1)[0],
            debug_vis=False,
        ),
    )
    return cfg

def create_tactile_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    sensor_type:Literal['gsmini', 'xensews', 'gf225'] = "gsmini",
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
    optical_backend: Literal["taxim", "pix2pix"] = "taxim",
) -> TactileCfg:
    if sensor_type == "gsmini":
        return create_gelsight_mini_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            name=name,
            data_type=data_type,
            optical_backend=optical_backend,
        )
    elif sensor_type == "xensews":
        return create_xensews_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            name=name,
            data_type=data_type,
        )
    elif sensor_type == "gf225":
        return create_gf225_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            gelpad_attachment_prim_path=gelpad_attachment_prim_path,
            name=name,
            data_type=data_type,
        )
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")


class VisualTactileSensor:
    def __init__(self, name:str, cfg:TactileCfg, robot, scene: 'UipcInteractiveScene', uipc_sim:'UipcSim'):
        self.cfg = cfg
        self.name = name
        self.scene = scene
        self.robot = robot
        self.uipc_sim = uipc_sim

        self.cfg.gelpad_cfg.constraint_cfg = self.cfg.gelpad_attachment_cfg
        self.gelpad = UipcDeformableObject(self.cfg.gelpad_cfg, self.uipc_sim)
        self.attachment: UipcIsaacAttachments = self.gelpad.constraint
        self.attachment.isaaclab_rigid_object = self.robot
        self.sensor = GelSightSensor(self.cfg.sensor_cfg, self.gelpad)
        # self.scene.sensors[f'tactile_{self.cfg.name}'] = self.sensor
    
    def setup(self):
        self.device = self.uipc_sim.cfg.device
        init_pts = self.gelpad._data.nodal_pos_w[self.attachment.attachment_points_idx].cpu().numpy()
        init_world_trans = self.gelpad.init_world_transform.cpu().numpy()
        self.origin_pts = (init_pts - init_world_trans[:3, 3]) @ (init_world_trans[:3, :3].T).T
        attach_pts = self.attachment.attachment_offsets
        init_trans = estimate_rigid_transform(self.origin_pts, attach_pts)
        self.attach_to_init = np.linalg.inv(init_trans)
        self.attach_to_init = torch.tensor(self.attach_to_init, dtype=torch.float64, device=self.device)

    def get_attach_pose(self):
        if type(self.attachment.isaaclab_rigid_object) is Articulation:
            # this only works when rigid body is an articulation
            # self.attachment.isaaclab_rigid_object._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self.attachment.isaaclab_rigid_object.root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            pose = poses[:, self.attachment.rigid_body_id, 0:7].clone()
        elif type(self.attachment.isaaclab_rigid_object) is RigidObject:
            # only works with rigid body
            pose = self.attachment.isaaclab_rigid_object.root_physx_view.get_transforms().clone()
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            pose = pose.view(-1, 1, 7)
            pose = pose[:, self.attachment.rigid_body_id, 0:7].clone()
        else:
            raise RuntimeError("Need an Articulation or a RigidBody object for the Isaac X UIPC attachment.")
        return Pose.from_list(pose.flatten().tolist())

    def get_init_pts(self):
        curr_attach_pose = self.get_attach_pose()
        trans_to_attach = np.linalg.inv(curr_attach_pose.to_transformation_matrix())
        trans_to_attach = torch.tensor(trans_to_attach, dtype=torch.float64, device=self.device)
        trans_to_init = self.attach_to_init @ trans_to_attach
        return self.gelpad.data.nodal_pos_w @ trans_to_init[:3, :3].T + trans_to_init[:3, 3]
 
    def update(self, dt, force_recompute=False):
        self.gelpad.update(dt=dt)
        self.sensor.update(dt=dt, force_recompute=force_recompute)
    
    def set_debug_vis(self):
        if not self.sensor.cfg.debug_vis:
            return 
        for data_type in ['marker_motion']:
            self.sensor._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
    
    def get_observations(self, data_types: list[str] = None):
        obs = {}
        if data_types is None:
            data_types = ['rgb', 'rgb_marker', 'depth', 'press_depth', 'points', 'pose', 'flow']
        for data_type in data_types:
            if data_type == 'rgb':
                obs['rgb'] = self.sensor.data.output['tactile_rgb'].squeeze(0)
            elif data_type == 'rgb_marker':
                obs['rgb_marker'] = self.sensor.data.output['marker_rgb'].squeeze(0)
            elif data_type == 'depth':
                # Legacy UniVTAC depth: raw sensor-camera distance in millimetres.
                camera_depth = self.sensor.camera.data.output['depth'][0, :, :, 0].clone()
                far_plane = self.sensor.cfg.sensor_camera_cfg.clipping_range[1]
                camera_depth = torch.nan_to_num(
                    camera_depth, nan=far_plane, posinf=far_plane, neginf=far_plane
                )
                obs['depth'] = camera_depth * 1000.0
                # Requesting legacy depth always records the new semantics too.
                obs['press_depth'] = self.get_press_depth_map()
            elif data_type == 'press_depth':
                obs['press_depth'] = self.get_press_depth_map()
            elif data_type == 'marker':
                obs['marker'] = self.sensor.data.output['marker_motion'].squeeze(0)
            elif data_type == 'points':
                obs['points'] = self.get_init_pts()
            elif data_type == 'pose':
                obs['pose'] = self.get_attach_pose().totensor()
        return obs
    
    def _reset_idx(self):
        self.init_pose_mat = self.get_attach_pose().to_transformation_matrix()
        # self.gelpad.write_vertex_positions_to_sim(vertex_positions=self.gelpad.init_vertex_pos)

    def get_marker_attachment_error(self) -> float | None:
        marker_simulator = self.sensor.marker_motion_simulator
        if marker_simulator is None or not hasattr(marker_simulator, 'marker_motion_sim'):
            return None
        return marker_simulator.marker_motion_sim.attachment_error()

    def calibrate_marker_reference(self) -> dict | None:
        marker_simulator = self.sensor.marker_motion_simulator
        if marker_simulator is None or not hasattr(marker_simulator, 'marker_motion_sim'):
            return None
        return marker_simulator.marker_motion_sim.calibrate_reference()
    
    def get_press_depth_map(self):
        """Return positive gel indentation in millimetres (zero means no press)."""
        return (-self.sensor.data.output['height_map'].squeeze(0)).clamp_min(0.0)

    def get_press_depth(self):
        """Return the maximum positive indentation in millimetres."""
        return max(0.0, self.sensor.indentation_depth.squeeze().item())

    def get_min_camera_depth(self) -> tuple[float, float]:
        """Return nearest visible geometry and the far plane in millimetres.

        Raw camera distance is deliberately kept separate from press depth.  It
        is only a proximity signal for switching the gripper to its slow mode;
        contact targets and stopping conditions must use ``get_press_depth``.
        """
        far_plane_m = self.sensor.cfg.sensor_camera_cfg.clipping_range[1]
        camera_depth = self.sensor.camera.data.output['depth'][0, :, :, 0]
        camera_depth = torch.nan_to_num(
            camera_depth,
            nan=far_plane_m,
            posinf=far_plane_m,
            neginf=far_plane_m,
        ).clamp(max=far_plane_m)
        return camera_depth.amin().item() * 1000.0, far_plane_m * 1000.0

    def has_relative_gel_height_map(self) -> bool:
        """Whether ``height_map`` is expressed relative to the gel surface.

        The migrated TacEx representation is zero without indentation and
        negative below the nominal gel surface.  Older TacEx versions exposed
        positive absolute camera distance instead.  Detecting the contract at
        the sensor boundary keeps adaptive grasp compatible with both forms.
        """
        height_map = self.sensor.data.output['height_map']
        gel_height_mm = self.sensor.cfg.gelpad_dimensions.height * 1000.0
        finite = height_map[torch.isfinite(height_map)]
        if finite.numel() == 0:
            return False
        tolerance_mm = 1e-3
        return bool(
            finite.amax().item() <= tolerance_mm
            and finite.amin().item() >= -gel_height_mm - tolerance_mm
        )

    def get_nominal_gel_surface_camera_depth(self) -> float:
        """Return nominal camera-to-gel-surface distance in millimetres."""
        camera_cfg = self.sensor.cfg.sensor_camera_cfg
        return (
            camera_cfg.clipping_range[0]
            + self.sensor.cfg.gelpad_dimensions.height
        ) * 1000.0

class TactileManager:
    def __init__(self, cfg_list: list[TactileCfg], task:'BaseTask'):
        self.task = task
        self.scene = task.scene
        self.uipc_sim = task.uipc_sim
        self.robot = task._robot_manager.robot
        
        self.tactiles = {
            cfg.name: VisualTactileSensor(
                cfg.name, cfg, self.robot, self.scene, self.uipc_sim
            ) for cfg in cfg_list
        }

    def update(self, dt, force_recompute=False):
        for tact in self.tactiles.values():
            tact.update(dt=dt, force_recompute=force_recompute)
 
    def set_debug_vis(self, debug_vis):
        if not debug_vis: return
        for tact in self.tactiles.values():
            tact.set_debug_vis()

    def get_observations(self, data_types: list[str] = None):
        obs = {}
        for name, tact in self.tactiles.items():
            obs[name] = tact.get_observations(data_types)
        return obs

    def get_press_depth(self):
        """Return one positive indentation value per tactile sensor in millimetres."""
        self.task._update_render()
        depth = []
        for tact in self.tactiles.values():
            depth.append(tact.get_press_depth())
        return torch.tensor(depth, dtype=torch.float32, device=self.task.device)

    def get_grasp_control_depths(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return press depth plus raw-camera proximity data for gripper control.

        All tensors are one value per tactile sensor and use millimetres.
        ``press_depth`` is the only quantity suitable for contact decisions.
        ``min_camera_depth`` and ``far_plane_depth`` only decide whether nearby
        geometry is visible and the gripper should therefore move slowly.
        """
        self.task._update_render()
        press_depth = []
        min_camera_depth = []
        far_plane_depth = []
        for tact in self.tactiles.values():
            press_depth.append(tact.get_press_depth())
            nearest, far_plane = tact.get_min_camera_depth()
            min_camera_depth.append(nearest)
            far_plane_depth.append(far_plane)
        tensor_args = {'dtype': torch.float32, 'device': self.task.device}
        return (
            torch.tensor(press_depth, **tensor_args),
            torch.tensor(min_camera_depth, **tensor_args),
            torch.tensor(far_plane_depth, **tensor_args),
        )

    def has_relative_gel_height_maps(self) -> bool:
        """Return true only when every tactile uses real gel-relative depth."""
        return all(tact.has_relative_gel_height_map() for tact in self.tactiles.values())

    def get_nominal_gel_surface_camera_depths(self) -> torch.Tensor:
        """Return one nominal camera-to-gel distance per tactile sensor."""
        return torch.tensor(
            [
                tact.get_nominal_gel_surface_camera_depth()
                for tact in self.tactiles.values()
            ],
            dtype=torch.float32,
            device=self.task.device,
        )

    def _reset_idx(self):
        for tact in self.tactiles.values():
            tact._reset_idx()

    def get_marker_attachment_errors(self) -> dict[str, float]:
        errors = {}
        for name, tactile in self.tactiles.items():
            error = tactile.get_marker_attachment_error()
            if error is not None:
                errors[name] = error
        return errors

    def calibrate_marker_references(self) -> dict[str, dict]:
        reports = {}
        for name, tactile in self.tactiles.items():
            report = tactile.calibrate_marker_reference()
            if report is not None:
                reports[name] = report
        return reports

    def setup(self):
        for tact in self.tactiles.values():
            tact.setup()
