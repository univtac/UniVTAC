from tacex_assets.robots.franka.franka_gsmini_gripper_uipc_high_res import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_xensews_gripper_uipc import (
    FRANKA_PANDA_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_gf225_gripper_uipc import (
    FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)

from typing import Literal

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from ..sensors.tactile import TactileCfg, create_tactile_cfg


# Positive indentation limits used by task-level safety checks.  These are
# press depths in millimetres, never distances from the tactile camera.
SAFE_PRESS_DEPTH_LIMIT_MM = {
    "gsmini": 8.0,
    "gf225": 7.0,
    "xensews": 5.0,
}

DEFAULT_ARM_STIFFNESS = 1_250_000.0
DEFAULT_ARM_DAMPING = 2_500.0


def _apply_franka_arm_pd(
    robot: ArticulationCfg,
    stiffness: float,
    damping: float,
) -> ArticulationCfg:
    """Configure the implicit-PD gains used by arm position control."""
    if stiffness <= 0:
        raise ValueError("arm_stiffness must be positive.")
    if damping < 0:
        raise ValueError("arm_damping must be non-negative.")

    arm_actuators = {"panda_shoulder", "panda_forearm"}
    robot.actuators = {
        name: actuator.replace(stiffness=stiffness, damping=damping)
        if name in arm_actuators else actuator
        for name, actuator in robot.actuators.items()
    }
    return robot


@configclass
class RobotCfg:
    robot: ArticulationCfg = None
    tactiles: list[TactileCfg] = []

    gripper_offset: float = 0.131 # in m
    gripper_max_qpos: float = 0.039 # in m

    tactile_far_plane: float = 30.0 # raw camera far plane; proximity only, in mm
    adaptive_grasp_depth_threshold: float = 0.5 # positive press depth in mm
    contact_threshold: tuple[float, float] = (0.1, 0.5) # positive press-depth hysteresis band in mm


def create_franka_gsmini_gripper(
    data_type: list[str],
    optical_backend: Literal["taxim", "pix2pix"] = "taxim",
    arm_stiffness: float = DEFAULT_ARM_STIFFNESS,
    arm_damping: float = DEFAULT_ARM_DAMPING,
):
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ),
    )
    robot = _apply_franka_arm_pd(robot, arm_stiffness, arm_damping)
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_left",
            gelpad_attachment_body_name="gelsight_mini_case_left",
            name="left_tactile",
            sensor_type="gsmini",
            data_type=data_type,
            optical_backend=optical_backend,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_right",
            gelpad_attachment_body_name="gelsight_mini_case_right",
            name="right_tactile",
            sensor_type="gsmini",
            data_type=data_type,
            optical_backend=optical_backend,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=34.0,
        adaptive_grasp_depth_threshold=0.5,
        contact_threshold=(0.1, 0.5),
    )


def create_franka_gf225_gripper(
    data_type: list[str],
    arm_stiffness: float = DEFAULT_ARM_STIFFNESS,
    arm_damping: float = DEFAULT_ARM_DAMPING,
):
    robot = FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ), 
    )
    robot = _apply_franka_arm_pd(robot, arm_stiffness, arm_damping)
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_left",
            gelpad_attachment_body_name="GF225_left",
            name="left_tactile",
            sensor_type="gf225",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_right",
            gelpad_attachment_body_name="GF225_right",
            name="right_tactile",
            sensor_type="gf225",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=29.0,
        adaptive_grasp_depth_threshold=0.2,
        contact_threshold=(0.1, 0.5),
    )


def create_franka_xensews_gripper(
    data_type: list[str],
    arm_stiffness: float = DEFAULT_ARM_STIFFNESS,
    arm_damping: float = DEFAULT_ARM_DAMPING,
):
    robot = FRANKA_PANDA_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ),
    )
    robot = _apply_franka_arm_pd(robot, arm_stiffness, arm_damping)
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/XenseWS_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/XenseWS_gelpad_left",
            gelpad_attachment_body_name="XenseWS_left",
            name="left_tactile",
            sensor_type="xensews",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/XenseWS_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/XenseWS_gelpad_right",
            gelpad_attachment_body_name="XenseWS_right",
            name="right_tactile",
            sensor_type="xensews",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.125,
        gripper_max_qpos=0.039,
        tactile_far_plane=30.0,
        adaptive_grasp_depth_threshold=0.2,
        contact_threshold=(0.1, 0.5),
    )
