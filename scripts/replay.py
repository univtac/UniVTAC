import sys

sys.path.append(".")
sys.path.append('../')

import os
import time
import json
import torch
import argparse
import traceback
import numpy as np
import h5py
from pathlib import Path
from omegaconf import OmegaConf

from envs.utils.env_parser import (
    FIXED_POST_REPLAY_DELAY_STEPS,
    add_config_override_argument,
    collection_data_dir,
    create_task_env,
    load_task_config,
)

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Replay Data"
)
parser.add_argument(
    "task_name",
    type=str,
    help="Task name",
)
parser.add_argument(
    "task_config",
    type=str,
    help="Task name",
)
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
)
parser.add_argument(
    "--start-index",
    type=int,
    default=0,
    help="Zero-based index of the first episode to replay after sorting by file name.",
)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=None,
    help=(
        "Replay only the listed dataset seeds, preserving dataset order. "
        "For example: --seeds 0 2 3."
    ),
)
add_config_override_argument(parser)
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 0 if args_cli.headless else 2
args_cli.num_envs = 1

task_config, task_config_file = load_task_config(
    args_cli.task_config,
    args_cli.config_overrides,
)

if args_cli.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from typing import TYPE_CHECKING
from envs.utils.data import HDF5Handler
if TYPE_CHECKING:
    from envs._base_task import BaseTask

log_path = Path('./log')
def log(msg):
    global log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def replay(
    task: 'BaseTask',
    seed,
    data_path: Path,
    action_stride: int,
    replay_force: bool,
    post_replay_delay_steps: int,
):
    eval_start = time.perf_counter()
    task.reset(seed=seed)

    succ = False
    # Replay only needs the robot state. Loading every observation would decode all
    # camera frames and consume several GB for some of the original episodes.
    with h5py.File(data_path, "r") as hdf5_file:
        recorded_actor_names = [
            name for name in task._actor_manager.actors
            if f'actor/{name}' in hdf5_file
        ]
    traj_data = HDF5Handler().load_hdf5(
        data_path,
        data_paths=[
            ('embodiment/joint', None),
            ('embodiment/ee', None),
        ] + [(f'actor/{name}', None) for name in recorded_actor_names],
    )
    # Keep the complete recorded state for diagnostics. The Franka controller takes
    # seven arm joints plus one gripper command, while the observation contains both
    # finger joints (nine values in total).
    recorded_qpos_list = torch.from_numpy(
        traj_data['embodiment/joint'][:, :9]
    ).to(device=task.device)
    qpos_list = recorded_qpos_list[:, :8]
    # vel_list = torch.from_numpy(traj_data['embodiment']['vel'][:, :8]).to(device=task.device)
    ee_list = torch.from_numpy(traj_data['embodiment/ee'][:, :3]).to(device=task.device)
    recorded_actor_pose_lists = {
        # UIPC actor poses are exposed as CPU tensors, so keep their recorded
        # counterparts on CPU as well. Robot state remains on the simulation device.
        name: torch.from_numpy(traj_data[f'actor/{name}'])
        for name in recorded_actor_names
    }
    initial_actor_poses = {
        name: pose.detach().cpu().tolist()
        for name, pose in task._actor_manager.get_observations().items()
    }
    replay_start_physics_step = task._physics_step_count

    required_action_count = len(range(0, qpos_list.shape[0], action_stride))
    if task.cfg.step_lim < required_action_count:
        log(
            f"Increasing replay step_lim from {task.cfg.step_lim} to "
            f"{required_action_count} so every dataset action is executed."
        )
        task.cfg.step_lim = required_action_count
    
    traj_list = []
    qpos_errors = []
    actor_translation_errors = {name: [] for name in recorded_actor_names}
    actor_orientation_errors = {name: [] for name in recorded_actor_names}
    for idx in np.arange(0, qpos_list.shape[0], action_stride):
        idx = int(idx)
        action = qpos_list[idx]

        exec_succ, eval_succ = task.take_action(
            action,
            action_type='qpos',
            force=replay_force,
            stop_on_success=False,
        )
        observation = task._get_observations()
        target_qpos = recorded_qpos_list[idx]
        real_qpos = observation['embodiment']['joint'][:9]
        qpos_error = real_qpos - target_qpos
        qpos_errors.append(qpos_error.detach().cpu())

        arm_dis = torch.abs(qpos_error[:7])
        gripper_dis = torch.abs(qpos_error[7:9])
        ee_error = observation['embodiment']['ee'][:3] - ee_list[idx]
        ee_dis = torch.abs(ee_error)
        real_actor_poses = task._actor_manager.get_observations()
        target_actor_poses = {
            name: recorded_actor_pose_lists[name][idx]
            for name in recorded_actor_names
        }
        for name in recorded_actor_names:
            translation_error = real_actor_poses[name][:3] - target_actor_poses[name][:3]
            quaternion_dot = torch.sum(
                real_actor_poses[name][3:] * target_actor_poses[name][3:]
            ).abs().clamp(max=1.0)
            orientation_error = 2.0 * torch.acos(quaternion_dot)
            actor_translation_errors[name].append(translation_error.detach().cpu())
            actor_orientation_errors[name].append(orientation_error.detach().cpu())

        if torch.any(gripper_dis > 1e-3) or torch.any(ee_dis > 1e-3):
            log(f"[{idx:3d}] arm_dis: {np.max(arm_dis.cpu().numpy())}, gripper_dis: {np.max(gripper_dis.cpu().numpy())}, ee_dis: {ee_dis.cpu().numpy()}, eval_succ: {eval_succ}, exec_succ: {exec_succ}")
        
        traj_list.append({
            'sample_index': idx,
            'target_qpos': target_qpos.cpu().tolist(),
            'applied_action': action.cpu().tolist(),
            'real_qpos': real_qpos.cpu().tolist(),
            # delta_qpos is the signed tracking error used by this replay test.
            'delta_qpos': qpos_error.cpu().tolist(),
            'qpos_error': qpos_error.cpu().tolist(),
            'qpos_abs_error': torch.abs(qpos_error).cpu().tolist(),
            'target_ee': ee_list[idx].cpu().tolist(),
            'real_ee': observation['embodiment']['ee'][:3].cpu().tolist(),
            'ee_error': ee_error.cpu().tolist(),
            'target_actor_pose': {
                name: pose.detach().cpu().tolist()
                for name, pose in target_actor_poses.items()
            },
            'real_actor_pose': {
                name: real_actor_poses[name].detach().cpu().tolist()
                for name in recorded_actor_names
            },
            # Retain the old keys so existing analysis scripts keep working.
            'target_action': action.cpu().tolist(),
            'result_qpos': observation['embodiment']['joint'][:8].cpu().tolist(),
            'result_ee': observation['embodiment']['ee'][:3].cpu().tolist(),
        })
    replay_end_physics_step = task._physics_step_count

    # ``take_action`` stops advancing physics after ``step_lim`` is reached, so
    # the old post-replay take_action loop did not actually let released objects
    # settle. Advance the simulation directly, include these frames in the video,
    # and keep checking success because a short threshold crossing is valid.
    success_during_trajectory = bool(task.eval_success)
    success_during_post_delay = False
    first_post_delay_success_step = None
    if post_replay_delay_steps > 0:
        task.logger.info(
            f"Delaying for {post_replay_delay_steps} post-replay steps"
        )
        task.atom_tag = 'delay'
        task.atom_id += 1
        with task._configured_decimation():
            for delay_step in range(1, post_replay_delay_steps + 1):
                task._step(is_save=True)
                if task.check_success():
                    success_during_post_delay = True
                    task.eval_success = True
                    if first_post_delay_success_step is None:
                        first_post_delay_success_step = delay_step
        task._update_render()
    terminal_eval_success = bool(task.check_success())
    success_latched = bool(task.eval_success or terminal_eval_success)
    task.eval_success = success_latched
    task.metadata['success_during_trajectory'] = success_during_trajectory
    task.metadata['success_during_post_delay'] = success_during_post_delay
    task.metadata['first_post_delay_success_step'] = first_post_delay_success_step
    task.metadata['terminal_success'] = terminal_eval_success
    task.metadata['success_latched'] = success_latched
    terminal_actor_poses = {
        name: pose.detach().cpu().tolist()
        for name, pose in task._actor_manager.get_observations().items()
    }
 
    seed_root = task.save_root / 'replay_traj'
    seed_root.mkdir(parents=True, exist_ok=True)
    with open(seed_root / f'{seed}.json', 'w') as f:
        json.dump(traj_list, f, indent=4)

    qpos_error_tensor = torch.stack(qpos_errors)
    abs_error = torch.abs(qpos_error_tensor)
    squared_error = torch.square(qpos_error_tensor)
    max_error_flat_index = int(torch.argmax(abs_error).item())
    max_error_frame = max_error_flat_index // abs_error.shape[1]
    max_error_joint = max_error_flat_index % abs_error.shape[1]

    def error_metrics(error_slice):
        return {
            'mae': float(torch.mean(torch.abs(error_slice)).item()),
            'rmse': float(torch.sqrt(torch.mean(torch.square(error_slice))).item()),
            'max_abs': float(torch.max(torch.abs(error_slice)).item()),
        }

    summary = {
        'seed': int(seed),
        'source_hdf5': str(data_path.resolve()),
        'action_stride': int(action_stride),
        'replay_force': replay_force,
        'arm_stiffness': task.cfg.arm_stiffness,
        'arm_damping': task.cfg.arm_damping,
        'frame_count': len(traj_list),
        'configured_decimation': int(task.cfg.decimation),
        'post_replay_delay_steps': int(post_replay_delay_steps),
        'transient_eval_success': success_during_trajectory,
        'success_during_trajectory': success_during_trajectory,
        'success_during_post_delay': success_during_post_delay,
        'first_post_delay_success_step': first_post_delay_success_step,
        'terminal_eval_success': terminal_eval_success,
        'success_latched': success_latched,
        'terminal_actor_pose': terminal_actor_poses,
        'replay_physics_step_count': int(
            replay_end_physics_step - replay_start_physics_step
        ),
        'initial_actor_pose': initial_actor_poses,
        'error_definition': 'real_qpos_after_step - target_qpos_from_hdf5',
        'joint_labels': [
            'arm_joint_0', 'arm_joint_1', 'arm_joint_2', 'arm_joint_3',
            'arm_joint_4', 'arm_joint_5', 'arm_joint_6',
            'finger_joint_0', 'finger_joint_1',
        ],
        'overall': error_metrics(qpos_error_tensor),
        'arm': error_metrics(qpos_error_tensor[:, :7]),
        'gripper': error_metrics(qpos_error_tensor[:, 7:9]),
        'per_joint': {
            f'qpos_{joint_index}': {
                'mae': float(torch.mean(abs_error[:, joint_index]).item()),
                'rmse': float(torch.sqrt(torch.mean(squared_error[:, joint_index])).item()),
                'max_abs': float(torch.max(abs_error[:, joint_index]).item()),
            }
            for joint_index in range(qpos_error_tensor.shape[1])
        },
        'largest_error': {
            'sample_index': traj_list[max_error_frame]['sample_index'],
            'joint_index': max_error_joint,
            'target_qpos': traj_list[max_error_frame]['target_qpos'][max_error_joint],
            'real_qpos': traj_list[max_error_frame]['real_qpos'][max_error_joint],
            'signed_error': traj_list[max_error_frame]['qpos_error'][max_error_joint],
            'abs_error': traj_list[max_error_frame]['qpos_abs_error'][max_error_joint],
        },
        'actor_tracking': {},
    }
    for name in recorded_actor_names:
        translation_error = torch.stack(actor_translation_errors[name])
        translation_distance = torch.linalg.vector_norm(translation_error, dim=1)
        orientation_error_deg = torch.rad2deg(
            torch.stack(actor_orientation_errors[name])
        )
        summary['actor_tracking'][name] = {
            'translation_mean_mm': float(torch.mean(translation_distance).item() * 1000.0),
            'translation_rmse_mm': float(
                torch.sqrt(torch.mean(torch.square(translation_distance))).item() * 1000.0
            ),
            'translation_max_mm': float(torch.max(translation_distance).item() * 1000.0),
            'orientation_mean_deg': float(torch.mean(orientation_error_deg).item()),
            'orientation_max_deg': float(torch.max(orientation_error_deg).item()),
            'final_target_pose': traj_list[-1]['target_actor_pose'][name],
            'final_real_pose': traj_list[-1]['real_actor_pose'][name],
        }
    with open(seed_root / f'{seed}_qpos_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    log(
        f"Seed {seed} qpos error: overall MAE={summary['overall']['mae']:.6g}, "
        f"RMSE={summary['overall']['rmse']:.6g}, "
        f"max={summary['overall']['max_abs']:.6g}; "
        f"arm MAE={summary['arm']['mae']:.6g}, "
        f"gripper MAE={summary['gripper']['mae']:.6g}."
    )

    succ = success_latched

    eval_cost = time.perf_counter() - eval_start
    succ_status = 'success' if succ else 'failed'
    task.clean_cache(result=succ_status)
    return succ_status, eval_cost

def replay_seeds(
    task: 'BaseTask',
    data,
    action_stride: int,
    replay_force: bool,
    post_replay_delay_steps: int,
):
    test_num, succ_num = 0, 0
    for seed, data_path in data:
        test_num += 1
        result, eval_cost = replay(
            task,
            seed,
            data_path,
            action_stride,
            replay_force,
            post_replay_delay_steps,
        )
        succ_num += 1 if result == 'success' else 0
        log(f"[{test_num:<3d}] Seed {seed} {result} after {eval_cost:.2f} s.\n"
        f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
        f"Instruction: {task.instruction}\n"
        f"Total {succ_num}/{test_num}({succ_num/test_num*100:.2f}%) success.")
    return {
        'test_num': test_num,
        'succ_num': succ_num
    }


def main():
    global args_cli, log_path

    task_file_name = args_cli.task_name

    init_start = time.perf_counter()
    task_env = create_task_env(
        task_file_name,
        task_config,
        task_config_file.stem,
        "eval",
        device=args_cli.device,
    )
    task: BaseTask = task_env.task
    env_cfg = task_env.env_cfg
    task_init_cost = time.perf_counter() - init_start
    
    log_path = task.save_root / f"log.log"
    log(f"Task Name: {task_file_name}")
    log(f"Task Config: {task_config_file.absolute()}")
    log(f"Resolved Config:\n{OmegaConf.to_yaml(task_config, resolve=True)}")
    log(f"Timing Plan: {task_env.timing}")
    log(f"Rendering Mode: {env_cfg.sim.render.rendering_mode}")
    log(f"Task init finish in {task_init_cost:.2f} seconds.")

    # The dataset configuration name is intentionally fixed to the active task
    # config file stem.  There is no CLI/data_config override path.
    data_root = collection_data_dir(
        task_config,
        task_file_name,
        task_config_file.stem,
    ).resolve()
    if (data_root / 'hdf5').exists():
        print(f"Found hdf5 data in {data_root / 'hdf5'}, start replaying.")
        # self collect data
        data_root = data_root / 'hdf5'
        data = sorted([(int(p.stem), p) for p in data_root.glob('*.hdf5')], key=lambda x: x[0])
    elif any(data_root.glob('*.hdf5')):
        # Some self-collected datasets store their HDF5 files directly in the
        # configuration directory instead of an additional ``hdf5`` folder.
        print(f"Found hdf5 data in {data_root}, start replaying.")
        data = sorted(
            [(int(p.stem), p) for p in data_root.glob('*.hdf5')],
            key=lambda x: x[0],
        )
    else:
        print(f"Found downloaded data in {data_root}, start replaying.")
        # dataset
        metadata_file = data_root / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        data = []
        for k, v in metadata.items():
            if (data_root / f'{k}.hdf5').exists() and 'seed' in v:
                data.append((int(v['seed']), data_root / f'{k}.hdf5'))
        data.sort(key=lambda item: int(item[1].stem))

    if args_cli.seeds is not None:
        requested_seeds = set(args_cli.seeds)
        available_seeds = {seed for seed, _ in data}
        missing_seeds = requested_seeds - available_seeds
        if missing_seeds:
            raise ValueError(
                f"Requested seeds are not present in {data_root}: "
                f"{sorted(missing_seeds)}"
            )
        data = [(seed, path) for seed, path in data if seed in requested_seeds]

    if args_cli.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    max_episodes = task_config.replay_settings.max_episodes
    action_stride = task_env.timing.action_stride
    replay_force = task_config.replay_settings.force_action
    post_replay_delay_steps = FIXED_POST_REPLAY_DELAY_STEPS

    stop_index = args_cli.start_index + max_episodes
    data = data[args_cli.start_index:stop_index]

    if not data:
        raise RuntimeError(f"No replayable HDF5 files found in {data_root}.")
 
    log(
        f"Start replaying {len(data)} seeds from {data_root} with "
        f"action_stride={action_stride}, decimation={env_cfg.decimation}, "
        f"force={replay_force}, post_replay_delay_steps="
        f"{post_replay_delay_steps}."
    )

    results = replay_seeds(
        task,
        data=data,
        action_stride=action_stride,
        replay_force=replay_force,
        post_replay_delay_steps=post_replay_delay_steps,
    )
    log(f"Final Result: {results['succ_num']}/{results['test_num']}({results['succ_num']/results['test_num']*100:.2f}%) success.")
    
    task.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
