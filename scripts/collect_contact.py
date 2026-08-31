import sys
sys.path.append('.')

import time
import torch
import argparse
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher
from typing import TYPE_CHECKING
from omegaconf import OmegaConf

from envs.utils.env_parser import (
    add_config_override_argument,
    create_task_env,
    load_task_config,
    timing_plan,
)

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect data"
)
parser.add_argument(
    "task",
    type=str,
    help="Task file name",
)
parser.add_argument(
    "config",
    type=str,
    help="Config file name",
    default="contact.yml"
)
add_config_override_argument(parser)
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1

task_config, task_config_file = load_task_config(
    args_cli.config,
    args_cli.config_overrides,
)

if timing_plan(task_config, "collect").render_hz == 0:
    args_cli.livestream = 2

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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

def run(task: 'BaseTask', episode_num, use_seed):
    suc_num, seed = 0, 0
    suc_map = []
    
    if use_seed:
        suc_map_path = task.save_root / 'suc_map.txt'
        if suc_map_path.exists():
            with open(suc_map_path, 'r') as f:
                suc_map = f.read().strip().split(' ')
            suc_num = sum([1 for s in suc_map if s == '1'])
            seed = len(suc_map)
            log(f"Use seed with {suc_num} successful episodes. Starting from seed {seed}.")

    mean_steps = 0.0
    while suc_num < episode_num:
        try:
            start_t = time.perf_counter()
            task.reset(seed=seed)
            task.play_once()
            cost_t = time.perf_counter() - start_t
        except Exception as e:
            log(f"[{suc_num:<3d}] Seed {seed} failed with error: {traceback.format_exc()}")
            suc_map.append('0')
            task.clean_cache(mean_steps=mean_steps, result='error')
        else:
            # if task.plan_success and task.check_success():
            if task.check_success():
                task.save_to_hdf5()
                log(f"[{suc_num:<3d}] Seed {seed} success in {cost_t:.2f} s.\n"
                    f"steps: {task.step_count:<5d}, save frames: {task.save_count:<5d}.\n")
                suc_num += 1
                suc_map.append('1')
                if mean_steps > 0: 
                    mean_steps = ((suc_num - 1) * mean_steps + task.step_count) / suc_num
                else:
                    mean_steps = task.step_count
                task.clean_cache(mean_steps=mean_steps)
            else:
                log(f"[{suc_num:<3d}] Seed {seed} failed in {cost_t:.2f} s.\n"
                    f"Plan {task.plan_success}, Check {task.check_success()}")
                suc_map.append('0')
                task.clean_cache(mean_steps=mean_steps, result='fail')
        
        with open(task.save_root / 'suc_map.txt', 'w') as f:
            f.write(' '.join([s for s in suc_map]))
        
        seed += 1
    
    log(f'Complete collection, success rate: {suc_num}/{seed} ({(suc_num / seed) * 100:.2f}%)')

    task.close()
    simulation_app.close()

def main():
    global args_cli, task_config, task_config_file, log_path
    task_file_name = args_cli.task

    import os
    prism_name = os.environ.get('PRISM_NAME', 'Default')
    save_dir = (
        Path(str(task_config.collect_settings.save_root_dir))
        / task_config_file.stem
        / prism_name
    )
    
    init_start = time.perf_counter()
    task_env = create_task_env(
        task_file_name,
        task_config,
        task_config_file.stem,
        "collect",
        device=args_cli.device,
        save_dir=save_dir,
    )
    task: 'BaseTask' = task_env.task
    env_cfg = task_env.env_cfg
    init_cost = time.perf_counter() - init_start
    
    log_path = task.save_root / f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}.log"
    log(f"Task Name: {task_file_name}")
    log(f"Config Name: {task_config_file.stem}")
    log(f"Task Config:\n{OmegaConf.to_yaml(task_config, resolve=True)}")
    log(f"Timing Plan: {task_env.timing}")
    log(f"Env Config: \n{env_cfg}\n{'-' * 20}\n")
    log(f"Init cost {init_cost:.2f} seconds, device: {env_cfg.sim.device}")
    run(
        task,
        episode_num=task_config.collect_settings.episode_num,
        use_seed=task_config.collect_settings.use_seed,
    )

if __name__ == "__main__":
    main()
