from shutil import ExecError
import sys

sys.path.append(".")
sys.path.append(f"./policy")

import time
import json
import yaml
import torch
import argparse
import traceback
from pathlib import Path
from typing import Literal
from omegaconf import OmegaConf

from envs.utils.env_parser import (
    add_config_override_argument,
    create_task_env,
    load_task_config,
)

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Eval Policy"
)
parser.add_argument(
    "task_name",
    type=str,
    help="Task name",
)
parser.add_argument(
    "task_config",
    type=str,
    help="Task config",
)
parser.add_argument(
    "deploy_config",
    type=str,
    help="Deploy file name",
)
parser.add_argument(
    "--expert_check",
    action='store_true',
    help="Whether to do expert check before eval"
)
parser.add_argument(
    "--start_seed",
    type=int,
    default=-1
)
parser.add_argument(
    "--max_seed",
    type=int,
    default=-1
)
parser.add_argument(
    "--total_num",
    type=int,
    default=100
)
parser.add_argument(
    "--print_only",
    action='store_true',
)
add_config_override_argument(parser)
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 2
args_cli.num_envs = 1

task_config, task_config_file = load_task_config(
    args_cli.task_config,
    args_cli.config_overrides,
)

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
import importlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from envs._base_task import BaseTask
    from policy._base_policy import BasePolicy

log_path = Path('./log')
def log(msg):
    global log_path, args_cli
    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    if not args_cli.print_only:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    print(msg)

def eval_policy(
    task: 'BaseTask', policy: 'BasePolicy', expert_check,
    start_seed, max_seed, test_total_num, instructions, instruciton_type:Literal['seen', 'unseen']='seen'
):
    test_num, succ_num, seed = 0, 0, start_seed

    seed_path = task.save_root.parent / 'seeds.json'
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    if seed_path.exists():
        with open(seed_path, 'r') as f:
            seed_status = json.load(f)
    else:
        seed_status = {}
 
    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        if not seed_status.get(str(seed), True):
            seed += 1
            continue
        
        if expert_check and str(seed) not in seed_status:
            test_start = time.perf_counter()
            task.mode = 'eval_test'
            try:
                task.reset(seed=seed)
                task.play_once()
                if not task.check_success() or not task.plan_success:
                    raise ExecError(f'seed {seed} Expert check failed, check {task.check_success()}, plan {task.plan_success}.')
                else:
                    seed_status[seed] = True
                    with open(seed_path, 'w') as f:
                        json.dump(seed_status, f)
                test_cost = time.perf_counter() - test_start
                task.clean_cache(result='test_success')
                log(f'Expert check succ, seed {seed}, cost {test_cost:.2f}s')
            except Exception as e:
                test_cost = time.perf_counter() - test_start
                log(f'Expert check failed, seed {seed}, cost {test_cost:.2f}s, with exception {e}')
                task.clean_cache(result='test_fail')
                seed_status[seed] = False
                with open(seed_path, 'w') as f:
                    json.dump(seed_status, f)
                seed += 1
                continue
        test_num += 1

        succ = False
        eval_start = time.perf_counter()
        task.mode = 'eval'
        try:
            task.reset(seed=seed, instructions=instructions[instruciton_type])
            task.mean_steps = task.cfg.step_lim
            policy.reset()
            while task.take_action_cnt < task.cfg.step_lim:
                observation = task._get_observations()
                policy.eval(task, observation)
                if task.eval_success:
                    succ = True
                    break
                if task.check_early_stop():
                    break
        except Exception as e:
            log(f"[{test_num:<3d}] Seed {seed} occurred exception: {e}\n{traceback.format_exc()}")
            succ_status = 'error'
            task.clean_cache(result=succ_status)
            test_num -= 1
        else:
            eval_cost = time.perf_counter() - eval_start
            
            if succ:
                succ_num += 1
            succ_status = 'success' if succ else 'failed'
            task.clean_cache(result=succ_status)
            log(f"[{test_num:<3d}] Seed {seed} {succ_status} after {eval_cost:.2f} s.\n"
                f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
                f"Instruction: {task.instruction}\n"
                f"Total {succ_num}/{test_num}({succ_num/test_num*100:.2f}%) success.")
        finally:
            seed += 1
    
    return {
        'test_num': test_num,
        'succ_num': succ_num
    }

def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file

task_module, policy_module = None, None
def main():
    global args_cli, task_module, policy_module, log_path

    task_file_name = args_cli.task_name
    deploy_config, deploy_config_file = get_config(
        args_cli.deploy_config, default_root=Path(__file__).parent.parent / 'policy', type='yaml'
    )
    policy_name = deploy_config['policy_name']
    deploy_config['task_name'] = task_file_name
    deploy_config['task_config'] = task_config_file.stem
    import os
    if os.environ.get('TRAIN_CONFIG'):
        deploy_config['train_config_name'] = os.environ['TRAIN_CONFIG']
    if os.environ.get('EP_NUM'):
        deploy_config['expert_data_num'] = os.environ['EP_NUM']
 
    deploy_config['instuction_file'] = deploy_config.get('instuction_file', task_file_name)
    if deploy_config['instuction_file'] is not None:
        instructions, _ = get_config(
            deploy_config['instuction_file'], default_root=Path(__file__).parent.parent / 'instructions', type='json'
        )
    else:
        instructions = {'seen': ['Empty'], 'unseen': ['Empty']}

    policy_module = importlib.import_module(f"policy.{policy_name}")
    
    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')

    save_dir = (
        Path(str(task_config.replay_settings.save_root_dir))
        / policy_name
        / task_file_name
        / deploy_config_file.stem
        / curr_time
    )
    seed = deploy_config.get("seed", 0)

    init_start = time.perf_counter()
    policy:BasePolicy = policy_module.Policy(deploy_config)
    policy_init_cost = time.perf_counter() - init_start

    init_start = time.perf_counter()
    task_env = create_task_env(
        task_file_name,
        task_config,
        task_config_file.stem,
        "eval",
        device=args_cli.device,
        save_dir=save_dir,
    )
    task: BaseTask = task_env.task
    task_init_cost = time.perf_counter() - init_start
    
    log_path = task.save_root / f"log.log"
    log(f"Task Name: {task_file_name}")
    log(f"Task Config: {task_config_file.absolute()}")
    log(f"Resolved Task Config:\n{OmegaConf.to_yaml(task_config, resolve=True)}")
    log(f"Timing Plan: {task_env.timing}")
    log(f"Eval Config: {json.dumps(deploy_config, ensure_ascii=False, indent=4)}\n{'-' * 20}\n") 
    log(f"Task init finish in {task_init_cost:.2f} seconds.")
    log(f"Policy init finish in {policy_init_cost:.2f} seconds.")

    results = eval_policy(
        task=task, policy=policy,
        expert_check=args_cli.expert_check,
        start_seed=1000000 * (1 + seed) if args_cli.start_seed == -1 else args_cli.start_seed,
        max_seed=args_cli.max_seed,
        test_total_num=args_cli.total_num,
        instructions=instructions,
        instruciton_type=deploy_config.get("instruction_type", "seen")
    )
    log(f"Final Result: {results['succ_num']}/{results['test_num']}({results['succ_num']/results['test_num']*100:.2f}%) success.")
    
    task.close()
    policy.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
