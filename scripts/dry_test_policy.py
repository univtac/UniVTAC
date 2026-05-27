import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Literal

sys.path.append(".")
sys.path.append("./policy")


parser = argparse.ArgumentParser(
    description="Dry-test a deploy policy without launching IsaacSim."
)
parser.add_argument("task_name", type=str, help="Task name")
parser.add_argument("task_config", type=str, help="Task config")
parser.add_argument("deploy_config", type=str, help="Deploy file name")
parser.add_argument(
    "--expert_check",
    action="store_true",
    help="Accepted for parity with eval_policy.py; always succeeds in dry mode.",
)
parser.add_argument("--start_seed", type=int, default=-1)
parser.add_argument("--max_seed", type=int, default=-1)
parser.add_argument("--total_num", type=int, default=100)
parser.add_argument("--print_only", action="store_true")

# Common AppLauncher args accepted by eval_policy.py. They are parsed here so
# existing launch configs can point at dry_test_policy.py unchanged.
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--enable_cameras", action="store_true")
parser.add_argument("--livestream", type=int, default=-1)
parser.add_argument("--num_envs", type=int, default=1)

args_cli, unknown_args = parser.parse_known_args()

log_path = Path("./log")


def _torch():
    import torch

    return torch


def log(msg):
    global log_path, args_cli
    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    if not args_cli.print_only:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    print(msg)


def get_config(file, default_root: Path, type: Literal["yaml", "json"]):
    if type == "yaml":
        import yaml

        if file.endswith(".yml") or file.endswith(".yaml"):
            file = Path(file)
        else:
            file = default_root / f"{file}.yml"
        with open(file, "r") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file

    if file.endswith(".json"):
        file = Path(file)
    else:
        file = default_root / f"{file}.json"
    with open(file, "r") as f:
        config = json.load(f)
    return config, file


class DryTaskCfg:
    def __init__(self, task_config: dict, save_dir: Path):
        self.step_lim = int(task_config.get("step_lim", task_config.get("dry_step_lim", 10)))
        self.save_dir = save_dir
        self.obs_data_type = task_config.get("observations", {})


class DryTask:
    def __init__(self, cfg: DryTaskCfg, mode: str = "eval", device: str | None = None):
        torch = _torch()

        self.cfg = cfg
        self.mode = mode
        self.save_root = cfg.save_dir
        self.save_root.mkdir(parents=True, exist_ok=True)

        if device is not None and device.startswith("cuda") and not torch.cuda.is_available():
            log(f"Requested device '{device}' is unavailable; dry task uses CPU.")
            device = "cpu"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.take_action_cnt = 0
        self.step_count = 0
        self.eval_success = False
        self.plan_success = True
        self.mean_steps = cfg.step_lim
        self.instruction = "Empty"
        self.actions = []

    def reset(self, seed=None, instructions=None):
        self.seed = seed
        self.take_action_cnt = 0
        self.step_count = 0
        self.eval_success = False
        self.plan_success = True
        self.actions = []
        self.instruction = self._pick_instruction(seed, instructions)
        return self._get_observations()

    def _pick_instruction(self, seed, instructions):
        if not instructions:
            return "Empty"
        if isinstance(instructions, str):
            return instructions
        if isinstance(instructions, (list, tuple)):
            if len(instructions) == 0:
                return "Empty"
            index = 0 if seed is None else int(seed) % len(instructions)
            return str(instructions[index])
        return str(instructions)

    def _zeros_image(self, height: int, width: int):
        torch = _torch()

        return torch.zeros((height, width, 3), dtype=torch.uint8, device=self.device)

    def _get_observations(self):
        torch = _torch()

        visual = {
            "rgb": self._zeros_image(270, 480),
        }
        tactile = {
            "rgb": self._zeros_image(240, 320),
            "rgb_marker": self._zeros_image(240, 320),
            "marker": torch.zeros((240, 320, 3), dtype=torch.float32, device=self.device),
            "depth": torch.zeros((240, 320), dtype=torch.float32, device=self.device),
            "pose": torch.zeros((7,), dtype=torch.float32, device=self.device),
        }
        joint = torch.zeros((9,), dtype=torch.float32, device=self.device)
        return {
            "observation": {
                "head": dict(visual),
                "wrist": dict(visual),
            },
            "tactile": {
                "left_tactile": dict(tactile),
                "right_tactile": dict(tactile),
                "left_gsmini": dict(tactile),
                "right_gsmini": dict(tactile),
            },
            "embodiment": {
                "ee": torch.zeros((7,), dtype=torch.float32, device=self.device),
                "joint": joint,
            },
            "joint_action": joint.clone(),
        }

    def take_action(self, action, action_type="qpos"):
        torch = _torch()

        if isinstance(action, torch.Tensor):
            action_for_log = action.detach().cpu().reshape(-1).tolist()
        else:
            action_for_log = torch.as_tensor(action).detach().cpu().reshape(-1).tolist()
        self.actions.append({"action_type": action_type, "action": action_for_log})
        self.take_action_cnt += 1
        self.step_count += 1
        self.eval_success = False
        return True, False

    def check_early_stop(self):
        return self.take_action_cnt >= self.cfg.step_lim

    def check_success(self):
        return False

    def play_once(self):
        self.plan_success = True

    def clean_cache(self, result):
        log(
            f"DryTask seed {getattr(self, 'seed', None)} finished with {result}; "
            f"actions={self.take_action_cnt}."
        )

    def close(self):
        pass


def eval_policy(
    task: DryTask,
    policy,
    expert_check,
    start_seed,
    max_seed,
    test_total_num,
    instructions,
    instruciton_type: Literal["seen", "unseen"] = "seen",
):
    test_num, succ_num, seed = 0, 0, start_seed

    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        if expert_check:
            task.mode = "eval_test"
            task.reset(seed=seed)
            task.play_once()
            log(f"Dry expert check succ, seed {seed}.")

        test_num += 1
        succ = False
        eval_start = time.perf_counter()
        task.mode = "eval"
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
            task.clean_cache(result="error")
            test_num -= 1
        else:
            eval_cost = time.perf_counter() - eval_start
            if succ:
                succ_num += 1
            succ_status = "success" if succ else "failed"
            task.clean_cache(result=succ_status)
            log(
                f"[{test_num:<3d}] Seed {seed} {succ_status} after {eval_cost:.2f} s.\n"
                f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
                f"Instruction: {task.instruction}\n"
                f"Total {succ_num}/{test_num}({succ_num/test_num*100:.2f}%) success."
            )
        finally:
            seed += 1

    return {
        "test_num": test_num,
        "succ_num": succ_num,
    }


def main():
    global log_path

    if unknown_args:
        print(f"[dry_test_policy] Ignoring Isaac/AppLauncher args: {' '.join(unknown_args)}")

    task_file_name = args_cli.task_name
    task_config, task_config_file = get_config(
        args_cli.task_config,
        default_root=Path(__file__).parent.parent / "task_config",
        type="yaml",
    )
    deploy_config, deploy_config_file = get_config(
        args_cli.deploy_config,
        default_root=Path(__file__).parent.parent / "policy",
        type="yaml",
    )

    policy_name = deploy_config["policy_name"]
    deploy_config["task_name"] = task_file_name
    deploy_config["task_config"] = task_config_file.stem

    import os

    if os.environ.get("TRAIN_CONFIG"):
        deploy_config["train_config_name"] = os.environ["TRAIN_CONFIG"]
    if os.environ.get("EP_NUM"):
        deploy_config["expert_data_num"] = os.environ["EP_NUM"]

    deploy_config["instuction_file"] = deploy_config.get("instuction_file", task_file_name)
    if deploy_config["instuction_file"] is not None:
        instructions, _ = get_config(
            deploy_config["instuction_file"],
            default_root=Path(__file__).parent.parent / "instructions",
            type="json",
        )
    else:
        instructions = {"seen": ["Empty"], "unseen": ["Empty"]}

    policy_module = importlib.import_module(f"policy.{policy_name}")

    curr_time = time.strftime(r"%Y-%m-%d_%H:%M:%S")
    save_dir = Path("dry_eval_result") / policy_name / task_file_name / deploy_config_file.stem / curr_time
    dry_cfg = DryTaskCfg(task_config, save_dir)

    seed = deploy_config.get("seed", 0)

    init_start = time.perf_counter()
    policy = policy_module.Policy(deploy_config)
    policy_init_cost = time.perf_counter() - init_start

    init_start = time.perf_counter()
    task = DryTask(dry_cfg, mode="eval", device=args_cli.device)
    task_init_cost = time.perf_counter() - init_start

    log_path = task.save_root / "log.log"
    log(f"Task Name: {task_file_name}")
    log(f"Task Config: {task_config_file.absolute()}")
    log(f"Dry Eval Config: {json.dumps(deploy_config, ensure_ascii=False, indent=4)}\n{'-' * 20}\n")
    log(f"Dry task init finish in {task_init_cost:.2f} seconds.")
    log(f"Policy init finish in {policy_init_cost:.2f} seconds.")
    log(f"Dry step limit: {task.cfg.step_lim}")

    try:
        results = eval_policy(
            task=task,
            policy=policy,
            expert_check=args_cli.expert_check,
            start_seed=1000000 * (1 + seed) if args_cli.start_seed == -1 else args_cli.start_seed,
            max_seed=args_cli.max_seed,
            test_total_num=args_cli.total_num,
            instructions=instructions,
            instruciton_type=deploy_config.get("instruction_type", "seen"),
        )
        rate = results["succ_num"] / results["test_num"] * 100 if results["test_num"] else 0.0
        log(f"Final Result: {results['succ_num']}/{results['test_num']}({rate:.2f}%) success.")
    finally:
        task.close()
        policy.close()


if __name__ == "__main__":
    main()
