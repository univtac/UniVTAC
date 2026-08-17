"""Headless phase-one smoke test for the Isaac Sim 5.1 migration."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=("taxim", "pix2pix"), default="taxim")
parser.add_argument("--output-dir", type=Path, default=Path("/tmp/univtac-isaac51-smoke"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from pxr import UsdGeom
from uipc import builtin, view

import omni.usd

import tacex_uipc  # noqa: F401 - fail loudly instead of optional-import masking
from envs.grasp_classify import Task, TaskCfg


def _rigid_velocity(actor) -> np.ndarray:
    actor.body._state_accessor.copy_to(actor.body._state_geo)
    velocity = view(actor.body._state_geo.instances().find(builtin.velocity))
    return np.asarray(velocity).copy()


def main() -> None:
    cfg = TaskCfg()
    cfg.scene.num_envs = 1
    cfg.tactile_sensor_type = "gsmini"
    cfg.tactile_optical_backend = args.backend
    cfg.save_dir = args.output_dir / args.backend
    cfg.save_frequency = 0
    cfg.video_frequency = 0
    cfg.render_frequency = 0

    task = None
    try:
        task = Task(cfg, mode="eval_test")
        assert task.num_envs == 1
        assert len(task._tactile_manager.tactiles) == 2

        update_count = 0
        update_render_meshes = task.uipc_sim.update_render_meshes

        def counted_update_render_meshes():
            nonlocal update_count
            update_count += 1
            return update_render_meshes()

        task.uipc_sim.update_render_meshes = counted_update_render_meshes
        task._update_render()
        assert update_count == 1, f"expected one UIPC/Fabric copy, got {update_count}"

        stage = omni.usd.get_context().get_stage()
        optical_models = []
        for name, tactile in task._tactile_manager.tactiles.items():
            sensor = tactile.sensor
            camera_path = sensor.cfg.prim_path.replace("env_.*", "env_0") + "/Camera"
            camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
            assert camera and camera.GetPrim().IsValid(), f"missing USD camera: {camera_path}"
            assert sensor.camera_cfg.spawn is None

            rgb = sensor.data.output["tactile_rgb"]
            assert rgb.shape == (1, 240, 320, 3), (name, rgb.shape)
            assert rgb.dtype == torch.uint8, (name, rgb.dtype)
            assert 0 <= int(rgb.min()) <= int(rgb.max()) <= 255
            optical_models.append(getattr(sensor.optical_simulator, "generator_model", None))

            print(
                f"{name}: camera={camera_path}, spawn=None, "
                f"focal={camera.GetFocalLengthAttr().Get()}, rgb={tuple(rgb.shape)} {rgb.dtype}"
            )

        if args.backend == "pix2pix":
            from tacex.simulation_approaches.pix2pix import Pix2PixSimulator

            assert optical_models[0] is optical_models[1], "the two sensors did not share the Pix2Pix model"
            assert len(Pix2PixSimulator._MODEL_CACHE) == 1

        actor = task.rough_prism
        assert actor.body_type == "rigid"
        actor.set_pose(actor.init_pose, soft=True)
        assert actor.body.constraint.active

        actor.body.write_velocity_to_sim(
            torch.ones((1, 4, 4), dtype=torch.float64, device=actor.body.device)
        )
        actor.set_pose(actor.init_pose, soft=False)
        assert not actor.body.constraint.active
        assert np.allclose(_rigid_velocity(actor), 0.0), "hard set_pose did not clear rigid velocity"

        # Exercise an actual Franka motion. A stale USD camera pose collapses
        # the moving FEM marker grid to a small block even though Taxim RGB is
        # still rendered correctly through Fabric.
        task.reset(seed=0)
        task._update_render()
        for name, tactile in task._tactile_manager.tactiles.items():
            marker_motion = tactile.sensor.data.output["marker_motion"]
            assert marker_motion.shape == (1, 2, 63, 2), (name, marker_motion.shape)
            current_markers = marker_motion[0, 1]
            marker_span = current_markers.amax(dim=0) - current_markers.amin(dim=0)
            assert float(marker_span[0]) > 100.0 and float(marker_span[1]) > 80.0, (
                name,
                marker_span,
            )
            print(f"{name}: marker_span_after_motion={marker_span.tolist()}")

        print(f"PASS backend={args.backend}: 2x GelSight Mini, Actor API, one render copy")
    except BaseException:
        # SimulationApp.close() can replace Kit's active exception hook during
        # shutdown, so emit the original failure before closing the app.
        traceback.print_exc()
        raise
    finally:
        if task is not None:
            task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
