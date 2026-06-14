# Task Creation Guide

Task creation covers asset generation, tetrahedral meshing, point annotation, initial placement, scripted actions, success checks, and early-stop checks. A new task usually corresponds to two classes in `envs/<task_name>.py`: `TaskCfg` and `Task`. The `Task` class inherits from `BaseTask` and implements the following methods as needed:

- `create_actors`: Create all assets used by this task.
- `_reset_actors`: Randomize asset states on every `reset`.
- `pre_move`: Preparation actions before data collection starts.
- `_play_once`: The actual task action used for data collection or demonstration.
- `check_success`: Success condition.
- `check_early_stop`: Early failure condition used during evaluation.

## Asset Generation

Assets should first be modeled in Blender, CAD software, or another modeling tool, then exported to a format supported by `scripts/convert.py`, such as `obj`, `stl`, `fbx`, or `glb`. Before exporting, use the following conventions:

- Use meters as the unit, or verify the export scale in the modeling software to avoid incorrect simulation sizes.
- Use Z-up coordinates, and place the asset origin at a position that is convenient for later placement, grasping, and success checks.
- Merge meshes that do not need independent control. The conversion script also enables `merge_all_meshes`.
- Surface meshes should be closed, have consistent normals, and avoid obvious self-intersections. Otherwise, tetrahedral meshing can fail. Blender's 3D Print Toolbox can be useful for making a mesh manifold.
- File names should be valid USD identifiers. English letters, numbers, and underscores are recommended.

Converted `.usd` files are usually placed under `assets/objects/` and referenced by file name in task code, for example:

```python
self.block = self._actor_manager.add_from_usd_file(
    name="blue_block",
    asset_path="BlueBlock.usd",
    pose=Pose([0.7, 0.0, 0.005], [1, 0, 0, 0]),
)
```

## Tetrahedral Meshing

`scripts/convert.py` converts regular meshes to USD and writes tetrahedral mesh attributes required by UIPC soft-body simulation onto the USD Mesh prim:

- `tet_points`
- `tet_indices`
- `tet_surf_points`
- `tet_surf_indices`

These attributes are saved with the `.usd` file and are used directly when `Actor.from_usd_file` loads the asset. Basic usage:

```bash
python scripts/convert.py \
  --input assets/objects/ipt/MyObject.obj \
  --output assets/objects/MyObject.usd \
  --show
```

`--input` can be a single file or a directory. Directory mode batch-converts `obj`, `stl`, `fbx`, and `glb` files inside the directory. `--output` can be a target USD file or an output directory. Common option:

- `--show`: Visualize the generated tetrahedral mesh with trimesh, so you can check for missing surfaces, flipped surfaces, or meshes that are too coarse or too dense.

There are two meshing backends:

- `tetgen`: The current default backend. The script first cleans the surface with `pymeshfix`, then calls `tetgen.TetGen(...).tetrahedralize()` to generate tetrahedra, and extracts the surface mesh from `pyvista`.
- `MeshGenerator`: The built-in backend, using `MeshGenerator` and `TetMeshCfg` from `scripts/utils/mesh_gen.py`. This branch is used when `gen_tet(..., backend=...)` receives a backend name other than `tetgen`. Mesh quality and density can be adjusted through `stop_quality`, `max_its`, `edge_length_r`, and `epsilon_r`.

If `--show` reveals poor mesh quality, switch backends or tune the built-in backend's `TetMeshCfg` parameters.

## Point Annotation

Points in a task are local coordinate frames with poses. They describe grasp points, target points, functional points, and other task-specific references. `Actor` supports four point types:

- `contact`: Grasp points. `atom.grasp_actor` selects grasp poses from this type by default.
- `target`: Target points, often used as intermediate references during task planning.
- `functional`: Functional points, representing the actual interaction part of an object, such as a hammer head, insertion tip, or bottle mouth.
- `orientation`: Orientation points, used to describe object direction.

When registering a point, pass a `Pose` in the current world frame. `Actor.register_point` automatically converts it into the asset's local frame before saving it:

```python
grasp_pose = construct_grasp_pose(
    target_pose.p,
    [0, 0, 1],
    [1, 0, 0],
)
grasp_idx = self.prism.register_point(grasp_pose, type="contact")
```

When using a point, `Actor.get_point` converts the saved local point back into the world frame according to the object's current pose:

```python
world_pose = self.prism.get_point(type="contact", idx=grasp_idx)
world_matrix = self.prism.get_point(type="contact", idx=grasp_idx, ret="matrix")
```

You can also preconfigure `contact_points`, `functional_points`, and similar matrix lists in `ActorCfg`. Most tasks dynamically register points in `pre_move` or `_play_once` according to the randomized object pose. Dynamic registration is useful because random offsets, grasp heights, and functional point poses can be written directly into the current episode.

Common `Pose` helper methods:

- `add_bias(vec, coord="local")`: Translate along local or world coordinates.
- `add_rotation(euler, coord="local")`: Apply an additional Euler-angle rotation.
- `add_offset(noise)`: Add a random pose offset.
- `rebase(to_coord=pose)`: Express the pose in another coordinate frame. This is commonly used in success checks.
- `to_transformation_matrix()`: Convert to a 4x4 transformation matrix.

## Initial Placement

`create_actors` is called only once during the environment lifecycle. It creates every asset that may be used by the task. Since dynamic asset loading at runtime is not currently supported, every object that may appear in any branch must be created here. Assets that are not used immediately can be placed far away from the workspace and moved into the task area later in `_reset_actors`.

Example:

```python
def create_actors(self):
    self.slot = self._actor_manager.add_from_usd_file(
        name="slot",
        asset_path="TestTubeHoleSlot.usd",
        pose=Pose([0.6, 0.0, 0.002], [1, 0, 0, 0]),
        density=1e5,
    )
    self.prism = self._actor_manager.add_from_usd_file(
        name="prism",
        asset_path="TestTube.usd",
        pose=Pose([0.4, 0.0, 0.005], [1, 0, 0, 0]),
        density=10,
    )
```

`_reset_actors` is called on every `reset`. It is used to randomize positions, poses, category choices, and episode-level variables. If `_reset_actors` is not implemented, assets keep the initial poses specified in `create_actors`. A common pattern is to use `create_noise` to generate a random offset, then write the pose back with `set_pose`:

```python
def _reset_actors(self):
    block_noise = self.create_noise([0.02, 0.05, 0.0], [0, 0, np.pi / 3])
    block_pose = Pose([0.7, 0.0, 0.005], [1, 0, 0, 0]).add_offset(block_noise)
    self.block.set_pose(block_pose)
```

If multiple randomized objects must not intersect or be too close, sample in a loop until the distance constraint is satisfied:

```python
while True:
    hammer_noise = self.create_noise([0.02, 0.05, 0.0], [0, 0, np.pi / 3])
    hammer_pose = base_hammer_pose.add_offset(hammer_noise)
    if np.linalg.norm(hammer_pose.p - block_pose.p) > 0.1:
        break
self.hammer.set_pose(hammer_pose)
```

`_reset_actors` can also record episode state needed later, such as the selected category, target object, target pose, or labels written to `metadata`.

## Scripted Actions

Task actions are mainly written in `pre_move` and `_play_once`.

`pre_move` is the preparation stage before data collection. It is not saved as collected data, but it also runs during evaluation. It is suitable for stabilizing the scene, opening the gripper, pre-grasping an object, moving an object to the initial interaction pose, and similar setup actions.

`_play_once` is the actual data-collection or demonstration stage. In collection mode, observation saving starts from this method.

Actions are executed through `self.move(...)` and `self.delay(...)`. Common atomic actions are available under `self.atom`:

- `self.atom.grasp_actor(actor, contact_point_id=..., pre_dis=..., dis=..., is_close=True)`: Move to a grasp point and optionally close the gripper.
- `self.atom.place_actor(actor, target_pose, functional_point_id=..., pre_dis=..., dis=..., is_open=True, constrain="align")`: Place an object, or a functional point on the object, at a target pose.
- `self.atom.move_by_displacement(x=..., y=..., z=..., xyz_coord="world" | "local" | Pose, rpy=...)`: Translate or rotate relative to the current end-effector pose.
- `self.atom.move_to_pose(target_pose)`: Move the end effector directly to a target pose.
- `self.atom.open_gripper(pos=1.0)` / `self.atom.close_gripper(pos=0.0)`: Open or close the gripper.
- `self.atom.back_to_origin()`: Return the robot to its initial pose.

Example flow:

```python
def pre_move(self):
    self.delay(10)
    self.move(self.atom.open_gripper(0.5))

    target_pose = self.prism.get_pose().add_bias([0.0, 0.0, 0.04])
    grasp_pose = construct_grasp_pose(target_pose.p, [0, 0, 1], [1, 0, 0])
    grasp_idx = self.prism.register_point(grasp_pose, type="contact")

    self.move(self.atom.grasp_actor(
        self.prism,
        contact_point_id=grasp_idx,
        pre_dis=0.04,
        dis=0.0,
    ))
    self.move(self.atom.move_by_displacement(z=0.05))

def _play_once(self):
    self.move(self.atom.place_actor(
        self.prism,
        target_pose=self.target_pose,
        pre_dis=0.04,
        dis=0.0,
        is_open=False,
    ))
    self.delay(20, is_save=False)
```

Common `self.move` parameters:

- `tag`: Label the current action segment. The value is saved to the observation's `atom` field.
- `is_save`: Whether to save observations for this action segment.
- `delay`: Whether to automatically delay several steps after each action.
- `constraint_pose`: Constraint mask passed to the motion planner, for example to restrict specific degrees of freedom. A common value `[1, 1, 1, 0, 0, 0]` constrains rotation and only plans translation. `[1, 1, 1, 1, 1, 0]` only allows the gripper to move along its local z axis, which is the gripper approach direction, and is often used for insertion tasks. The stricter the constraint, the more likely planning is to fail.
- `time_dilation_factor`: Adjust the execution speed of the planned action.
- `gripper_depth_threshold`: Override the tactile depth threshold used by adaptive grasping.

If `self.move` planning fails, `self.plan_success` is set to `False`, and later actions stop executing.

## Success and Early-Stop Checks

`check_success` determines whether the task has succeeded. In collection mode, it usually runs only once after `_play_once` finishes. In evaluation mode, it runs after every policy action. Common checks include:

- Whether the target object's position error relative to the target pose is below a threshold.
- Whether the target object's orientation axis is aligned with the expected direction.
- Whether the object is still in the gripper, or has been inserted, placed, or stabilized.
- Whether necessary `metadata` has been recorded for post-hoc failure analysis.

Example:

```python
def check_success(self):
    rel_pose = self.prism.get_pose().rebase(self.target_pose)
    self.metadata["rel_pose"] = rel_pose.tolist()
    return (
        np.all(np.abs(rel_pose.p) < np.array([0.02, 0.02, 0.01]))
        and np.dot(rel_pose.to_transformation_matrix()[:3, 2], [0, 0, 1]) > 0.965
    )
```

To check whether a functional point has reached the target, use `get_point` to retrieve the current world pose of the functional point, then rebase it to the target pose:

```python
functional_pose = self.hammer.get_point("functional", self.function_idx)
rel_pose = functional_pose.rebase(self.place_pose)
return np.all(np.abs(rel_pose.p) < [0.02, 0.02, 0.02])
```

`check_early_stop` runs step-by-step only in evaluation mode. It does not run in collection mode. It ends obviously failed trajectories early to save evaluation resources. It is suitable for detecting:

- The object has slipped out of the gripper.
- Tactile depth or contact state is clearly abnormal.
- The object has moved too far away from the target area.
- The pose has become impossible to recover to a successful state.

Example:

```python
def check_early_stop(self):
    prism_inhand_pose = self.prism.get_pose().rebase(
        self._robot_manager.get_gripper_center_pose()
    )
    inhand_bias = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])
    if inhand_bias > 0.04:
        self.metadata["early_stop"] = True
        self.metadata["inhand_bias"] = float(inhand_bias)
        return True
    return False
```

Keep both `check_success` and `check_early_stop` lightweight. They should only read the current simulation state and return a boolean value; they should not execute actions that modify the simulation state.
