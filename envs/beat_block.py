from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    adaptive_grasp_depth_threshold = 0.2 # positive indentation in millimetres

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)
    
    def create_actors(self):
        block_pose = Pose([0.7, 0.0, 0.005], [1, 0, 0, 0])
        hammer_pose = Pose([0.5, 0.0, 0.056], [0.55, 0.44, 0.46, 0.55])

        self.block = self._actor_manager.add_from_usd_file(
            name='blue_block',
            asset_path="BlueBlock.usd", 
            pose=block_pose
        )
        self.hammer = self._actor_manager.add_from_usd_file(
            name='hammer',
            asset_path="Hammer.usd",
            pose=hammer_pose
        )
    
    def _reset_actors(self):
        block_noise = self.create_noise([0.02, 0.05, 0.0], [0, 0, np.pi/3])
        block_pose = Pose([0.7, 0.0, 0.005], [1, 0, 0, 0]).add_offset(block_noise)

        while True:
            hammer_noise = self.create_noise([0.02, 0.05, 0.0], [0, 0, np.pi/3])
            hammer_pose = Pose([0.5, 0.0, 0.056], [0.55, 0.44, 0.46, 0.55]).add_offset(hammer_noise)
            
            if np.linalg.norm(hammer_pose.p - block_pose.p) > 0.1:
                break
 
        self.block.set_pose(block_pose)
        self.hammer.set_pose(hammer_pose)

    def pre_move(self):
        self.delay(10)

    def _play_once(self):
        hammer_pose = self.hammer.get_pose()
        grasp_target = hammer_pose.add_bias([0, -0.01, -0.015])
        grasp_mat = grasp_target.to_transformation_matrix()
        grasp_pose = construct_grasp_pose(
            grasp_target.p,
            grasp_mat[:3, 1],
            grasp_mat[:3, 2]
        )
        function_pose = hammer_pose.add_bias([0.0, -0.06, 0.08]).add_rotation([0, -np.pi/2, -np.pi/2])
        grasp_idx = self.hammer.register_point(
            pose=grasp_pose,
            type='contact'
        )
        self.function_idx = self.hammer.register_point(
            pose=function_pose,
            type='functional'
        )
        self.move(self.atom.grasp_actor(
            self.hammer, contact_point_id=grasp_idx
        ))
        
        self.move(self.atom.move_by_displacement(z=0.08))
        self.place_pose = self.block.get_pose().add_bias([0.0, 0.0, 0.05])
        
        self.move(self.atom.place_actor(
            self.hammer, self.place_pose, functional_point_id=self.function_idx, dis=0.01,
            constrain='free', is_open=False
        ))

        self.delay(30, is_save=False)

    def check_success(self):
        rel_pose = self.hammer.get_point(type='functional', idx=self.function_idx).rebase(self.place_pose)
        self.metadata['rel_pose'] = rel_pose.tolist()
        return np.all(np.abs(rel_pose.p) < [0.02, 0.02, 0.02]) \
            and np.dot(self.place_pose.to_transformation_matrix()[:3, 2], [0, 0, 1]) > 0.99
