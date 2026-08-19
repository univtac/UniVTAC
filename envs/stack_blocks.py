from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 1000
    # Positive indentation target in millimetres.
    adaptive_grasp_depth_threshold = {'gsmini': 0.2, 'gf225': 1.4, 'xensews': 0.0}

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)
    
    def create_actors(self):
        blue_pose = Pose([0.6, 0.0, 0.005], [1, 0, 0, 0])
        orange_pose = blue_pose.add_bias([-0.08, 0.05, 0.0005])
        green_pose = blue_pose.add_bias([-0.08, -0.05, 0.0005])

        self.blue_block = self._actor_manager.add_from_usd_file(
            name='blue_block',
            asset_path="BlueBlock.usd", 
            pose=blue_pose
        )
        self.orange_block = self._actor_manager.add_from_usd_file(
            name='orange_block',
            asset_path="OrangeBlock.usd", 
            pose=orange_pose
        )
        self.green_block = self._actor_manager.add_from_usd_file(
            name='green_block',
            asset_path="GreenBlock.usd", 
            pose=green_pose
        )
    
    def _reset_actors(self):
        blue_noise = self.create_noise([0.02, 0.05, 0.0], [0, 0, np.pi/3])
        blue_pose = Pose([0.6, 0.0, 0.005], [1, 0, 0, 0]).add_offset(blue_noise)
        
        while True:
            noise1 = self.create_noise([0.15, 0.15, 0.0], [0, 0, np.pi/3])
            noise2 = self.create_noise([0.15, 0.15, 0.0], [0, 0, np.pi/3])
            pose1, pose2 = blue_pose.add_offset(noise1), blue_pose.add_offset(noise2)
            
            if np.linalg.norm(pose1.p - pose2.p) > 0.1 \
                and np.linalg.norm(pose1.p - blue_pose.p) > 0.1 \
                and np.linalg.norm(pose2.p - blue_pose.p) > 0.1:
                break
        
        self.blue_block.set_pose(blue_pose)
        self.orange_block.set_pose(pose1)
        self.green_block.set_pose(pose2)

    def pre_move(self):
        self.delay(10)

    def grasp_and_place(self, grasp_block:Actor, place_block:Actor, z_high:float=0.06):
        grasp_block_pose = grasp_block.get_pose()
        grasp_target = grasp_block_pose.add_bias([0.0, 0.0, 0.03])
        grasp_mat = grasp_target.to_transformation_matrix()
        target_pose = construct_grasp_pose(
            grasp_target.p,
            grasp_mat[:3, 2],
            grasp_mat[:3, 0]
        )
        grasp_idx = grasp_block.register_point(
            pose=target_pose,
            type='contact'
        )
        self.move(self.atom.grasp_actor(
            grasp_block,
            contact_point_id=grasp_idx,
            pre_dis=z_high
        ))
        self.move(self.atom.move_by_displacement(z=z_high))
        
        place_block_pose = place_block.get_pose()
        place_target = place_block_pose.add_bias([0.0, 0.0, 0.049])
        self.move(self.atom.place_actor(
            grasp_block, place_target,
            pre_dis=0.02, dis=0.0
        ))
        self.move(self.atom.move_by_displacement(z=z_high))
 
    def _play_once(self):
        self.grasp_and_place(self.orange_block, self.blue_block, z_high=0.06)
        if self.check_mid_success():
            self.grasp_and_place(self.green_block, self.orange_block, z_high=0.16)
        self.delay(30, is_save=False)
    
    def check_mid_success(self):
        blue_block_pose = self.blue_block.get_pose()
        orange_block_pose = self.orange_block.get_pose()

        rel_pose = orange_block_pose.rebase(blue_block_pose)
        self.metadata['rel_pose1'] = rel_pose.tolist()

        return np.all(np.abs(rel_pose[:2]) < np.array([0.02, 0.02])) \
            and rel_pose[2] > 0.045 and np.dot(
                orange_block_pose.to_transformation_matrix()[:3, 2], [0, 0, 1]
            ) > 0.99

    def check_success(self):
        blue_block_pose = self.blue_block.get_pose()
        orange_block_pose = self.orange_block.get_pose()
        green_block_pose = self.green_block.get_pose()

        rel_pose1 = orange_block_pose.rebase(blue_block_pose)
        rel_pose2 = green_block_pose.rebase(orange_block_pose)
        self.metadata['rel_pose1'] = rel_pose1.tolist()
        self.metadata['rel_pose2'] = rel_pose2.tolist()

        return self.check_mid_success() and \
            np.all(np.abs(rel_pose2[:2]) < np.array([0.02, 0.02])) \
            and rel_pose2[2] > 0.045 and np.dot(
                green_block_pose.to_transformation_matrix()[:3, 2], [0, 0, 1]
            ) > 0.99
