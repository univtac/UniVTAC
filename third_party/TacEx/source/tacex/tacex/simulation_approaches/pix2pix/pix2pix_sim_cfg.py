from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .pix2pix_sim import Pix2PixSimulator


@configclass
class Pix2PixSimulatorCfg(GelSightSimulatorCfg):
    """Config for pix2pix sim approach for TactileRGB images."""

    simulation_approach_class: type = Pix2PixSimulator

    calib_folder_path: str = ""
    tactile_img_res: tuple = (320, 240)
    """Resolution (width, height) of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is up/down sampled.
    """
