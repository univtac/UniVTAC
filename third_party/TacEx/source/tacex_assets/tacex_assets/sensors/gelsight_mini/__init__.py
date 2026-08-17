"""Ready to use configurations for the GelSight Mini.

The configurations differ mainly in the employed simulation approach and their combination.
"""

from .generic_gsmini_cfg import GeneralGelSightMiniCfg

from .gsmini_taxim import GELSIGHT_MINI_TAXIM_CFG
from .gsmini_taxim_fots import GELSIGHT_MINI_TAXIM_FOTS_CFG
from .gsmini_taxim_fem import GELSIGHT_MINI_TAXIM_FEM_CFG
from .gsmini_pix2pix import GELSIGHT_MINI_PIX2PIX_CFG