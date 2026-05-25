from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import av
import zarr
import numcodecs
import multiprocessing
from diffusion_policy.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k

register_codecs()
def real_data_to_replay_buffer(
    dataset_path: str,
) -> ReplayBuffer:

    # verify input
    input = pathlib.Path(os.path.expanduser(dataset_path))
    in_zarr_path = input.joinpath("replay_buffer.zarr")
    assert in_zarr_path.is_dir()

    in_replay_buffer = ReplayBuffer.create_from_path(
        str(in_zarr_path.absolute()), mode="r"
    )


    return in_replay_buffer
