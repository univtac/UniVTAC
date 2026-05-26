from collections.abc import Sequence
from absl import app
import numpy as np
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer 
import os
from pathlib import Path
import cv2

dataset_dir = 'data/ycb_sight'

def batch_process(batch):
    processed_data = []
    for img in batch:
        processed_img = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
        processed_data.append(processed_img)
    return np.array(processed_data)

def load_data(dataset_dir):
    data_by_episode = {}
    labels_by_episode = {}
    episode_counter = 0
    
    # Get all object folders (e.g., 004_sugar_box, 005_tomato_soup_can)
    data_path = Path(dataset_dir)
    object_folders = sorted(data_path.glob('[0-9][0-9][0-9]*'))
    
    # Create a mapping from original class IDs to continuous IDs (0-5)
    unique_class_ids = sorted([int(folder.name[:3]) for folder in object_folders])
    class_id_mapping = {orig: new for new, orig in enumerate(unique_class_ids)}
    print(class_id_mapping)
    
    for obj_folder in object_folders:
        original_class_id = int(obj_folder.name[:3])
        continuous_class_id = class_id_mapping[original_class_id]
        
        # Simplified path to gelsight images
        gelsight_path = obj_folder / 'gelsight'
        
        # Get all odd-numbered gelsight images
        image_files = sorted(gelsight_path.glob('gelsight_*_*'))
        odd_images = [img for img in image_files if int(img.stem.split('_')[1]) % 2 == 1]
        
        if odd_images:
            images = batch_process(odd_images)
            data_by_episode[episode_counter] = images
            labels_by_episode[episode_counter] = np.full(len(images), continuous_class_id)
            episode_counter += 1
    
    return data_by_episode, labels_by_episode

def add_episode_to_buffer(buffer, episode_data):
    episode_length = len(episode_data['image'])
    if episode_length == 0:
        return  # No data to add

    episode_data = {key: np.array(value) for key, value in episode_data.items()}
    buffer.add_episode(episode_data, compressors="disk")

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    output_dir = "data/debug"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)
    zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")
    
    image_data, label_data = load_data(dataset_dir)
    episode_ids = set(image_data)

    for episode_id in episode_ids:
        print(f"Processing episode {episode_id}, shape: {image_data[episode_id].shape}")
        episode_data = {
            'image': image_data[episode_id],
            'label': label_data[episode_id]
        }
        add_episode_to_buffer(replay_buffer, episode_data)

if __name__ == '__main__':
    app.run(main)