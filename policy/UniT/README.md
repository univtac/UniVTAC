# UniT: Data Efficient Tactile Representation with Generalization to Unseen Objects

_Accepted by IEEE RA-L_

[Project](https://zhengtongxu.github.io/unit-website/) | [arXiv](https://arxiv.org/abs/2408.06481) | [Summary Video](https://drive.google.com/file/d/1RrW7xk7SjMaIHqksxg0vrhm7SPVrPtg8/view?usp=sharing)

![ChickenLegsHanging](teasers/teaser_chicken.gif)


UniT is an approach to tactile representation learning, using VQGAN to learn a compact latent space and serve as the tactile representation. It uses tactile images obtained from a single simple object to train the representation with transferability and generalizability. This tactile representation can be zero-shot transferred to various downstream tasks, including perception tasks and manipulation policy learning. Our benchmarking on an in-hand 3D pose estimation task shows that UniT outperforms existing visual and tactile representation learning methods. Additionally, UniT's effectiveness in policy learning is demonstrated across three real-world tasks involving diverse manipulated objects and complex robot-object-environment interactions. Through extensive experimentation, UniT is shown to be a simple-to-train, plug-and-play, yet widely effective method for tactile representation learning.

## Dataset

We have released the dataset for representation learning, in-hand 3D pose estimation, and manipulation policy learning at this [link](https://drive.google.com/drive/folders/1CkPqgNFCE6B1mr2pxYdNdSR-xAkSnxQc?usp=sharing).

The "Chicken Legs Hanging," "Chips Grasping," and "Allen Key Insertion" datasets contain 200, 180, and 170 demonstrations, respectively. Specifically, the "Allen Key Insertion" dataset includes 100 demonstrations collected with one type of rack and 70 with another type.

To unzip the human demonstration datasets, you can run the following command as an example
```console
$ sudo apt-get install p7zip-full
$ 7z x chips_grasping.zip
```

## Installation

For installation, please run

```console
$ cd UniT
$ mamba env create -f conda_environment.yaml && bash install_custom_packages.sh
```

Please note that in the `install_custom_packages.sh` script, the following command is executed
```console
$ source ~/miniforge3/etc/profile.d/conda.sh
```

This command is generally correct. However, if your Conda environments are not located in the `~/miniforge3` directory, please adjust the command to match the path of your environment.

## Representation Training

Activate conda environment
```console
$ conda activate unitenv
```

For example, launch representation training with UniT and Allen key dataset
```console
$ python train.py --config-dir=./UniT/config --config-name=vqvae_representation_key.yaml
```
The pretrained models will be saved in /representation_models

Similarly, launch representation training with MAE and Allen key dataset
```console
$ python train.py --config-dir=./UniT/config --config-name=mae_representation_key.yaml
```

We provide multiple YAML files as examples for representation learning, covering all the methods benchmarked in our paper. Feel free to explore and experiment with them!

For a quick implementation, we offer the UniT representation model, pretrained using the Allen key data, in this [folder](https://drive.google.com/drive/u/0/folders/1h7LEpv7DFSN3-MpCDPeyHbNRfPuVBqTL).

Please note that the pretrained representation is based on the GelSight Mini with markers. If you're using a GelSight Mini without markers or another version of GelSight, consider training the representation from scratch with your own data. Additionally, we cropped the raw images from the GelSight Mini with markers, removing the edges where no markers were present. Specifically, we use [gsmini](https://github.com/raghavauppuluri13/gsmini) and [this recorder](https://github.com/ZhengtongXu/UniT/blob/main/UniT/utils/tactile_recorder.py) for data streaming and cropping. Please compare your raw images with those displayed on our [website](https://zhengtongxu.github.io/unifiedtactile.github.io/)  to check whether you are using a significantly different cropping method. If so, consider cropping your data similarly or training the representation from scratch using your data.


## Tactile Perception Training
All logs from tactile perception training will be uploaded to wandb. Login to [wandb](https://wandb.ai) (if you haven't already)
```console
$ wandb login
```
For example, deploy the trained UniT representation to the in-hand 3D pose estimation task, just run
```console
$ python train.py --config-dir=./UniT/config --config-name=key_vqvae_perception.yaml hydra.run.dir=data/outputs/your_folder_name
```

For in-hand 6D pose estimation task, run
```console
$ python train.py --config-dir=./UniT/config --config-name=key_vqvae_perception_6D.yaml hydra.run.dir=data/outputs/your_folder_name
```

Note that each job need to create a new folder to save the outputs. You don't need to create the folder manually, the system will automatically create a new folder for you when you run the job with a different `hydra.run.dir`.

Please ensure that the path to the pretrained VQVAE is specified in the YAML configuration file. You may use our provided pretrained representation model available [here](https://drive.google.com/drive/u/0/folders/1h7LEpv7DFSN3-MpCDPeyHbNRfPuVBqTL), or alternatively, you can train your own representation models using either our released data or your own dataset.

Similary, you can run other config files to launch in-hand 3D pose and 6D pose estimation training with all different methods.

For classification on [YCB-Sight](https://github.com/Robo-Touch/YCB-Sight), train representations on YCB-Sight first. For example, to train UniT representation with `master_chef_can`, run
```console
$ python train.py --config-dir=./UniT/config --config-name=vqvae_representation_ycb_0.yaml
```

Then, run the following command to launch the classification task
```console
$ python train.py --config-dir=./UniT/config --config-name=vqvae_perception_clf.yaml hydra.run.dir=data/outputs/your_folder_name
```
Make sure to specify the correct path to the pretrained representation model in the YAML configuration file.

We provide the replay buffer for YCB-Sight in this [folder](https://drive.google.com/drive/u/0/folders/1h7LEpv7DFSN3-MpCDPeyHbNRfPuVBqTL). We use [this script](https://github.com/ZhengtongXu/UniT/blob/main/UniT/ycb_sight_to_replaybuffer.py) to convert the YCB-Sight data to the replay buffer format, which is more compatible with our codebase.

## Policy Training
For example, to train the diffusion policy with UniT for the chicken legs hanging task, run
```console
$ python train.py --config-dir=./UniT/config --config-name=train_legs_unit_policy.yaml hydra.run.dir=data/outputs/your_folder_name
```

## Human Demonstration Data Visualization
We have set up a visualization script using [rerun.io](https://rerun.io/).

You can drag the progress bar to view visual observations, proprioception data, and action data for a single episode or multiple specified episodes within the dataset.

![RerunExample](teasers/rerun_example.gif)

Run the visualization for any episode index in the chicken hanging task

```console
# visualize last episode
$ EPISODES=-1 python data_vis.py --config-name data_vis_chickenlegs_hanging.yaml
# visualize first 3 episodes
$ EPISODES="0,1,2" python data_vis.py --config-name data_vis_chickenlegs_hanging.yaml
```

Results will be saved to `data/<dataset_name>/debug.rrd` folder
Open the `debug.rrd` file with rerun (on linux machine)

```console
$ rerun
```
Then
```
File > Open RRD file
```

## BibTex

If you find this codebase useful, consider citing:

```bibtex
@misc{xu2025unit,
      title={{UniT}: Data Efficient Tactile Representation with Generalization to Unseen Objects}, 
      author={Zhengtong Xu and Raghava Uppuluri and Xinwei Zhang and Cael Fitch and Philip Glen Crandall and Wan Shou and Dongyi Wang and Yu She},
      year={2025},
      eprint={2408.06481},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2408.06481}, 
}
```


## Acknowledgement

This repository, during construction, referenced the code structure of [diffusion policy](https://github.com/real-stanford/diffusion_policy). We sincerely thank the authors of diffusion policy for open-sourcing such an elegant codebase!

In this repository, [taming](UniT/taming) is adapted from [taming-transformers](https://github.com/CompVis/taming-transformers) and [policy](UniT/policy) is adapted from [UMI](https://github.com/real-stanford/universal_manipulation_interface).
