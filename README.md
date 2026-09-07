<h1 align="center">UniVTAC</h1>

> UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking<br>
> [arXiv](https://arxiv.org/abs/2602.10093) | [PDF](https://arxiv.org/pdf/2602.10093) | [Website](https://univtac.github.io/) | [HuggingFace Dataset](https://huggingface.co/datasets/byml/UniVTAC) | [Modelscope Dataset](https://modelscope.cn/datasets/byml2024/UniVTAC)

**UniVTAC** is a tactile-aware simulation benchmark for robotic manipulation built on top of **NVIDIA Isaac Lab** and **TacEx (UIPC-based tactile simulation)**. It provides a unified framework for collecting expert demonstrations, training visuotactile policies, and evaluating them across a diverse suite of contact-rich manipulation tasks — all with high-fidelity tactile feedback from simulated GelSight Mini, ViTai GF225, or XenseWS sensors.

## News

- **2026-09** — UniVTAC was accepted to **CoRL 2026**.
- **2026-09** — Uploaded and updated the Isaac Sim 5.1 task dataset.

> [!IMPORTANT]
> The `main` branch targets Isaac Sim 4.5, while the `isaac51` branch
> supports Isaac Sim 5.1. The Isaac Sim 5.1 release delivers up to **5x
> higher data-collection throughput** than the Isaac Sim 4.5 version and
> supports NVIDIA RTX 40- and 50-series GPUs. Data collected with the two
> versions is not cross-compatible. Both Isaac Sim 4.5 and 5.1 task datasets
> are available in the dataset repository; select the version that matches the
> code branch and simulator environment you use.

## Installation

The `main` branch targets Isaac Sim 4.5.0 and Isaac Lab 2.1.1. Run `scripts/install.sh` to set up the `UniVTAC` Conda environment with Python 3.10 and install Isaac Sim, Isaac Lab, TacEx, and cuRobo.

```bash
git clone https://github.com/univtac/UniVTAC.git
cd UniVTAC
bash scripts/install.sh
```

See the [Installation Guide](./docs/Installation.md) for detailed setup instructions, including installing the environment, installing TacEx from the modified local source and setting up cuRobo for motion planning.

## Task Gallery

UniVTAC currently includes the following manipulation tasks, all featuring tactile sensing:

| Task | Module | Description |
|---|---|---|
| **Collect** | `collect` | Collect contact-rich tactile data for pretraining |
| **Lift Bottle** | `lift_bottle` | Grasp and lift a bottle off a surface near a wall |
| **Lift Can** | `lift_can` | Grasp and lift a cylindrical can |
| **Insert HDMI** | `insert_HDMI` | Insert an HDMI connector into a port |
| **Insert Hole** | `insert_hole` | Precision peg-in-hole insertion |
| **Insert Tube** | `insert_tube` | Insert a tube into a fixture |
| **Pull Out Key** | `pull_out_key` | Extract a key from a lock |
| **Put Bottle in Shelf** | `put_bottle_in_shelf` | Place a bottle onto a shelf |
| **Grasp & Classify** | `grasp_classify` | Grasp an object and classify it by tactile feedback |

To build more tasks, refer to the [Task Creation Guide](./docs/TaskCreation.md) for instructions on how to define new manipulation tasks within the UniVTAC framework.

## Data Collection

See the [Data Collection Guide](./docs/Collection.md) for instructions on how to run the automated data collection pipeline, configure task-specific parameters, and understand the output data structure.

The dataset is available from [HuggingFace](https://huggingface.co/datasets/byml/UniVTAC)
and [ModelScope](https://modelscope.cn/datasets/byml2024/UniVTAC). Its published
layout is:

| Path | Contents |
|---|---|
| `isaac45/<task>/` | Isaac Sim 4.5 demonstrations: 100 HDF5 episodes and one `metadata.json` for each of the 8 tasks |
| `isaac51/<task>/` | Isaac Sim 5.1 demonstrations: 100 HDF5 episodes and one `metadata.json` for each of the 8 tasks |
| `contact/<shape>/` | Contact-pretraining trajectories for 14 non-empty shapes (638 HDF5 episodes in the current release) |
| `checkpoints/` | Policy checkpoints, metadata, dataset statistics, logs, and the shared tactile encoder; the current checkpoint release is trained for the Isaac Sim 4.5 dataset only |

Task trajectories are stored under `hdf5/*.hdf5`. Because simulator versions
are not interchangeable, downloading task data requires an explicit version.
Use `--version 45` with this `main` branch; use `--version 51` with the
`isaac51` branch:

```bash
# All Isaac Sim 4.5 task demonstrations for main
bash data/download.sh --task --version 45

# One Isaac Sim 4.5 task
bash data/download.sh --task lift_can --version 45

# Other independently selectable dataset components
bash data/download.sh --contact
bash data/download.sh --checkpoint

# Select multiple components in one invocation
bash data/download.sh --task --version 45 --contact --checkpoint
```

Selectors may be narrowed and repeated, for example
`--contact Cross --contact Sphere` or
`--checkpoint lift_can --checkpoint insert_hole`. Files are placed under
`data/` by default while preserving the paths shown above. Run
`bash data/download.sh --help` for output-directory, revision,
parallelism, and force-download options.

> [!NOTE]
> The currently released policy checkpoints are associated with the Isaac Sim
> 4.5 demonstrations. They should not be treated as Isaac Sim 5.1 checkpoints.

## Data Export

Raw UniVTAC episodes can be exported to the [TLabel](https://github.com/liesliy/tlabel) unified tactile annotation format (a COCO-style common schema for robot tactile data) for cross-dataset labelling and benchmarking:

```bash
pip install tlabel
python -m policy.tlabel_data_preprocessor <task_name> <task_config> <episode_num> --output ./data/tlabel/<task_name>-<task_config>
```

See the [TLabel Export Guide](./docs/TLabelExport.md) for the supported fields, output schema, CLI options and the new/old HDF5 format compatibility notes.

## Train & Eval Policies

UniVTAC includes several baseline policies implemented under the `policy/` directory:

- ACT: Action Chunking with Transformers with/without tactile inputs
- Abation: ACT ablation variants for modality comparison
- ViTAL: ACT with CLIP-pretrained tactile-vision encoders in ViTAL

Each policy is a self-contained module under `policy/` with its own data processing, training, and deployment scripts. All policies share a unified evaluation entry point at the project root:

```bash
bash eval_policy.sh ${task_name} ${task_config} ${policy_config} ${gpu_id}
```

For parallel evaluation over many seeds:

```bash
bash parallel_eval.sh ${task_name} ${task_config} ${policy_config} ${gpu_id} [num_processes] [total_num]
```

The evaluation results, including videos and success rate logs, will be saved in the `eval_result/` directory under the project root.

To deploy your own policy, refer to the [Deploy Your Policy](./docs/Deploy.md).

## TODO

- Data collection and evaluation are now only supported on the GelSight Mini sensor. We will add support for ViTai GF225 and XenseWS in the near future.

## 👍 Citations
If you find our work useful, please consider citing:

```
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

## Contact
<div style="text-align: center;">
  <img src="https://box.nju.edu.cn/seafhttp/f/fc1021a908ff49309f22/?op=view" alt="Wechat Group" width="300"/>
</div>
