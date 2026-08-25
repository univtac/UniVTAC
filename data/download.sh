#!/bin/bash

HF_CACHE_DIR=/media/maki/New/hf_cache
mkdir -p "$HF_CACHE_DIR"

export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export MODELSCOPE_CACHE="$HF_CACHE_DIR"

# Download the dataset using modelscope
if ! command -v modelscope &> /dev/null
then
    pip install modelscope
fi

modelscope download --dataset byml2024/UniVTAC --cache_dir "$HF_CACHE_DIR"
