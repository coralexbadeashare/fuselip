#!/bin/bash

# Set environment
ulimit -n 250000
export PYTHONPATH="$PYTHONPATH:./src"

# Check if the model size argument is provided
if [ "$1" == "small" ]; then
    tokenizer_path="yucornetto/tokenizer_titok_s128_imagenet"
    transformer_size="small"
elif [ "$1" == "base" ]; then
    tokenizer_path="yucornetto/tokenizer_titok_bl128_vq8k_imagenet"
    transformer_size="base"
else
    echo "Usage: $0 <small|base>"
    exit 1
fi

# Create output directory
OUTPUT_DIR="/workspace/models/fuselip_roccov2_${transformer_size}"
mkdir -p ${OUTPUT_DIR}

# Run the fine-tuning
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29508 --nproc_per_node=8 -m open_clip_train.main \
    --log-every-n-steps 1 \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --train-data="/workspace/ROCOv2_data/train" \
    --val-data="/workspace/ROCOv2_data/validation" \
    --dataset-type="finetuning" \
    --csv-img-key="image_path" \
    --csv-caption-key="caption" \
    --pretrained="chs20/FuseLIP-S-CC3M-MM" \
    --context-len=77 \
    --model="fuselip_${transformer_size}" \
    --mask-pad \
    --combined-sampling \