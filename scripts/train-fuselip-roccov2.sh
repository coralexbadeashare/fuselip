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

# First, download ROCCO v2 dataset if not already downloaded
python scripts/download_rocov2.py

# Run the fine-tuning
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29508 --nproc_per_node=8 -m open_clip_train.main \
    --log-every-n-steps 1 \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --train-data="roccov2" \
    --val-data="roccov2" \
    --dataset-type=roccov2 \
    --warmup 1000 \
    --batch-size=128 \
    --lr=5e-5 \
    --wd=0.1 \
    --epochs=10 \
    --workers=8 \
    --model fuse-clip-titok \
    --pretrained "chs20/FuseLIP-S-CC3M-MM" \
    --image-tokenizer "${tokenizer_path}" \
    --transformer-size "${transformer_size}" \
    --context-len 180 \
    --mask-pad \
    --siglip \
    --grad-clip-norm 1.0 \
    --mlm-loss \
    --mlm-probability 0.1 \
    --mlm-loss-weight 0.25