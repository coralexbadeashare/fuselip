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

# Print training configuration
echo "[INFO] Training FuseLIP with the following settings:"
echo "  tokenizer_path: $tokenizer_path"
echo "  transformer_size: $transformer_size"
echo "  PYTHONPATH: $PYTHONPATH"

# Create output directory
OUTPUT_DIR="/workspace/models/fuselip_roccov2_${transformer_size}"
mkdir -p ${OUTPUT_DIR}

# Run the fine-tuning
# Run the fine-tuning
torchrun --standalone --nproc_per_node 1 -m open_clip_train.main \
--log-every-n-steps 1 \
--save-frequency 1 --save-most-recent --zeroshot-frequency 1 --report-to wandb \
--train-data="/workspace/ROCOv2_data/train" \
--val-data="/workspace/ROCOv2_data/validation" \
--dataset-type merged --combined-sampling \
--pretrained="chs20/FuseLIP-S-CC3M-MM" \
--warmup 12000 --batch-size=256 --lr=1e-3 --wd=1.0 \
--workers=16 --model fuse-clip-titok \
--image-tokenizer "$tokenizer_path" --transformer-size "$transformer_size" \
--context-len 77 --mask-pad --siglip \
--grad-clip-norm 1.0 \
--mlm-loss --mlm-probability .1 --mlm-loss-weight .25
    