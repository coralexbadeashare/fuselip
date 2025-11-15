#!/bin/bash

# Set environment
ulimit -n 250000
export PYTHONPATH="$PYTHONPATH:./src"

PRETRAINED="/workspace/fuselip/logs/2025_11_13-16_35_15-model_fuse-clip-titok-lr_0.001-b_256-j_16-p_amp/checkpoints/epoch_final.pt"

# Default context length (can be overridden by env var CONTEXT_LEN)
# We'll attempt to auto-detect the required context length from the checkpoint
# using a small helper script. If detection fails we fall back to 77.
CONTEXT_LEN=${CONTEXT_LEN:-}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$CONTEXT_LEN" ]; then
    # user override via env var
    :
else
    CONTEXT_LEN=77
    if [ -f "$PRETRAINED" ]; then
        detected=$(python3 "$SCRIPT_DIR/get_checkpoint_context_len.py" "$PRETRAINED" 2>/dev/null || true)
        if [[ "$detected" =~ ^[0-9]+$ ]]; then
            CONTEXT_LEN=$detected
        fi
    fi
fi

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
echo "  pretrained: $PRETRAINED"
echo "  context_len: $CONTEXT_LEN"

# Create output directory
OUTPUT_DIR="/workspace/models/fuselip_roccov2_${transformer_size}"
mkdir -p ${OUTPUT_DIR}

# Run the fine-tuning
# Run the fine-tuning
torchrun --standalone --nproc_per_node 1 -m open_clip_train.main \
--log-every-n-steps 1 \
--save-frequency 1 --save-most-recent --zeroshot-frequency 1 --report-to wandb \
--train-data="/workspace/fuselip/ROCOv2_data/train_roco.csv" \
--val-data="/workspace/fuselip/ROCOv2_data/test_roco.csv" \
--dataset-type csv \
--pretrained "$PRETRAINED" \
--warmup 12000 --batch-size=256 --lr=1e-3 --wd=1.0 \
--workers=${WORKERS:-16} --model fuse-clip-titok \
--image-tokenizer "$tokenizer_path" --transformer-size "$transformer_size" \
--context-len "$CONTEXT_LEN" --mask-pad --siglip \
--grad-clip-norm 1.0 \
--mlm-loss --mlm-probability .1 --mlm-loss-weight .25
    