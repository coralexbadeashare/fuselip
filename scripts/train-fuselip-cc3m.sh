#!/bin/bash

ulimit -n 250000

export PYTHONPATH="$PYTHONPATH:./src"


if [ "$1" == "small" ]; then
  tokenizer_path="yucornetto/tokenizer_titok_s128_imagenet"
  transformer_size="small"
fi
if [ "$1" == "base" ]; then
  tokenizer_path="yucornetto/tokenizer_titok_bl128_vq8k_imagenet"
  transformer_size="base"
fi
if [ -z "$tokenizer_path" ] || [ -z "$transformer_size" ]; then
  echo "Usage: $0 <small|base>"
  exit 1
fi

echo "[INFO] Training FuseLIP with the following settings:"
echo "  tokenizer_path: $tokenizer_path"
echo "  transformer_size: $transformer_size"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  torchrun args: --rdzv_backend=c10d --rdzv_endpoint=localhost:29506 --nproc_per_node 1 -m open_clip_train.main ..."
echo "[INFO] Starting torchrun..."

torchrun --standalone --nproc_per_node 1 -m open_clip_train.main \
--log-every-n-steps 1 \
--save-frequency 10 --save-most-recent --zeroshot-frequency 1 --report-to wandb \
--train-data="cc3m" \
--val-data "cc3m" \
--dataset-type merged --combined-sampling  \
--warmup 12000 --batch-size=256 --lr=1e-3 --wd=1.0 \
--epochs=8 --workers=16 --model fuse-clip-titok \
--image-tokenizer "$tokenizer_path" --transformer-size "$transformer_size" \
--context-len 180 --mask-pad --siglip \
--grad-clip-norm 1.0 \
--mlm-loss --mlm-probability .1 --mlm-loss-weight .25