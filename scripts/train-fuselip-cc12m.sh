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

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29508 --nproc_per_node=8 -m open_clip_train.main \
  --log-every-n-steps 1 \
  --save-frequency 10 --save-most-recent --zeroshot-frequency 0 --report-to wandb \
  --train-data="cc12m-aug,hq-edit,cc3m-vqa,vg-crop,vg-vqa" \
  --val-data="cc3m,cc3m-aug,hq-edit,cc3m-vqa,vg-crop,vg-vqa" \
  --dataset-type=merged --combined-sampling \
  --warmup 12000 --batch-size=256 --lr=1e-3 --wd=0.5 \
  --epochs=16 --workers=${WORKERS:-16} --model fuse-clip-titok \
  --image-tokenizer "yucornetto/tokenizer_titok_bl128_vq8k_imagenet" --transformer-size base \
  --context-len 180 --mask-pad --siglip \
  --grad-clip-norm 1.0 \
  --mlm-loss --mlm-probability .1 --mlm-loss-weight .25 \
  --val-frequency 0 --zeroshot-frequency 0 --skip-first-val --skip-mmeb-val