#!/bin/bash

ulimit -n 250000

export PYTHONPATH="$PYTHONPATH:./src"

if [ "$1" == "sf" ]; then
  fusion="add"
fi
if [ "$1" == "mlf" ]; then
  fusion="magiclens"
fi

if [ "$2" == "small" ]; then
  model="ViT-S-16-256-180-mask"
fi
if [ "$2" == "base" ]; then
  model="ViT-B-16-256-180-mask"
fi

if [ -z "$fusion" ] || [ -z "$model" ]; then
  echo "Usage: $0 <sf|mlf> <small|base>"
  exit 1
fi


torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29506 --nproc_per_node 8 -m open_clip_train.main -m open_clip_train.main \
--log-every-n-steps 1 \
--save-frequency 10 --save-most-recent --zeroshot-frequency 1 --report-to wandb \
--train-data="cc3m-aug,hq-edit,cc3m-vqa,vg-crop,vg-vqa" --val-data="cc3m,cc3m-aug,hq-edit,cc3m-vqa,vg-crop,vg-vqa" \
--dataset-type=merged --combined-sampling \
--warmup 12000 --batch-size=256 --lr=1e-3 --wd=1.0 \
--epochs=8 --workers=${WORKERS:-16} --model "$model" \
--grad-clip-norm 1.0

# if it does not fit on GPU, use --accum-freq 2 --accum-negatives