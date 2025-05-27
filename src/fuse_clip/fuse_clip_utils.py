import argparse
import json
import logging
import os
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from huggingface_hub import repo_exists

from config import LOG_PATH
from fuse_clip.fuse_clip_arch import FuseCLIP
from fuse_clip.fuse_clip_hub import FuseLIP
from fuse_clip.fuse_clip_preprocess import get_fuse_clip_image_preprocess
from fuse_clip.feature_fuse_clip import FeatureFuseCLIP
from open_clip import create_model_and_transforms, get_tokenizer, get_input_dtype, SimpleTokenizer

from vlm2vec_new.src.collator import EvalCollator
from vlm2vec_new.src.model import MMEBModel
from vlm2vec_new.src.arguments import ModelArguments, DataArguments


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError

def is_fused(args: argparse.Namespace) -> bool:
    return 'fuse' in args.model

def top10_parameters_by_size(model):
    params = [(name, p.numel()) for name, p in model.named_parameters()]
    params = sorted(params, key=lambda x: x[1], reverse=True)[:10]
    return params

def get_cos_sims(tens: torch.Tensor) -> float:
    tens = tens / tens.norm(dim=-1, keepdim=True)
    cos_sims = tens @ tens.t()
    # diagonal is 1, disregard it
    mask = ~torch.eye(tens.shape[0], dtype=bool, device=tens.device)
    cos_sim = cos_sims[mask].mean().item()
    return cos_sim

def load_fuse_clip_sd(state_dict_path: str, device: str = "cuda") -> dict:
    sd = torch.load(state_dict_path, map_location=device, weights_only=False)["state_dict"]
    if "module." in list(sd.keys())[0]:
        sd = {k[len("module."):]: v for k, v in sd.items()}
    # remove unused keys
    for k in list(sd.keys()):
        if "image_tokenizer.model.pixel_" in k or "image_tokenizer.model.decoder" in k:
            del sd[k]
    return sd

def load_model(model_id: str, device: str, ckpt_name: str = "epoch_final.pt"):
    model_config_file = os.path.join(LOG_PATH, model_id, "fuse-clip-config.json")

    if "vlm2vec" in model_id:
        model, (processor, collator), _ =  load_vlm2vec(model_id=model_id, device=device)
        return model, (processor, collator), None
    elif model_id.startswith("hf-hub:"):
        logging.info("loading pretrained baseline")
        model, preprocess_val, tokenizer = load_pretrained_baseline(model_id, device=device)
        return model, preprocess_val, tokenizer
    elif os.path.exists(model_config_file) or model_id.startswith("chs20/"):
        return load_fuse_clip(model_id=model_id, device=device, ckpt_name=ckpt_name)
    else:
        logging.info("loading feature fused clip")
        model, preprocess_val, tokenizer = load_our_baseline(model_id, device=device)
        return model, preprocess_val, tokenizer

def load_fuse_clip(model_id: str, device: str, ckpt_name: str = "epoch_final.pt") -> Tuple[FuseCLIP, callable, callable]:
    # if repo_exists(model_id):
    if model_id.startswith("chs20/"):
        logging.info(f"Loading FuseLIP from Hugging Face Hub: {model_id}")
        model = FuseLIP.from_pretrained(model_id, device=device)
        image_processor_val = get_fuse_clip_image_preprocess(train=False)
        tokenizer = model.text_tokenizer
        return model, image_processor_val, tokenizer

    pretrained = os.path.join(LOG_PATH, model_id, "checkpoints", ckpt_name)
    model_config_file = os.path.join(LOG_PATH, model_id, "fuse-clip-config.json")

    with open(model_config_file, "r") as f:
        model_config = json.load(f)

    use_eoi = model_config["use_eoi"]
    mask_pad = model_config["mask_pad"]
    image_tokenizer = model_config["image_tokenizer"]
    context_len = model_config["context_length"]
    transformer_size = model_config.get("transformer_size", "small")
    mlm_prob = model_config.get("mlm_probability", 0.0)

    image_processor_val = get_fuse_clip_image_preprocess(train=False)

    tokenizer = get_tokenizer(
        "fuse-clip-titok", context_length=context_len, is_mlm=mlm_prob>0)
    model = FuseCLIP(
        model_name="fuse-clip-titok", text_tokenizer=tokenizer, image_tokenizer_path=image_tokenizer,
        transformer_size=transformer_size,
        device=device, input_dtype=get_input_dtype("fp32"),
        init_logit_scale=np.log(10), init_logit_bias=-10,
        use_eoi=use_eoi, mask_pad=mask_pad
    )
    if pretrained is not None:
        sd = load_fuse_clip_sd(state_dict_path=pretrained, device=device)
        if "mask_prediction_head.bias" in sd and not "mask_prediction_head.bias" in model.state_dict():
            # load without mask prediction head
            del sd["mask_prediction_head.bias"]
            del sd["mask_prediction_head.dense.weight"]
            del sd["mask_prediction_head.dense.bias"]
            del sd["mask_prediction_head.layer_norm.weight"]
            del sd["mask_prediction_head.layer_norm.bias"]
            del sd["mask_prediction_head.decoder.weight"]
        model.load_state_dict(sd)

    model.eval()
    return model, image_processor_val, tokenizer

def load_pretrained_baseline(model_id: str, device: str = "cuda"):
    tokenizer = get_tokenizer(model_id)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name=model_id, pretrained=None,
        image_resize_mode="squash",
        device=device
    )
    model = FeatureFuseCLIP(
        model, fusion_module_name="add", text_tokenizer=tokenizer, device="cuda"
    )
    logging.info("setting append_empty_text to False")
    model.append_empty_text = False

    return model, preprocess_val, tokenizer

def load_our_baseline(model_id: str, device: str = "cuda"):
    sd_path = os.path.join(LOG_PATH, model_id, "checkpoints", "epoch_final.pt")
    sd = load_fuse_clip_sd(sd_path, device=device)
    params_file = os.path.join(LOG_PATH, model_id, "params.txt")
    with open(params_file, "r") as f:
        params = f.readlines()
    params = {p.split(":")[0]: p.split(":")[1].strip() for p in params}
    model_name = params["model"]
    fusion_module_name = params["feature_fusion"]
    if fusion_module_name in [True, "True", False, "False", None, "None", "none"]:
        fusion_module_name = "add"

    tokenizer = get_tokenizer(model_name)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name=model_name, pretrained=None, image_resize_mode="squash",
        device=device
    )
    model = FeatureFuseCLIP(
        model, fusion_module_name=fusion_module_name, text_tokenizer=tokenizer, device="cuda"
    )
    if not list(sd.keys())[0].startswith("model."):
        sd = {f"model.{k}": v for k, v in sd.items()}
    model.load_state_dict(sd)
    if params["feature_fusion"] in [False, "False", None, "None", "none"]:
        # set this only for non-fused baseline, as it has not seen fusion during training
        logging.info("setting append_empty_text to False")
        model.append_empty_text = False
    model = model.eval()

    return model, preprocess_val, tokenizer

def load_vlm2vec(model_id: str, device: str = "cuda"):
    from transformers import AutoProcessor

    model_args = ModelArguments(
        model_name='TIGER-Lab/VLM2Vec-Full',
        pooling='last',
        normalize=True,
        model_backbone='phi3_v',
        num_crops=4,
    )
    data_args = DataArguments(
        max_len=256
    )

    # processor = load_processor(model_args)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to('cuda', dtype=torch.bfloat16)

    # collator = EvalCollator(data_args=data_args, model_args=model_args, processor=processor)
    collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )
    return model, (processor, collator), None


def ensure_list(inp) -> List:
    return [inp] if not isinstance(inp, list) else inp


def fused_train_collator(batch: List[Tuple]) -> Dict:
    # output list of samples
    texts_left = []
    images_left = []
    texts_right = []
    images_right = []
    # flatten batch
    for qry_txt, qry_img, pos_txt, pos_img in batch:
        texts_left.extend(ensure_list(qry_txt))
        images_left.extend(ensure_list(qry_img))
        texts_right.extend(ensure_list(pos_txt))
        images_right.extend(ensure_list(pos_img))
    texts_left = torch.stack(texts_left)
    texts_right = torch.stack(texts_right)
    out_dict = {
        "images_left": images_left,
        "texts_left": texts_left,
        "images_right": images_right,
        "texts_right": texts_right,
    }
    return out_dict

def disentangle_batch(batch, device):
    # print(len(batch))
    # for item in batch:
    #     print(type(item))
    #try:
    texts_left, images_left, texts_right, images_right = (
        batch["texts_left"], batch["images_left"], batch["texts_right"], batch["images_right"]
    )
    # except TypeError:
    #     texts_left, images_left, texts_right, images_right = batch
    if isinstance(texts_left, list):
        texts_left = torch.stack(texts_left)
        texts_right = torch.stack(texts_right)
    texts_left = texts_left.to(device=device, non_blocking=True)
    texts_right = texts_right.to(device=device, non_blocking=True)
    images = (images_left, images_right)
    texts = (texts_left, texts_right)
    return images, texts


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def compute_cos_sim_matrix(tensor_a, tensor_b):
    tensor_a = tensor_a / tensor_a.norm(dim=-1, keepdim=True)
    tensor_b = tensor_b / tensor_b.norm(dim=-1, keepdim=True)
    cos_sims = tensor_a @ tensor_b.t()
    return cos_sims


def decode_tokens(tokenizer, tokens):
    tokens = tokens.squeeze()

    if isinstance(tokenizer, SimpleTokenizer):
        tokens = tokens[1:-1]
    tokens = tokens[tokens != tokenizer.pad_id].tolist()

    if not isinstance(tokens, list):  # can happen if only one token
        tokens = [tokens]

    return tokenizer.decode(tokens)


def compute_iou(bboxA, bboxB):
    """Compute IoU between two bounding boxes in [x1, x2, y1, y2] format."""
    if isinstance(bboxA, dict) and "height" in bboxA:
        bboxA = [bboxA["x"], bboxA["x"] + bboxA["width"], bboxA["y"], bboxA["y"] + bboxA["height"]]
        bboxB = [bboxB["x"], bboxB["x"] + bboxB["width"], bboxB["y"], bboxB["y"] + bboxB["height"]]

    x1A, x2A, y1A, y2A = bboxA
    x1B, x2B, y1B, y2B = bboxB

    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)

    union_area = areaA + areaB - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area



if __name__ == '__main__':
    model_id = "2024_12_11-08_19_56-model_ViT-S-32-256-180-lr_0.001-b_512-j_32-p_amp"

    model = load_our_baseline(model_id)
    print(top10_parameters_by_size(model))
