from typing import Optional, List, Callable

import torch
from torch import nn

from open_clip import CLIP, SimpleTokenizer
from open_clip.transformer import Transformer, LayerNorm

FUSION_CONFIG_S = {
    "width": 384, # corresponding to ViT-S-16
    "layers": 4,  # corresponding to MagicLens
    "heads": 6, # corresponding to ViT-S-16
    "mlp_ratio": 4.0, # corresponding to MagicLens
    "num_query_token": 1 # corresponding to MagicLens
}

FUSION_CONFIG_B = {
    "width": 512,
    "layers": 4,
    "heads": 8,
    "mlp_ratio": 4.0,
    "num_query_token": 1
}

def get_fusion_module(fusion_module_name: str, model, device):
    if fusion_module_name.lower() == "add":
        return torch.add
    elif fusion_module_name.lower() == "magiclens":
        model_width = model.text.width
        fusion_config = {384: FUSION_CONFIG_S, 512: FUSION_CONFIG_B}[model_width]
        return FusionModule(fusion_config).to(device)
    else:
        raise ValueError(f"Unknown fusion module: {fusion_module_name}")

class FeatureFuseCLIP(nn.Module):
    def __init__(self, model: CLIP, fusion_module_name: str, text_tokenizer, device):
        super(FeatureFuseCLIP, self).__init__()
        self.model = model
        self.fusion_module = get_fusion_module(fusion_module_name, model, device)
        self.device = device

        self.output_dict = True
        # assert isinstance(text_tokenizer, SimpleTokenizer)
        self.text_tokenizer = text_tokenizer
        self.empty_text = self.text_tokenizer("").to(device=self.device)
        self.pad_id = self.text_tokenizer.pad_id
        self.append_empty_text = True
        if fusion_module_name.lower() == "add":
            self.normalize_before_fusion = True
        elif fusion_module_name.lower() == "magiclens":
            self.normalize_before_fusion = False
        else:
            raise ValueError(f"Unknown fusion module: {fusion_module_name}")

        if isinstance(text_tokenizer, SimpleTokenizer):
            assert self.pad_id == 0
        else:
            assert self.pad_id == 1

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def forward(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None,
                force_fused: bool = False, normalize: bool = True):
        # assert force_fused
        if force_fused:
            return self.encode_multimodal(image, text, normalize=True)

        is_multimodal = isinstance(image, tuple)
        if is_multimodal:
            image_left, image_right = image
            text_left, text_right = text
            left_features = self.encode_multimodal(image_left, text_left, normalize=normalize)
            del image_left, text_left
            right_features = self.encode_multimodal(image_right, text_right, normalize=normalize)
        else:
            left_features = right_features = None
            if (image is not None) and (image.numel() > 0):
                empty_text = self.empty_text.repeat(image.shape[0], 1)
                left_features = self.encode_multimodal(image=image, text=empty_text, normalize=normalize)
            if (text is not None) and (text.numel() > 0):
                right_features = self.encode_multimodal(image=None, text=text, normalize=normalize)


        if self.output_dict:
            out_dict = {
                "image_features": left_features,
                "text_features": right_features,
                "logit_scale": self.model.logit_scale.exp()
            }
            if self.model.logit_bias is not None:
                out_dict["logit_bias"] = self.model.logit_bias
            return out_dict
        else:
            return left_features, right_features, self.logit_scale.exp()

    def encode_input_list(self, input_list: List, normalize: bool, is_image: bool):
        forward_fn = self.model.encode_image if is_image else self.model.encode_text
        # remember positions where image is None
        no_input_pos = [i for i, img in enumerate(input_list) if img is None]
        if len(no_input_pos) == len(input_list):
            return 0.

        # stack all other into tensor
        non_empty_images = [img for img in input_list if img is not None]
        if len(non_empty_images) > 0:
            input_tensor = torch.stack(non_empty_images, dim=0)
            input_tensor = input_tensor.to(device=self.device, non_blocking=True)
            # encode them
            features = forward_fn(input_tensor, normalize=normalize)
        else:
            features = []

        # where image was None, zero-tensor
        zero_tens = torch.zeros_like(features[0])
        features_lst = []
        cur_img_idx = 0
        for i in range(len(input_list)):
            if i in no_input_pos:
                features_lst.append(zero_tens)
            else:
                features_lst.append(features[cur_img_idx])
                cur_img_idx += 1
        return torch.stack(features_lst)

    def encode_multimodal(self, image: torch.Tensor, text: torch.Tensor, normalize: bool = True):
        image_features = text_features = 0.
        # check if whole text batch is empty
        text_is_empty = self.text_tokenizer.is_empty(text)
        # image features
        if isinstance(image, list):
            image_features = self.encode_input_list(image, normalize=self.normalize_before_fusion, is_image=True)
        elif (image is not None) and (image.numel() > 0):
            image_features = self.model.encode_image(image, normalize=self.normalize_before_fusion)
        # text features
        if self.append_empty_text or ((text.numel() > 0) and (not text_is_empty)):
            text_features = self.model.encode_text(text, normalize=self.normalize_before_fusion)
        # combine
        mm_features = self.fusion_module(image_features, text_features)
        if normalize:
            mm_features = torch.nn.functional.normalize(mm_features, dim=-1)
        return mm_features

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return self.forward(image=None, text=text, normalize=normalize)["text_features"]

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return self.forward(image=image, text=None, normalize=normalize)["image_features"]



class FusionModule(nn.Module):
    def __init__(self, config: dict):
        super(FusionModule, self).__init__()
        self.transformer = Transformer(
            width=config["width"],
            layers=config["layers"],
            heads=config["heads"],
            mlp_ratio=config["mlp_ratio"]
        )
        self.attention_pooler = AttentionalPoolerMagicLens(
            d_model=config["width"],
            context_dim=config["width"],
            n_head=config["heads"],
            n_queries=config["num_query_token"],
        )

    def forward(self, img_features: torch.Tensor, txt_features: torch.Tensor):
        if isinstance(img_features, float):
            img_features = torch.zeros_like(txt_features)  # black image
        img_features = img_features.unsqueeze(1)  # [B, 1, D]
        txt_features = txt_features.unsqueeze(1)  # [B, 1, D]
        mm_features = torch.cat([img_features, txt_features], dim=1)  # [B, 2, D]
        mm_features = self.transformer(mm_features)  # [B, 2, D]
        mm_features = self.attention_pooler(mm_features).squeeze(1)  # [B, D]
        return mm_features


class AttentionalPoolerMagicLens(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.empty(n_queries, d_model))
        nn.init.normal_(self.query, mean=0, std=1/d_model**0.5)
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.pool_attn_ln = norm_layer(d_model)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        q = self.query
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        out = self.pool_attn_ln(out)
        return out