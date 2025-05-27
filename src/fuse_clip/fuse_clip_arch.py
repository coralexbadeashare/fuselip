import logging
from typing import Optional, List, Callable, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from open_clip import SimpleTokenizer
from open_clip.model import _build_text_tower
from titok.modeling.titok import TiTok
from titok.modeling.tatitok import TATiTok


def get_transformer_params(transformer_size):
    if transformer_size == "small":
        transformer_params = {
            "width": 384,
            "heads": 6,
            "layers": 12,
            "mlp_ratio": 4.0,
        }
    elif transformer_size == "base":
        transformer_params = {
            "width": 512,
            "heads": 8,
            "layers": 12,
            "mlp_ratio": 4.0,
        }
    elif transformer_size == "large":
        transformer_params = {
            "width": 768,
            "heads": 12,
            "layers": 12,
            "mlp_ratio": 4.0,
        }
    else:
        raise ValueError(f"Invalid transformer size: {transformer_size}")
    return transformer_params


class TiTokImageTokenizer(nn.Module):
    def __init__(self, tokenizer_path: str, eot_token_id: int, max_text_token: int, use_eoi: bool):
        super(TiTokImageTokenizer, self).__init__()
        is_tatitok = "tatitok" in tokenizer_path
        if is_tatitok:
            self.model = TATiTok.from_pretrained(tokenizer_path)
        else:
            self.model = TiTok.from_pretrained(tokenizer_path)
        self.model.eval()
        self.eval()
        self.tokenizer_path = tokenizer_path
        self.codebook_size = self.model.config.model.vq_model.codebook_size
        self.eot_token_id = eot_token_id
        self.eoi_token_id = max_text_token + 1
        self.max_non_image_token = self.eoi_token_id if use_eoi else max_text_token
        self.use_eoi = use_eoi
        self.enable_grad = False

    def forward(self, image: torch.Tensor, append_empty_text: bool):
        bs = image.shape[0]
        with torch.set_grad_enabled(self.enable_grad):
            image = self.model.encode(image)[-1]["min_encoding_indices"].squeeze(1)
            # add max text token, so that tokens don't overlap
            image = image + self.max_non_image_token + 1
            if self.use_eoi:
                # add eoi token
                eoi = torch.full((bs, 1), self.eoi_token_id, dtype=image.dtype, device=image.device)
                image = torch.cat([image, eoi], dim=1)
            if append_empty_text:
                # append [bot, eot] tokens, make sure bot = eot - 1!
                empty_txt = torch.tensor(
                    [[self.eot_token_id - 1, self.eot_token_id]], dtype=image.dtype, device=image.device
                ).expand(bs, 2)
                image = torch.cat([image, empty_txt], dim=1)

        return image


# Adapted from https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/losses/flava.py#L143.
class MaskedPredictionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        embedding_matrix: nn.Parameter = None,  # Shared embedding matrix
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
        # use_fp32_layer_norm: bool = True,
        **kwargs: Any,
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # self.layer_norm: nn.LayerNorm
        # if use_fp32_layer_norm:
        #     self.layer_norm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
        # else:
        #     self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # # The output weights are the same as the input embeddings, but there is
        # # an output-only bias for each token.
        # self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        # self.bias = nn.Parameter(torch.zeros(vocab_size))

        # # Need a link between the two variables so that the bias is
        # # correctly resized with `resize_token_embeddings`
        # self.decoder.bias = self.bias

        # self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

        # Decoder (with tied weights but independent bias)
        #self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)  # No bias here
        self.decoder = embedding_matrix
        self.bias = nn.Parameter(torch.zeros(vocab_size))  # Independent learnable bias
        # print(self.decoder.weight.data.shape, embedding_matrix.weight.data.transpose(0, 1).shape)

        # if embedding_matrix is not None:
        #     self.decoder.weight.data = embedding_matrix.weight.data.transpose(0, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        # hidden_states = self.decoder(hidden_states) + self.bias  # Add independent bias
        hidden_states = torch.matmul(hidden_states, self.decoder.weight.t()) + self.bias
        return hidden_states


class FuseCLIP(torch.nn.Module):
    def __init__(self, model_name, text_tokenizer, image_tokenizer_path, transformer_size,
                 init_logit_scale, init_logit_bias, input_dtype, mask_pad, device=None, use_eoi=False,
                 is_master=True, mlm_probability=0., return_pred=False, mlm_right_only=False,
                 mix_left_right=0., rand_repl_probability=0., return_repl_pred=False,
                 pre_mix_left_right=0.):
        super(FuseCLIP, self).__init__()

        self.eot_token_id = text_tokenizer.eot_token_id
        self.mask_token = getattr(text_tokenizer, 'mask_token', None)
        assert isinstance(text_tokenizer, SimpleTokenizer)  # otherwise modify bot token in image tokenizer
        self.text_tokenizer = text_tokenizer

        self.max_text_token = text_tokenizer.vocab_size - 1
        if "titok" in model_name:
            self.image_tokenizer = TiTokImageTokenizer(
                tokenizer_path=image_tokenizer_path,
                eot_token_id=self.eot_token_id,
                max_text_token=self.max_text_token,
                use_eoi=use_eoi
            )
            transformer_params = get_transformer_params(transformer_size=transformer_size)
            embed_dim = transformer_params["width"]
            self.text_config = {
                "context_length": text_tokenizer.context_length,  # 128 + 1 (img + eot) + 77 (text)
                "vocab_size": text_tokenizer.vocab_size + self.image_tokenizer.codebook_size + int(use_eoi),  # 49408 (text) + 4096 (image) + 1 (eoi)
                "width": embed_dim,
                "heads": transformer_params["heads"],
                "layers": transformer_params["layers"],
                "mlp_ratio": transformer_params["mlp_ratio"],
                "ls_init_value": None,
                "no_causal_mask": True,
                "pad_id": text_tokenizer.pad_token_id,
                "proj_bias": False,
                "output_tokens": False,
                "pool_type": "eot" if isinstance(text_tokenizer, SimpleTokenizer) else "last",
            }
            self.special_tokens_ids = [  # To not be masked.
                self.eot_token_id, self.max_text_token + 1,
                text_tokenizer.pad_token_id]
            if self.mask_token is not None:
                self.special_tokens_ids.append(self.mask_token)
            # Remember which tokens are for images and which for text.
            self.txt_tokens_range = list(range(0, self.max_text_token))
            self.img_tokens_range = list(range(self.max_text_token + 1,
                self.max_text_token + 1 + self.image_tokenizer.codebook_size))
            self.txt_tokens_range = torch.tensor(
                list(set(self.txt_tokens_range) - set(self.special_tokens_ids))).long()
            self.img_tokens_range = torch.tensor(
                list(set(self.img_tokens_range) - set(self.special_tokens_ids))).long()
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        if is_master:
            logging.info(f"text config: {self.text_config}")
        self.model_name = model_name

        # we don't train the image tokenizer
        self.image_tokenizer.eval()
        for param in self.image_tokenizer.parameters():
            param.requires_grad = False

        text = _build_text_tower(
            embed_dim=embed_dim, text_cfg=self.text_config, quick_gelu=False, cast_dtype=None
        )
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.mlm_probability = mlm_probability
        self.return_pred = return_pred
        if mlm_probability > 0:
            self.mask_prediction_head = MaskedPredictionHead(
                hidden_size=embed_dim,
                vocab_size=self.text_config['vocab_size'],
                embedding_matrix=text.token_embedding,
            )
            print('decoding layer', self.mask_prediction_head.decoder.weight.data.shape)
            print('embedding layer', self.token_embedding.weight.data.shape)
            #self.mask_prediction_head.decoder.weight.data = self.token_embedding.weight.data
        self.mlm_right_only = mlm_right_only
        self.mix_left_right = mix_left_right
        self.pre_mix_left_right = pre_mix_left_right

        self.rand_repl_probability = rand_repl_probability
        self.return_repl_pred = return_repl_pred
        if rand_repl_probability > 0:
            self.repl_clf = nn.Linear(embed_dim, 2)  # predict if a token has been replaced
            print('random replacement classifier', self.repl_clf.weight.data.shape)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        self.text_projection = None
        self.mask_pad = mask_pad
        self.text_pool_type = self.text_config["pool_type"]
        self.transformer_size = transformer_size

        self.output_dict = True

        self.input_dtype = input_dtype
        self.device = device
        self.to(device)


    def get_config(self):
        config = {
            "model_name": self.model_name,
            "text_config": self.text_config,
            "mask_pad": self.mask_pad,
            "image_tokenizer": self.image_tokenizer.tokenizer_path,
            "use_eoi": self.image_tokenizer.use_eoi,
            "context_length": self.context_length,
            "transformer_size": self.transformer_size,
            "mlm_probability": self.mlm_probability,
            "rand_repl_probability": self.rand_repl_probability,
        }
        return config

    def tokenize_image(self, image, append_empty_text):
        if isinstance(image, list):
            image_ids = self.tokenize_image_list(image, append_empty_text)
        elif image is None or image.numel() == 0:
            return image
        else:
            image = image.to(device=self.device, dtype=self.input_dtype, non_blocking=True)
            image_ids = self.image_tokenizer(image, append_empty_text=append_empty_text)
        return image_ids

    def tokenize_image_list(self, image: list, append_empty_text: bool) -> List[torch.Tensor]:
        # remember positions where image is None
        no_image_pos = [i for i, img in enumerate(image) if img is None]
        # stack all other into tensor
        non_empty_images = [img for img in image if img is not None]
        if len(non_empty_images) > 0:
            image_tensor = torch.stack(non_empty_images, dim=0)
            image_tensor = image_tensor.to(device=self.device, dtype=self.input_dtype, non_blocking=True)
            # tokenize them
            image_ids = self.image_tokenizer(image_tensor, append_empty_text=append_empty_text)
        else:
            image_ids = []

        # where image was None, put None
        image_ids_lst = []
        cur_img_idx = 0
        for i in range(len(image)):
            if i in no_image_pos:
                image_ids_lst.append(None)
            else:
                image_ids_lst.append(image_ids[cur_img_idx])
                cur_img_idx += 1
        return image_ids_lst

    def merge_image_text_ids(self, image_ids: List[torch.Tensor], text_ids: torch.Tensor) -> torch.Tensor:
        # image ids first, then text ids
        res = []
        for i in range(len(image_ids)):
            if image_ids[i] is None or image_ids[i].numel() == 0:
                res.append(self.truncate_tokens(text_ids[i]))
            else:
                res.append(
                    self.truncate_tokens(torch.cat([image_ids[i], text_ids[i]], dim=0))
                )
        res =  torch.stack(res, dim=0)
        return res

    def truncate_tokens(self, tokens):
        squeeze = False
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
            squeeze = True

        if tokens.shape[1] >= self.context_length:
            tokens = tokens[:, :self.context_length]
            # if eot is not present, put it at final element
            mask = (tokens == self.eot_token_id).sum(dim=1) == 0
            tokens[mask, -1] = self.eot_token_id
            # tokens[:, -1] = self.eot_token_id
        if squeeze:
            tokens = tokens.squeeze(0)

        return tokens

    def check_eot_present(self, tokens: torch.Tensor):
        # check that eot is present exactly once
        eot_count = (tokens == self.eot_token_id).sum(dim=1)
        if not torch.all(eot_count == 1):
            # raise ValueError(f"EOT token not present exactly once in all samples: {eot_count}")
            logging.error(f"EOT token not present exactly once in all samples: {eot_count}")
            logging.error(f"counts unequal one: {eot_count[eot_count != 1]}")
            logging.error(f"tokens: {tokens[eot_count != 1]}")
            return False
        return True


    def forward(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None,
                force_fused: bool = False):
        if force_fused:
            return self.encode_multimodal(image, text, normalize=True)

        is_multimodal = isinstance(image, tuple)
        if is_multimodal:
            image_left, image_right = image
            text_left, text_right = text
            # print('orig', len(image_left), image_left, image_right)
            if self.pre_mix_left_right > 0:
                assert isinstance(text_left, torch.Tensor), type(text_left)
                assert isinstance(text_right, torch.Tensor), type(text_right)
                assert isinstance(image_left, list), type(image_left)
                assert isinstance(image_right, list), type(image_right)
                bs = text_left.shape[0]
                alpha = (torch.rand((bs)) > self.pre_mix_left_right)
                # Text should be already tokenized and converted to tensors.
                alpha_text = alpha.long().view(-1, 1).to(text_left.device)
                _text_left = text_left * alpha_text + text_right * (1 - alpha_text)
                _text_right = text_left * (1 - alpha_text) + text_right * alpha_text
                text_left = _text_left
                text_right = _text_right
                # Images could be missing and should be list.
                _image_left, _image_right = [], []
                for side, _l, _r in zip(alpha, image_left, image_right):
                    if side:  # Keep original side.
                        _image_left.append(_l)
                        _image_right.append(_r)
                    else:
                        _image_left.append(_r)
                        _image_right.append(_l)
                image_left = _image_left
                image_right = _image_right
            # print('mixed', len(image_left), image_left, image_right)
            # Get features.
            left_features = self.encode_multimodal(
                image_left, text_left, normalize=True, skip_mask=self.mlm_right_only)
            del image_left, text_left
            right_features = self.encode_multimodal(
                image_right, text_right, normalize=True, skip_mask=False)
        else:
            left = self.tokenize_image(image, append_empty_text=True)
            right = text
            left_features = self.encode_text(left, normalize=True) if left is not None else None
            right_features = self.encode_text(right, normalize=True) if right is not None else None

        if self.mix_left_right > 0 and left_features is not None and right_features is not None:
            if isinstance(left_features, dict):
                bs = left_features['fts'].shape[0]
                alpha = (torch.rand((bs)) > self.mix_left_right).float().view(
                    -1, 1).to(left_features['fts'].device)
                _left_features = left_features['fts'] * alpha + right_features['fts'] * (1. - alpha)
                _right_features = left_features['fts'] * (1. - alpha) + right_features['fts'] * alpha
                left_features['fts'] = _left_features
                right_features['fts'] = _right_features
            elif isinstance(left_features, torch.Tensor):
                bs = left_features.shape[0]
                alpha = (torch.rand((bs)) > self.mix_left_right).float().view(
                    -1, 1).to(left_features.device)
                _left_features = left_features * alpha + right_features * (1. - alpha)
                _right_features = left_features * (1. - alpha) + right_features * alpha
                left_features = _left_features + 0.
                right_features = _right_features + 0.
            else:
                raise ValueError(f'Unknown format: {type(left_features)}.')

        if self.return_pred or self.return_repl_pred:
            if self.output_dict:
                out_dict = {
                    "image_features": left_features['fts'],
                    "text_features": right_features['fts'],
                    "logit_scale": self.logit_scale.exp(),
                }
                if self.return_pred:
                    new_out = {
                        "left_logits": left_features['logits'],
                        "left_labels": left_features['labels'],
                        'left_ignore_mlm_loss': left_features['ignore_mlm_loss'],
                        "right_logits": right_features['logits'],
                        "right_labels": right_features['labels'],
                        'right_ignore_mlm_loss': right_features['ignore_mlm_loss'],
                    }
                    out_dict.update(new_out)
                if self.return_repl_pred:
                    new_out = {
                        "left_repl_logits": left_features['repl_logits'],
                        "left_repl_labels": left_features['repl_labels'],
                        "right_repl_logits": right_features['repl_logits'],
                        "right_repl_labels": right_features['repl_labels'],
                    }
                    out_dict.update(new_out)
                if self.logit_bias is not None:
                    out_dict["logit_bias"] = self.logit_bias
                return out_dict
            raise NotImplementedError()

        if self.output_dict:
            out_dict = {
                "image_features": left_features,
                "text_features": right_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict
        else:
            return left_features, right_features, self.logit_scale.exp()

    def encode_multimodal(
            self, image, text, normalize: bool = True, skip_mask: bool = False):
        # encode image + text as one sequence
        image_ids = self.tokenize_image(image, append_empty_text=False)
        inp = self.merge_image_text_ids(image_ids, text)
        return self.encode_text(inp, normalize=normalize, skip_mask=skip_mask)

    def encode_image(
            self, image: torch.Tensor, normalize: bool = True,skip_mask: bool = False):
        image_ids = self.tokenize_image(image, append_empty_text=True)
        return self.encode_text(image_ids, normalize=normalize, skip_mask=skip_mask)

    def encode_text(self, text: torch.Tensor, normalize: bool = True,
                    skip_mask: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        if self.mask_pad:
            # cut off at the longest sequence
            longest = (text == self.eot_token_id).long().argmax(dim=1).max() + 1
            text = text[:, :longest]

        if self.mlm_probability > 0 and self.return_pred:
            if not skip_mask:
                # Replace random tokens with [MASK] (excluding special tokens).
                text, labels = self.mask_input(text, self.mlm_probability)
            else:
                # Skip masking for this input (e.g. when for text only).
                labels = None
        if self.rand_repl_probability > 0 and self.return_repl_pred:
            text, repl_labels = self.replace_tokens(text, self.rand_repl_probability)

        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        if self.attn_mask is not None:
            raise NotImplementedError()

        attn_mask = self.attn_mask
        if self.mask_pad:
            attn_mask = self.build_pad_mask(text, cast_dtype)

        # we could have different length for text and image, so we truncate the positional embedding accordingly
        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = self.transformer(x, key_padding_mask=attn_mask)
        ignore_mlm_loss = False
        if self.mlm_probability > 0 and self.return_pred:
            if not skip_mask:
                # Predict masked tokens.
                # TODO: check if here or after LN.
                _masked_tokens = labels.ne(-100)  # To compute predictions only for masked tokens.
                if _masked_tokens.long().sum() == 0:
                    #_masked_tokens.fill_(False)  # To be removed.
                    _masked_tokens[0, 0] = True
                    ignore_mlm_loss = True
                labels = labels[_masked_tokens].view(-1)
                # print(labels)
                _sequence_output = x[_masked_tokens, :]
                mask_pred = self.mask_prediction_head(_sequence_output).view(-1, self.vocab_size)
                    # print(mask_pred.shape, labels.shape)
                    # mask_pred = mask_pred.view(-1, self.vocab_size)
                    # labels = labels.view(-1)
                    # print(mask_pred.shape, labels.shape)
                # else:
                #     mask_pred = None
                #     labels = None
            else:
                mask_pred = None
        if self.rand_repl_probability > 0 and self.return_repl_pred:
            repl_pred = self.repl_clf(x)
            # print(x.shape, repl_pred.shape)
        x = self.ln_final(x)
        x, tokens = self.text_pool(x, text, pool_type=self.text_pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        if ((self.mlm_probability > 0 and self.return_pred) or
            (self.rand_repl_probability > 0 and self.return_repl_pred)):
            out_dict = {'fts': F.normalize(x, dim=-1) if normalize else x}
            if self.mlm_probability > 0 and self.return_pred:
                new_out = {
                    'logits': mask_pred,
                    'labels': labels,
                    'ignore_mlm_loss': ignore_mlm_loss,
                }
                out_dict.update(new_out)
            if self.rand_repl_probability > 0 and self.return_repl_pred:
                new_out = {
                    'repl_logits': repl_pred,
                    'repl_labels': repl_labels,
                }
                out_dict.update(new_out)
            return out_dict
        return F.normalize(x, dim=-1) if normalize else x

    def build_pad_mask(self, text, cast_dtype: torch.dtype):
        attn_mask = (text == self.text_config["pad_id"])
        return attn_mask

    def text_pool(self, x, text: Optional[torch.Tensor] = None, pool_type: str = "eot"):
        if pool_type == 'eot':
            # take features from the eot embedding
            assert text is not None
            self.check_eot_present(text)
            pooled = x[
                torch.arange(x.shape[0]),
                (text == self.eot_token_id).to(torch.float32).argmax(dim=-1)
            ]
            tokens = x
        elif pool_type == 'last':
            pooled, tokens = x[:, -1], x[:, :-1]
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")

        return pooled, tokens

    def mask_input(self, input, mlm_probability=.15):

        #print(input.shape)

        # Create a tensor of probabilities
        probs = torch.full(input.shape, 1 - mlm_probability, device=input.device)

        # Sample from Bernoulli distribution
        mask = torch.bernoulli(probs).to(bool).to(input.device)  # Tokens not to be masked.
        for id in self.special_tokens_ids:
            mask = mask + (input == id)  # Keep special tokens (eot, eoi, pad).

        masked_input = torch.where(mask, input, self.mask_token)
        labels = torch.where(mask, -100, input)  # -100 as label of non [MASK] token

        #print(masked_input.shape, labels.shape)

        return masked_input, labels
    
    def replace_tokens(self, input, p=.1):

        probs = torch.full(input.shape, 1 - p, device=input.device)
        mask = torch.bernoulli(probs).to(bool).to(input.device)  # Tokens not to be replaced.
        # mask = mask + (input <= self.max_text_token)  # Only image tokens for now.
        for id in self.special_tokens_ids:
            mask = mask + (input == id)  # Keep special tokens (eot, eoi, pad).

        # Replace image tokens.
        indices = torch.randint(0, len(self.img_tokens_range), input.shape)
        rand_img_tokens = self.img_tokens_range[indices].to(input.device)
        img_mask = mask + (input <= self.max_text_token)
        new_input = torch.where(img_mask, input, rand_img_tokens)

        # Replace text tokens.
        indices = torch.randint(0, len(self.txt_tokens_range), input.shape)
        rand_txt_tokens = self.txt_tokens_range[indices].to(input.device)
        txt_mask = mask + (input > self.max_text_token)
        new_input = torch.where(txt_mask, new_input, rand_txt_tokens)

        #new_input = torch.where(mask, input, rand_img_tokens)
        labels = mask.long()  # Include all tokens in the loss (0 -> synthetic, 1 -> real).
        return new_input, labels

    def set_return_preds(self, val=True):
        self.return_pred = val

    def set_return_repl_preds(self, val=True):
        self.return_repl_pred = val

    def train(self, mode: bool = True):
        # put in train mode
        super().train(mode)
        # but leave image tokenizer in eval mode
        self.image_tokenizer.eval()
        return self