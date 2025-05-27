import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from fuse_clip.fuse_clip_arch import FuseCLIP
from open_clip import get_input_dtype, SimpleTokenizer


class FuseLIP(FuseCLIP, PyTorchModelHubMixin):
    """FuseLIP with save_pretrained / from_pretrained / push_to_hub."""

    # ---------- save ----------
    def _save_pretrained(self, save_directory: Path, **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        (save_directory / "config.json").write_text(
            json.dumps(self.get_config(), indent=2)
        )

    # ---------- load ----------
    @classmethod
    def _from_pretrained(cls, model_id, **kwargs):
        cfg_path = hf_hub_download(
            repo_id=model_id, filename="config.json", revision=kwargs.get("revision"),
            cache_dir=kwargs.get("cache_dir"), force_download=kwargs.get("force_download")
        )
        cfg = json.loads(Path(cfg_path).read_text())

        tokenizer = SimpleTokenizer(context_length=cfg["context_length"])
        tokenizer.pad_token_id = 0

        if cfg["mlm_probability"] > 0:
            MASK_TOKEN = "[MASK]"
            if MASK_TOKEN not in tokenizer.encoder:
                # Assign a new token ID
                mask_token_id = max(tokenizer.encoder.values()) + 1  # Get a new unique ID

                # Add to tokenizer's vocabulary
                tokenizer.encoder[MASK_TOKEN] = mask_token_id
                tokenizer.decoder[mask_token_id] = MASK_TOKEN

                tokenizer.all_special_ids.append(mask_token_id)
                tokenizer.mask_token = mask_token_id
                tokenizer.vocab_size += 1

                print(f"Added `[MASK]` token with ID {mask_token_id}")
            else:
                mask_token_id = tokenizer.encoder[MASK_TOKEN]
                print(f"`[MASK]` token already exists with ID {mask_token_id}")

        cfg["image_tokenizer_path"] = cfg["image_tokenizer"]
        cfg["init_logit_scale"] = np.log(10)
        cfg["init_logit_bias"] = -10
        cfg["input_dtype"] = get_input_dtype("fp32")
        cfg["text_tokenizer"] = tokenizer
        del cfg["text_config"]
        del cfg["image_tokenizer"]
        del cfg["context_length"]

        model = cls(**cfg, device=kwargs.get("device"))  # device / dtype can be injected via kwargs
        model.mlm_probability = 0.0  # we set it only so that the head is initialized and weights can be loaded

        state_path = hf_hub_download(
            repo_id=model_id, filename="pytorch_model.bin", revision=kwargs.get("revision"),
            cache_dir=kwargs.get("cache_dir"), force_download=kwargs.get("force_download")
        )
        state = torch.load(
            state_path,
            map_location=kwargs.get("device", "cpu"),  # device can be injected via kwargs
            weights_only=True
        )
        model.load_state_dict(state, strict=True)
        return model
