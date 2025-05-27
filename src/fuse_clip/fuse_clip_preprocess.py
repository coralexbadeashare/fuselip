from torchvision import transforms

from open_clip.tokenizer import SimpleTokenizer
from open_clip.transform import _convert_to_rgb


def get_fuse_clip_image_preprocess(train: bool) -> transforms.Compose:
    if train:
        image_processor = transforms.Compose([
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                                         interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(256),
            _convert_to_rgb,
            transforms.ToTensor(),
        ])
    else:
        image_processor = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(256),
            _convert_to_rgb,
            transforms.ToTensor(),
        ])

    return image_processor

def get_fuse_clip_text_tokenizer(model_name: str, context_length: int) -> callable:
    # context_length = 180 # 77, 129, 206

    # if simple-tokenizer:
    tokenizer = SimpleTokenizer(context_length=context_length)
    tokenizer.pad_token_id = 0

    # if siglip-tokenizer:
    # tokenizer = HFTokenizer(
    #     "timm/ViT-B-16-SigLIP",
    #     context_length=context_length,
    # )
    # tokenizer.vocab_size = tokenizer.tokenizer.vocab_size
    # tokenizer.pad_token_id = tokenizer.tokenizer.pad_token_id
    # tokenizer.eot_token_id = tokenizer.tokenizer.eos_token_id

    tokenizer.context_length = context_length
    return tokenizer


def add_mask_token(tokenizer):
    """Add a special [MASK] token to tokenizer if not present."""

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


if __name__ == '__main__':
    pass
