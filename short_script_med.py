#!/usr/bin/env python3
import os
import sys
import glob
import warnings
import torch
from PIL import Image

# ensure local "src" is on sys.path so "fuse_clip" can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(SCRIPT_DIR)
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from fuse_clip.fuse_clip_utils import load_model

# Config
device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "./logs/2025_11_14-13_01_05-model_fuse-clip-titok-lr_0.001-b_256-j_16-p_amp/checkpoints/epoch_10.pt"

# Load base model + processors (will override weights if checkpoint present)
model, image_processor, text_tokenizer = load_model("chs20/FuseLIP-S-CC3M-MM", device=device)

# Try to load provided local checkpoint into model (non-strict)
if os.path.exists(CKPT):
    print(f"[INFO] Loading local checkpoint: {CKPT}")
    ck = torch.load(CKPT, map_location=device)
    sd = None
    if isinstance(ck, dict):
        for key in ("state_dict", "model_state_dict", "model", "state", "model_state"):
            if key in ck and isinstance(ck[key], dict):
                sd = ck[key]
                break
    # sometimes the state dict keys are nested under "state_dict" with "module." prefixes
    if sd is None and isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
        sd = ck
    if sd is None:
        # try to find sub-dict with tensor values
        for v in ck.values() if isinstance(ck, dict) else []:
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                sd = v
                break
    if sd is None:
        print("[WARN] Could not detect nested state_dict; attempting to use checkpoint object directly")
        sd = ck
    try:
        model.load_state_dict(sd, strict=False)
        print("[INFO] Checkpoint loaded into model (strict=False).")
    except Exception as e:
        print("[ERROR] Failed to load checkpoint into model:", e)
else:
    print(f"[WARN] Checkpoint not found at {CKPT}; using hub weights")

# Select image: first CLI arg or prefer file ending with _000001.jpg else first match
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    test_dir = os.path.join(REPO_ROOT, "ROCOv2_data", "test")
    candidates = sorted(glob.glob(os.path.join(test_dir, "ROCOv2_2023_test_*.jpg")))
    if not candidates:
        print("No ROCOv2 test images found under ./ROCOv2_data/test. Pass image path as first arg.")
        sys.exit(1)
    preferred = [p for p in candidates if p.endswith("_000001.jpg")]
    img_path = preferred[0] if preferred else candidates[0]

# Load and preprocess image
img = Image.open(img_path).convert("RGB")
pixel_values = image_processor(img).unsqueeze(0).to(device)

# Candidate captions (one requested, two more)
texts = [
    "CT chest axial view showing a huge ascending aortic aneurysm (*)",
    "CT scan of the abdomen showing hepatosplenomegaly with post-surgical changes",
    "CT chest axial image demonstrating cardiomegaly with pericardial effusion"
]

# Tokenize / encode texts via tokenizer (handle tokenizer API variations)
token_tensors = []
for t in texts:
    if hasattr(text_tokenizer, "encode"):
        toks = text_tokenizer.encode(t)
    elif hasattr(text_tokenizer, "tokenize"):
        toks = text_tokenizer.tokenize(t)
    else:
        raise RuntimeError("text_tokenizer has no encode/tokenize method")
    token_tensors.append(torch.tensor(toks, dtype=torch.long))

# Pad to same length
token_ids = torch.nn.utils.rnn.pad_sequence(token_tensors, batch_first=True, padding_value=0).to(device)

# Inference
model.eval()
with torch.no_grad():
    image_features = model.encode_image(pixel_values)
    text_features = model.encode_text(token_ids)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sims = (image_features @ text_features.T)[0].cpu().numpy()

# Print results
print(f"Image: {img_path}")
for txt, score in zip(texts, sims):
    print(f"{score:.4f}  ->  {txt}")

best = texts[sims.argmax()]
print("\nPredicted best match:", best)
