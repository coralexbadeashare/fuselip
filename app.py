import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
import requests

# Add FuseLIP src folder to Python path
sys.path.append(os.path.abspath("./src"))

from fuse_clip.fuse_clip_utils import load_model

# -------------------------------
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained FuseLIP
model_id = "chs20/FuseLIP-S-CC3M-MM"
model, image_processor, text_tokenizer = load_model(model_id, device=device)
model.eval()
print(f"Loaded {model_id} on {device}")

# -------------------------------
# Load test image
img_url = "https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9"
img = Image.open(BytesIO(requests.get(img_url).content)).convert("RGB")
print("Image loaded:", img.size, img.mode)

# Preprocess image (torchvision transforms pipeline)
pixel_values = image_processor(img).unsqueeze(0).to(device)

# Encode image
with torch.no_grad():
    if hasattr(model, "encode_image"):
        img_emb = model.encode_image(pixel_values)
    elif hasattr(model, "forward_image"):
        img_emb = model.forward_image(pixel_values)
    else:
        out = model(pixel_values)
        img_emb = out["image_embeds"] if isinstance(out, dict) else out

# Normalize image embedding
img_emb = F.normalize(img_emb, dim=-1)

# -------------------------------
# Candidate captions
captions = [
    "A cute dog sitting on the floor",
    "A plate of delicious pasta",
    "A car parked on the street",
    "A cat sleeping on the couch"
]

# Tokenize text with FuseLIP's SimpleTokenizer
text_tokens = text_tokenizer(captions)  # returns token IDs tensor
# Move to device
if isinstance(text_tokens, dict):
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
else:
    text_tokens = text_tokens.to(device)

# Encode text
with torch.no_grad():
    if hasattr(model, "encode_text"):
        text_emb = model.encode_text(text_tokens)
    else:
        out = model(text_tokens=text_tokens)
        text_emb = out["text_embeds"] if isinstance(out, dict) else out

# Normalize text embeddings
text_emb = F.normalize(text_emb, dim=-1)

# -------------------------------
# Compute cosine similarity
similarities = (img_emb @ text_emb.T).squeeze(0).cpu().numpy()

# Print results
for cap, score in zip(captions, similarities):
    print(f"{cap:35s} -> similarity {score:.4f}")

best_idx = int(np.argmax(similarities))
print("\nBest match:", captions[best_idx])
