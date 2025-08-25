import torch
from PIL import Image
from fuse_clip.fuse_clip_utils import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, image_processor, text_tokenizer = load_model("chs20/FuseLIP-S-CC3M-MM", device=device)

# Load your CIFAR-100 image
img = Image.open("eval_images/Place365/Places365_val_00000050.jpg").convert("RGB")
pixel_values = image_processor(img).unsqueeze(0).to(device)

# Candidate answers (you can add more relevant classes here)
texts = ["a cat", "a dog", "an airplane", "a truck", "a flower"]

# Tokenize with SimpleTokenizer
max_len = 180  # from FuseLIP config
text_token_ids = [torch.tensor(text_tokenizer.encode(t), dtype=torch.long) for t in texts]

# Pad to same length
text_token_ids = torch.nn.utils.rnn.pad_sequence(
    text_token_ids, batch_first=True, padding_value=0
).to(device)

with torch.no_grad():
    image_features = model.encode_image(pixel_values)
    text_features = model.encode_text(text_token_ids)

    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Similarity
    sims = (image_features @ text_features.T)[0].cpu().numpy()

# Show results
for txt, score in zip(texts, sims):
    print(f"{txt} → {score:.4f}")

best = texts[sims.argmax()]
print("\nPredicted best match:", best)
