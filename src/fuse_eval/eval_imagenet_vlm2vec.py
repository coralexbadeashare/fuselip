import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from tqdm import tqdm

from fuse_clip.fuse_clip_utils import load_model
from open_clip import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES
from vlm2vec_new.eval import batch_to_device
from vlm2vec_new.vlm2vec_imagenet_class_names import VLM2VEC_IMAGENET_CLASSNAMES


@torch.inference_mode()
def eval_imagenet_vlm2vec(model, collator, txt_tokenizer=None):
    # Load ImageNet validation dataset
    data_path = "/mnt/datasets/imagenet/val/"
    dataset = datasets.ImageFolder(data_path, transform=None)

    device = next(model.parameters()).device

    # build zero-shot classifier by averaging class name embeddings over templates
    zeroshot_weights = build_zeroshot_weights_vlm2vec(
        model, collator,
        VLM2VEC_IMAGENET_CLASSNAMES if use_vlm2vec_classnames else IMAGENET_CLASSNAMES,
        OPENAI_IMAGENET_TEMPLATES if use_ensemble else [lambda x: x],
        batch_size=batch_size_templates,
        device=device
        )

    # shuffle the dataset
    rng = np.random.default_rng(0)
    shuffled_idcs = rng.permutation(len(dataset))

    # run zero-shot classification
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    print(f"\nrunning zero-shot classification on {len(dataset)} samples at batch size {batch_size_classification}")

    pbar = tqdm(range(0, len(dataset), batch_size_classification))
    for i in pbar:
        end_idx = min(i + batch_size_classification, len(dataset))
        batch_indices = [shuffled_idcs[j] for j in range(i, end_idx)]  # Use shuffled indices from shuffled_idcs
        batch_images = []
        batch_labels = []

        # Get images and labels
        for idx in batch_indices:
            img_path, label = dataset.samples[idx]
            image = Image.open(img_path).convert('RGB')
            batch_images.append(image)
            batch_labels.append(label)

        labels = torch.tensor(batch_labels).to(device)

        # Prepare batch inputs
        text_prompt = "<|image_1|>\nRepresent the given image for classification\n"
        image_inputs = []
        for image in batch_images:
            image_inputs.append((text_prompt, image))

        # Process inputs using the collator
        inputs = collator(image_inputs)

        # Move inputs to device
        inputs = batch_to_device(inputs, device)

        # Get image features
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            image_features = model(qry=inputs)["qry_reps"]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute similarity with class embeddings
        logits = 100. * image_features @ zeroshot_weights

        # Get predictions
        top1_pred = logits.argmax(dim=-1)
        top5_pred = logits.topk(5, dim=-1).indices

        # Count correct predictions
        correct_top1 += (top1_pred == labels).sum().item()
        correct_top5 += sum(labels[i] in top5_pred[i] for i in range(len(labels)))
        total += len(labels)
        
        # Update progress bar with current top1 accuracy
        current_top1_acc = (correct_top1 / total) * 100
        pbar.set_postfix(top1_acc=f"{current_top1_acc:.2f}%")

    # Calculate final accuracy
    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total

    results = {
        "imagenet-top1": top1_accuracy * 100,
        "imagenet-top5": top5_accuracy * 100,
    }

    return results


@torch.inference_mode()
def build_zeroshot_weights_vlm2vec(model, collator, classnames, templates, device, batch_size):
    """
    Build zero-shot weights for VLM2Vec model by encoding class names with templates
    """
    print(f"Building zero-shot weights for {len(classnames)} classes with {len(templates)} templates "
          f"at batch size {batch_size}")

    zeroshot_weights = []

    # Process all classes with templates
    for classname in tqdm(classnames):
        # Process templates in batches
        template_batches = [templates[i:i + batch_size] for i in range(0, len(templates), batch_size)]
        class_embeddings = []

        for template_batch in template_batches:
            # Create prompts for all templates in the batch
            text_inputs = []
            for template in template_batch:
                prompt = template(classname)
                text_inputs.append([prompt, None])

            # Process batch of prompts
            inputs = collator(text_inputs)

            # Move to device
            inputs = batch_to_device(inputs, device)
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # Get text embeddings for the batch
                text_features = model(tgt=inputs)["tgt_reps"]
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Add embeddings from this batch to the list
            class_embeddings.extend(text_features)

        # Average embeddings over all templates
        class_embedding = torch.stack(class_embeddings).mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        zeroshot_weights.append(class_embedding)

    # Stack all class embeddings
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

    return zeroshot_weights


if __name__ == '__main__':
    # set seeds
    np.random.seed(0)
    torch.manual_seed(0)

    use_ensemble = True
    use_vlm2vec_classnames = True
    batch_size_templates = 1000
    batch_size_classification = 64

    model_id = "tiger-lab-vlm2vec-full"
    model, (processor, collator), _ = load_model(model_id, "cuda")

    print("\n" + "=" * 50 + "\n")
    print(f"\nuse ensemble: {use_ensemble}")
    print(f"use vlm2vec classnames: {use_vlm2vec_classnames}\n")
    results = eval_imagenet_vlm2vec(model=model, collator=collator)

    print("\n" + "=" * 50 + "\n")
    print(f"Results: {results}")