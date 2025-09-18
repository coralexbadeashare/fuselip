from datasets import load_dataset
import os

def download_rocov2(target_dir: str = "./ROCOv2_data"):
    """
    Downloads the ROCOv2 radiology dataset (images + captions) from Hugging Face.
    Requires: pip install datasets pillow
    """
    ds = load_dataset("eltorio/ROCOv2-radiology")

    os.makedirs(target_dir, exist_ok=True)
    for split in ds.keys():  # 'train', 'validation', 'test'
        split_path = os.path.join(target_dir, split)
        os.makedirs(split_path, exist_ok=True)
        print(f"Downloading split: {split} ({len(ds[split])} samples)")

        for idx, example in enumerate(ds[split]):
            image = example["image"]  # PIL Image
            caption = example["caption"]
            image_id = example.get("image_id", f"{split}_{idx}")

            # save image
            img_filename = os.path.join(split_path, f"{image_id}.jpg")
            image.save(img_filename)

            # save caption
            with open(img_filename + ".txt", "w", encoding="utf-8") as f:
                f.write(caption)

    print("✅ Finished downloading ROCOv2 into", target_dir)

if __name__ == "__main__":
    download_rocov2()
