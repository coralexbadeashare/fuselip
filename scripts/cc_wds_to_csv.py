import os
import random
import csv
import warnings
from joblib import Parallel, delayed
import pandas as pd
from PIL import Image
from tqdm import tqdm


def iter_txt_files(root_dir):
    """Yield .txt files recursively using scandir (fully streaming)."""
    for entry in os.scandir(root_dir):
        if entry.is_file() and entry.name.endswith(".txt"):
            yield entry.path
        elif entry.is_dir():
            yield from iter_txt_files(entry.path)


def convert_streaming(data_path, csv_path_all):
    """
    Stream through .txt caption files and write CSV line-by-line.
    Avoids memory buildup and produces relative paths.
    """
    print(f"[INFO] Starting streaming conversion in: {data_path}")
    not_found = []
    total = 0
    valid = 0

    # Prepare base for relative paths
    cwd = os.getcwd()

    # Open CSV for streaming write
    with open(csv_path_all, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "caption"])  # header

        for txt_path in iter_txt_files(data_path):
            total += 1
            base_name = os.path.splitext(txt_path)[0]
            img_path = base_name + ".jpg"

            # Try .png fallback
            if not os.path.exists(img_path):
                alt_img_path = base_name + ".png"
                if os.path.exists(alt_img_path):
                    img_path = alt_img_path
                else:
                    not_found.append(txt_path)
                    continue

            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()

                # Get path relative to where script is run
                rel_path = os.path.relpath(img_path, cwd)

                writer.writerow([rel_path, caption])
                valid += 1

                # Flush every few hundred to ensure safety
                if valid % 500 == 0:
                    csvfile.flush()

                # Print progress every 1000
                if valid % 1000 == 0:
                    print(f"[STREAM] {valid} captions processed...")

            except Exception as e:
                print(f"[WARN] Could not read {txt_path}: {e}")

    print(f"[INFO] Stream complete. {valid}/{total} valid entries written.")
    print(f"[WARN] Missing images: {len(not_found)}")
    if not_found:
        print(f"Example missing: {not_found[:3]}")
    return csv_path_all


def _check_image_validity(idx, path):
    """Check if image file exists and can be opened."""
    if not os.path.exists(path):
        return (idx, False)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            return (idx, False)
    try:
        with Image.open(path) as img:
            img.load()
            if img.size[0] == 0 or img.size[1] == 0:
                return (idx, False)
    except Exception:
        return (idx, False)
    return (idx, True)


def remove_invalid(csv_path, n_jobs=8):
    """Read CSV, validate images, and rewrite only valid ones."""
    print("[INFO] Checking image validity...")
    df = pd.read_csv(csv_path)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_check_image_validity)(i, path)
        for i, path in tqdm(enumerate(df["path"].tolist()), total=len(df), desc="Validating")
    )
    invalid_indices = [idx for (idx, valid) in results if not valid]
    print(f"[INFO] Found {len(invalid_indices)} invalid images.")
    df = df.drop(invalid_indices).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] {len(df)} valid samples remain.")
    return df


def split_train_val(df, val_ratio=0.2, seed=42):
    """Shuffle and split into train and val sets."""
    print(f"[INFO] Splitting dataset with {val_ratio*100:.0f}% for validation...")
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = int(len(shuffled) * val_ratio)
    df_val = shuffled.iloc[:val_size]
    df_train = shuffled.iloc[val_size:]
    print(f"[INFO] -> Train: {len(df_train)} | Val: {len(df_val)}")
    return df_train, df_val


if __name__ == "__main__":
    data_path = "./cc3m_data/cc3m_extracted"
    csv_all = "./cc3m_data/all_data.csv"
    csv_train = "./cc3m_data/train.csv"
    csv_val = "./cc3m_data/val.csv"

    print("===============================================")
    print("[STEP 1] Streaming conversion to CSV")
    print("===============================================")
    convert_streaming(data_path, csv_all)

    print("\n===============================================")
    print("[STEP 2] Removing invalid images")
    print("===============================================")
    df = remove_invalid(csv_all, n_jobs=8)

    print("\n===============================================")
    print("[STEP 3] Splitting into train/val")
    print("===============================================")
    df_train, df_val = split_train_val(df)

    df_train.to_csv(csv_train, index=False)
    df_val.to_csv(csv_val, index=False)

    print("\n===============================================")
    print(f"[DONE] Saved:\n  -> {csv_train}\n  -> {csv_val}")
    print("===============================================")
