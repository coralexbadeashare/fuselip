#!/usr/bin/env python3
# Script: create <name>_roco.csv files from ROCOv2 .txt caption files
# Produces: all_roco.csv, train_roco.csv, validation_roco.csv, test_roco.csv (if dirs present)

import os
import csv
import sys
import warnings
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
import pandas as pd

def iter_txt_files(root_dir):
    """Yield full paths to .txt files under root_dir (recursive)."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                yield os.path.join(dirpath, fn)

def _find_image_for_txt(txt_path):
    """
    Given a .txt path, return the existing image path or None.
    Handles names like "image.jpg.txt" and "image.txt".
    Tries .jpg then .png if needed.
    """
    base = os.path.splitext(txt_path)[0]  # removes only the final .txt
    candidates = []
    if base.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
        candidates = [base]
    else:
        candidates = [base + ext for ext in (".jpg", ".jpeg", ".png", ".webp")]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None

def read_caption_from_txt(txt_path):
    """Read caption text from .txt file (strip whitespace)."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text.replace("\n", " ").strip()
    except Exception:
        return ""

def convert_dir_to_csv(data_root, out_csv):
    """
    Stream through .txt caption files under data_root and write CSV rows:
    path,caption
    Paths are written relative to the current working directory.
    """
    print(f"[INFO] Converting '{data_root}' -> '{out_csv}'")
    not_found = []
    total = 0
    valid = 0

    cwd = os.getcwd()
    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "caption"])
        for txt_path in tqdm(list(iter_txt_files(data_root)), desc=f"Scanning {os.path.basename(data_root)}"):
            total += 1
            img_path = _find_image_for_txt(txt_path)
            if img_path is None:
                not_found.append(txt_path)
                continue
            caption = read_caption_from_txt(txt_path)
            rel_path = os.path.relpath(img_path, start=cwd)
            writer.writerow([rel_path, caption])
            valid += 1

    print(f"[INFO] Wrote {valid}/{total} entries to {out_csv}")
    if not_found:
        print(f"[WARN] {len(not_found)} .txt files without matching image (examples):")
        for p in not_found[:5]:
            print("  -", p)
    return out_csv

def _check_image_validity(idx, path):
    """Return (idx, True/False) depending on whether image exists and can be opened."""
    if not os.path.exists(path):
        return (idx, False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(path)
            img.verify()
        return (idx, True)
    except Exception:
        return (idx, False)

def remove_invalid(csv_path, n_jobs=None):
    """Validate image files listed in csv_path and rewrite CSV with only valid rows."""
    n_jobs = n_jobs or min(8, (os.cpu_count() or 1))
    print(f"[INFO] Validating images in {csv_path} using n_jobs={n_jobs} ...")
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[WARN] CSV is empty.")
        return df
    paths = df["path"].tolist()
    results = Parallel(n_jobs=n_jobs)(
        delayed(_check_image_validity)(i, p) for i, p in enumerate(tqdm(paths, desc="Validating"))
    )
    invalid_indices = [i for (i, ok) in results if not ok]
    print(f"[INFO] Found {len(invalid_indices)} invalid images.")
    if invalid_indices:
        df = df.drop(invalid_indices).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
    else:
        # ensure consistent csv formatting (no index)
        df.to_csv(csv_path, index=False)
    print(f"[INFO] {len(df)} valid samples remain in {csv_path}.")
    return df

def combine_csvs(csv_paths, out_csv):
    """Combine multiple CSV files (same columns) into one and write out_csv."""
    print(f"[INFO] Combining {len(csv_paths)} into {out_csv}")
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        print("[WARN] No CSV files to combine.")
        return None
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"[INFO] Combined CSV written: {out_csv} ({len(combined)} rows)")
    return out_csv

def main(data_root="./ROCOv2_data", out_dir=".", validate=True, n_jobs=None):
    data_root = os.path.abspath(data_root)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    parts = {}
    for part in ("train", "validation", "test"):
        part_path = os.path.join(data_root, part)
        if os.path.isdir(part_path):
            out_csv = os.path.join(out_dir, f"{part}_roco.csv")
            convert_dir_to_csv(part_path, out_csv)
            if validate:
                remove_invalid(out_csv, n_jobs=n_jobs)
            parts[part] = out_csv

    # produce combined all_roco.csv from available parts
    if parts:
        all_csv = os.path.join(out_dir, "all_roco.csv")
        combine_csvs(list(parts.values()), all_csv)
    else:
        print(f"[WARN] No train/validation/test subdirs found under {data_root}. You can point data_root directly at a folder containing .txt files.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert ROCOv2 .txt captions to <name>_roco.csv files")
    parser.add_argument("--data-root", default="./ROCOv2_data", help="Root folder containing train/validation/test")
    parser.add_argument("--out-dir", default=".", help="Where to write CSVs")
    parser.add_argument("--no-validate", dest="validate", action="store_false", help="Skip image validation step")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of jobs for validation (default: min(8, cpu_count))")
    args = parser.parse_args()

    main(data_root=args.data_root, out_dir=args.out_dir, validate=args.validate, n_jobs=args.n_jobs)