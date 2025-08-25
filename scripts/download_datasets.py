#!/usr/bin/env python3
"""Download CC3M/CC12M webdataset shards from HuggingFace into a local folder.

Usage examples:
  # list shards (no download)
  python scripts/download_datasets.py --datasets cc3m --list --max-files 10

  # download up to 50 shards into ./cc3m_data
  python scripts/download_datasets.py --datasets cc3m --outdir ./cc3m_data --max-files 50

This script uses the huggingface_hub API. Set HF_TOKEN env var if you need to access gated repos.
"""
import argparse
import os
from huggingface_hub import list_repo_files, hf_hub_download


REPO_MAP = {
    "cc3m": "pixparse/cc3m-wds",
    "cc12m": "pixparse/cc12m-wds",
}


def list_shards(repo_id, ext=".tar"):
    files = list_repo_files(repo_id, repo_type="dataset")
    shards = [f for f in files if f.endswith(ext)]
    return shards


def download_shards(repo_id, outdir, max_files=None, ext=".tar", force_download=False):
    os.makedirs(outdir, exist_ok=True)
    shards = list_shards(repo_id, ext=ext)
    if max_files:
        shards = shards[:max_files]
    downloaded = []
    for fn in shards:
        print(f"Downloading {fn} ...")
        # save into outdir by setting cache_dir to outdir and force_filename to the basename
        local_path = hf_hub_download(repo_id=repo_id, filename=fn, repo_type="dataset",
                                     cache_dir=outdir, force_filename=os.path.basename(fn), force_download=force_download)
        downloaded.append(local_path)
    return downloaded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs='+', choices=list(REPO_MAP.keys()), required=True,
                   help="Which datasets to download (cc3m, cc12m)")
    p.add_argument("--outdir", default="./cc3m_data", help="Local output directory")
    p.add_argument("--max-files", type=int, default=None, help="Limit number of shards (useful for testing)")
    p.add_argument("--list", action="store_true", help="Only list shard filenames, don't download")
    p.add_argument("--force", action="store_true", help="Force re-download even if cached")
    args = p.parse_args()

    for ds in args.datasets:
        repo = REPO_MAP[ds]
        print(f"Dataset {ds} -> {repo}")
        shards = list_shards(repo)
        print(f"Found {len(shards)} shard files (.{shards[0].split('.')[-1]}), showing up to 20:")
        for s in shards[:20]:
            print("  ", s)
        if args.list:
            continue

        outdir = os.path.join(args.outdir, ds) if len(args.datasets) > 1 else args.outdir
        print(f"Downloading to {outdir} (max_files={args.max_files})")
        downloaded = download_shards(repo, outdir, max_files=args.max_files, force_download=args.force)
        print(f"Downloaded {len(downloaded)} files to {outdir}")


if __name__ == "__main__":
    main()
