#!/usr/bin/env python3
"""Read a PyTorch checkpoint and print the length of positional embeddings.

Usage: get_checkpoint_context_len.py /path/to/checkpoint.pt

This script tries to be tolerant: it looks for keys containing 'positional' (case
insensitive) in the checkpoint's state_dict and prints the first tensor's first
dimension (typically the sequence length / context length). Exit code 0 on
success, non-zero on failure.
"""
import sys
import os

def main():
    try:
        import torch
    except Exception as e:
        print(f"ERROR: torch is required to inspect checkpoint: {e}", file=sys.stderr)
        return 2

    if len(sys.argv) < 2:
        print("Usage: get_checkpoint_context_len.py /path/to/checkpoint.pt", file=sys.stderr)
        return 2

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    try:
        ck = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"ERROR: failed to load checkpoint: {e}", file=sys.stderr)
        return 3

    # Some checkpoints store the model under 'state_dict' or 'model'
    if isinstance(ck, dict):
        # prefer common key names
        for k in ("state_dict", "model", "model_state_dict"):
            if k in ck and isinstance(ck[k], dict):
                sd = ck[k]
                break
        else:
            sd = ck
    else:
        print("ERROR: unexpected checkpoint format", file=sys.stderr)
        return 3

    # Search for a tensor whose key contains 'positional' (case-insensitive)
    import re
    pattern = re.compile("positional", flags=re.IGNORECASE)
    for k, v in sd.items():
        if pattern.search(k) and hasattr(v, 'shape'):
            # expect tensor with shape [L, D] or similar
            try:
                if len(v.shape) >= 1:
                    print(int(v.shape[0]))
                    return 0
            except Exception:
                continue

    # If not found, try some common names
    common_names = [
        'positional_embedding',
        'module.positional_embedding',
        'transformer.positional_embedding',
        'model.positional_embedding',
    ]
    for name in common_names:
        if name in sd:
            v = sd[name]
            try:
                print(int(v.shape[0]))
                return 0
            except Exception:
                pass

    # As a last resort, try to find any 2D tensor with reasonably large first dim (>50)
    for k, v in sd.items():
        try:
            if hasattr(v, 'shape') and len(v.shape) == 2 and v.shape[0] > 50:
                print(int(v.shape[0]))
                return 0
        except Exception:
            continue

    print("ERROR: could not determine positional embedding length in checkpoint", file=sys.stderr)
    return 4

if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
