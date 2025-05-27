import os
import warnings
from joblib import Parallel, delayed


import pandas as pd
from PIL import Image
from tqdm import tqdm


def convert(data_path, csv_path):
    # Use a list to collect data rows
    data = []

    # Get a set of all image filenames in the directory to check existence more efficiently
    image_filenames = set(os.listdir(data_path))

    not_found = []
    for fname in tqdm(os.listdir(data_path)):
        if fname.endswith(".txt"):
            img_path = os.path.join(data_path, fname.replace(".txt", ".jpg"))
            if fname.replace(".txt", ".jpg") not in image_filenames:
                # print(f"Image {img_path} not found")
                not_found.append(fname)
                continue

            # Read the caption
            with open(os.path.join(data_path, fname), "r") as f:
                caption = f.read()

            # Append data to the list
            img_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path))
            data.append([img_path, caption])

    # Create DataFrame from the list once
    df = pd.DataFrame(data, columns=["path", "caption"])
    df.to_csv(csv_path, index=False)

    print(f"Images not found: {len(not_found)}")
    print(not_found[:5])
    print(f"Converted {len(df)} samples to CSV")
    print("Done")


def _check_image_validity(idx, path):
    """
    Check whether an image is valid:
      1) The file must exist.
      2) Pillow verify() returns no exception/warning.
      3) A full decode with load() must succeed.
      4) The image size is non-zero.
    Returns a tuple: (idx, is_valid)
    """
    if not os.path.exists(path):
        return (idx, False)

    # 1) verify() may detect header issues
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            img = Image.open(path)
            img.verify()  # Closes the file handle internally
            # If any warnings were raised, consider it invalid
            if w:
                return (idx, False)
        except Exception:
            return (idx, False)

    # 2) Re-open to force a full decode
    try:
        with Image.open(path) as img:
            img.load()  # Force reading all pixel data
            if img.size[0] == 0 or img.size[1] == 0:
                return (idx, False)
    except Exception:
        return (idx, False)

    return (idx, True)


def remove_invalid(csv_path, n_jobs=8):
    """
    Reads a CSV file containing a 'path' column, checks which images are valid,
    drops invalid entries, and writes a new CSV: <csv_path>_valid.csv
    """
    df = pd.read_csv(csv_path)
    print(f"Read {len(df)} samples")

    # Convert to list for parallel iteration
    paths = df["path"].tolist()
    paths = [os.path.join(os.path.dirname(csv_path), p) for p in paths]

    # Parallel check for each row
    # Note: tqdm will only show overall progress of joblib tasks,
    # not a live-per-sample update.
    results = Parallel(n_jobs=n_jobs)(
        delayed(_check_image_validity)(i, p)
        for i, p in tqdm(enumerate(paths), total=len(paths), desc="Validating images")
    )

    # Gather invalid indices
    invalid_indices = [idx for (idx, valid) in results if not valid]

    len_orig = len(df)
    df.drop(invalid_indices, inplace=True)
    print(f"Removed {len(invalid_indices)} invalid or warned samples")
    print(f"Removed total: {len_orig - len(df)}")
    print(f"Remaining samples: {len(df)}")

    out_path = csv_path.replace(".csv", "_valid.csv")
    df.to_csv(out_path, index=False)
    return df



if __name__ == '__main__':
    data_path = "/path/to/cc3m/train"
    csv_path = "/path/to/cc3m/train.csv"

    # data_path = "/path/to/cc3m/val"
    # csv_path = "/path/to/cc3m/val.csv"

    # data_path = "/path/to/cc12m/images"
    # csv_path = "/path/to/cc12m/cc12m.csv"

    convert(data_path, csv_path)
    remove_invalid(csv_path, n_jobs=8)

    print("Done")

