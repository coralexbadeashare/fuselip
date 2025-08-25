import tarfile
import os

data_dir = "cc3m_data"
extract_dir = "cc3m_data/extracted"

os.makedirs(extract_dir, exist_ok=True)

for fname in os.listdir(data_dir):
    if fname.endswith(".tar"):
        tar_path = os.path.join(data_dir, fname)
        print(f"Extracting {tar_path} ...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir)

print("Extraction complete. All files are in:", extract_dir)