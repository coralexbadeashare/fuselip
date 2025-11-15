import tarfile
import os
import concurrent.futures

src_dir = "./cc3m_data/datasets--pixparse--cc3m-wds/snapshots/46f3d69f840e59d77d52e8decfe5baec97e94c7f"
dst_dir = "./cc3m_data/cc3m_extracted"
os.makedirs(dst_dir, exist_ok=True)

def extract_tar(tar_path):
    print(f"Extracting {os.path.basename(tar_path)}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dst_dir)

tars = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".tar")]

# Parallel extraction (tune max_workers based on your CPU)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    ex.map(extract_tar, tars)

print("✅ Done extracting all shards!")
