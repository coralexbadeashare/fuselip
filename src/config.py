import os.path

MACHINE = "VM"

if MACHINE == "VM":
    PROJECT_PATH = "/workspace/fuselip"
    DATA_DIR = "/workspace/fuselip/ROCOv2_data" #modify this as needed
else:
    raise ValueError("Unknown machine")


CC3M_TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
CC3M_VAL_CSV_PATH = os.path.join(DATA_DIR, "val.csv")
CC12M_TRAIN_CSV_PATH = os.path.join(DATA_DIR, "cc12m/cc12m.csv")
MMEB_TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "MMEB-train")
MMEB_VAL_IMAGE_DIR = os.path.join(DATA_DIR, "MMEB/eval_images")
OI_PATH = os.path.join(DATA_DIR, "openimages")
VG_DATA_PATH = os.path.join(DATA_DIR, "visual-genome")

OI_CROP_DATA_PATH = "./data/oi-crop.json"
OI_POSITIONAL_DATA_PATH = "./data/oi-crop-positional-v3.json"
CC3M_VQA_PATH = os.path.join(DATA_DIR, "cc3m/cc3m_vqa_train_v2.csv")
CC12M_VQA_PATH = os.path.join(DATA_DIR, "cc12m/cc12m_vqa_train_v2.csv")
IMAGENET_VAL_PATH = os.path.join(DATA_DIR, "imagenet/val")

RES_PATH = os.path.join(PROJECT_PATH, "results")
RES_MMEB_PATH = os.path.join(RES_PATH, "mmeb-res.json")
RES_SUGARCREPE_PATH = os.path.join(RES_PATH, "sugarcrepe.json")
LOG_PATH = os.path.join(PROJECT_PATH, "logs")