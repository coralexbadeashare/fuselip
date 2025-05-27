from dataclasses import dataclass, field
from typing import List, Union

import torch

from config import MMEB_TRAIN_IMAGE_DIR, MMEB_VAL_IMAGE_DIR

MMEB_TRAIN_SUBSETS = ['ImageNet_1K',
                      'N24News', 'HatefulMemes', 'VOC2007', 'SUN397',
                      'OK-VQA', 'A-OKVQA', 'DocVQA', 'InfographicsVQA', 'ChartQA', 'Visual7W',
                      'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 'MSCOCO_i2t', 'NIGHTS', 'WebQA',
                      'MSCOCO'
                      ]
MMEB_EVAL_SUBSETS = ['ImageNet-1K', 'CIFAR-100', 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397', 'Place365', 'ImageNet-A',
                     'ImageNet-R', 'ObjectNet', 'Country211', 'OK-VQA', 'A-OKVQA', 'DocVQA', 'InfographicsVQA',
                     'ChartQA', 'Visual7W', 'ScienceQA', 'VizWiz', 'GQA', 'TextVQA', 'VisDial', 'CIRR',
                     'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 'MSCOCO_i2t', 'NIGHTS', 'WebQA',
                     'FashionIQ', 'Wiki-SS-NQ', 'OVEN', 'EDIS', 'MSCOCO', 'RefCOCO', 'RefCOCO-Matching',
                     'Visual7W-Pointing']
MMEB_EVAL_SHORT_SUBSETS = ['ImageNet-1K', 'OK-VQA', 'ScienceQA', 'VisDial', 'OVEN', 'MSCOCO', 'RefCOCO-Matching']

# MMEB_I_T_SUBSETS = ['ImageNet-1K', 'CIFAR-100', 'VOC2007', 'SUN397', 'Place365', 'ImageNet-A',
#                      'ImageNet-R', 'ObjectNet', 'Country211']
MMEB_IT_T_SUBSETS = ['N24News','HatefulMemes', 'OK-VQA', 'A-OKVQA', 'DocVQA', 'InfographicsVQA',
                     'ChartQA', 'Visual7W', 'ScienceQA', 'VizWiz', 'GQA', 'TextVQA', 'WebQA', 'EDIS']
MMEB_IT_I_SUBSETS = ['CIRR', 'FashionIQ', 'MSCOCO', 'RefCOCO', 'Visual7W-Pointing']

MMEB_CLASSIFICATION_DATASETS = ['ImageNet-1K', 'CIFAR-100', 'N24News', 'HatefulMemes', 'VOC2007',
                                'SUN397', 'Place365', 'ImageNet-A', 'ImageNet-R', 'ObjectNet', 'Country211']
MMEB_VQA_DATASETS = ['OK-VQA', 'A-OKVQA', 'DocVQA', 'InfographicsVQA', 'ChartQA',
                     'Visual7W', 'ScienceQA', 'VizWiz', 'GQA', 'TextVQA']

HATEFUL_MEMES_PROMPT = "Is this image hateful?"


REPLACEMENTS_TRAIN = {  # we don't use these atm, we use no prompt at all
    "<|image_1|>\nRepresent the given image with the following question:": "VQA:",
    "<|image_1|>\nRepresent the given image for classification": "Classification",
    "<|image_1|>\nRepresent the given news image with the following caption for domain classification:": "Domain classification:",
    "<|image_1|>\nRepresent the given image for binary classification to determine whether it constitutes hateful speech or not": HATEFUL_MEMES_PROMPT,
    "Represent the given dialogue about an image, which is used for image retrieval:": "Image retrieval:",
    "<|image_1|>\nFind an image to match the fashion image and style note:": "Image retrieval:",
    "<|image_1|>\nRepresent the given image.": "",
    "<|image_1|>\nRepresent the given image": "",
    "<|image_1|>\nGiven an image, find a similar everyday image with the described changes:": "Change:",
    "<|image_1|>\nFind a day-to-day image that looks similar to the provided image.": "Similar image",
    "<|image_1|>\nFind a Wikipedia image that answers this question:": "Image retrieval:",
    "<|image_1|>\nRepresent the given Wikipedia image with related text information: ": "",
    "<|image_1|>\nSelect the portion of the image that isolates the object labeled as": "Object isolation:",
    "<|image_1|>\nRepresent the given cropped image of the object": "Object isolation",
    "<|image_1|>\nFind an image caption describing the given everyday image.": "Captioning",
    "Find me an everyday image that matches the given caption: ": "Image retrieval: ",
    "<|image_1|>\nIdentify the scene shown in the image": "Scene identification",
    "<|image_1|>\nIdentify the object shown in the image": "Object identification",
    "<|image_1|>\nFind a caption for the news in the given photo.": "Captioning",
    "Retrieve an image of this news caption.": "Image retrieval:",
}

REPLACEMENTS_EVAL = {  # we don't use these atm, we use no prompt at all
    "<|image_1|>\nIdentify the country depicted in the image": "In which country is this?",
    "<|image_1|>\nFind a news image that matches the provided caption:": "Image retrieval:",
    "<|image_1|>\nCrop the image to to isolate the object labeled as": "Object isolation:",
    "<|image_1|>\nRetrieve a Wikipedia image-description pair that provides evidence for the question of this image:": "Image retrieval:",
    "<|image_1|>\nSelect the portion of the image that follows the language expressions.": "Object isolation:",
    "<|image_1|>\nSelect the portion of the image that follows the language expressions:": "Object isolation:",
    "<|image_1|>\nSelect the portion of the image that answers the question": "Object isolation:",
    "Find the document image that can answer the given query:": "Image retrieval:"

}
REPLACEMENTS_EVAL.update(REPLACEMENTS_TRAIN)

PREFIXES_RIGHT = {  # we do use these
    'ImageNet-1K': "a photo of a ",
    'ImageNet_1K': "a photo of a ",
    'CIFAR-100': "a photo of a ",
    'CIFAR_100': "a photo of a ",
    'N24News': "a photo from the domain of ",
    'VOC2007': "a photo of a ",
    'SUN397': "a photo of a ",
    'Place365': "a photo of a ",
    'ImageNet-A': "a photo of a ",
    'ImageNet-R': "a photo of a ",
    'ObjectNet': "a photo of a ",
    'Country211': "this is in the country of ",
}


class TxtModifier:
    def __init__(
        self, model_name: str, is_train: bool, empty_prompt: Union[bool, List[str]] = False,
        drop_text: bool = False,
        prompt_prefixes: dict = None
    ):
        self.replacements = REPLACEMENTS_TRAIN.copy() if is_train else REPLACEMENTS_EVAL.copy()
        self._set_replacements(empty_prompt)

        self.drop_text = drop_text
        self.prompt_prefixes = prompt_prefixes.copy() if prompt_prefixes is not None else None

    def _set_replacements(self, empty_prompt: Union[bool, List[str]] = False) -> dict:
        replacements = {}
        if isinstance(empty_prompt, bool) and empty_prompt:
            replacements = {
                k: "" for k, v in self.replacements.items() if v != HATEFUL_MEMES_PROMPT
            }
        elif isinstance(empty_prompt, list):
            replacements = {k: "" for k, v in self.replacements.items() if v in empty_prompt}

        self.replacements.update(replacements)


    def __call__(self, txt: str, subset: str) -> str:
        if self.drop_text:
            return ""

        for k, v in self.replacements.items():
            txt = txt.replace(k, v)
        if subset in ["ImageNet-1K", "ImageNet_1K"]:
            txt = txt.split(",")[0]
        if self.prompt_prefixes is not None:
            prefix = self.prompt_prefixes.get(subset, "")
            txt = f"{prefix}{txt}"
        return txt.rstrip("\n").strip()


def mmeb_eval_collator(batch: dict) -> dict:
    txts = []
    imgs = []
    img_paths = []

    for sample in batch:
        txts.append(sample[0])
        img = sample[1]
        if img is None:
            img = torch.empty(0, 3, 1, 1)
        imgs.append(img)
        img_paths.append(sample[2])

    return {
        "txts": txts,
        "imgs": torch.stack(imgs),
        "img_paths": img_paths,
    }


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="TIGER-Lab/MMEB-eval", metadata={"help": "huggingface dataset name"}
    )
    subset_name: List[str] = field(
        default_factory=lambda: MMEB_EVAL_SUBSETS,
        metadata={"help": "Useful for datasets with subsets"}
    )
    dataset_split: str = field(
        default='test', metadata={"help": "dataset split"}
    )
    num_sample_per_subset: int = field(
        default=1000, metadata={"help": "number of training samples per subset"}
    )
    image_dir: str = field(
        default=MMEB_VAL_IMAGE_DIR, metadata={"help": "Image directory path"}
    )
    encode_output_path: str = field(
        default=None, metadata={"help": "encode output path"}
    )


@dataclass
class TrainDataArguments:
    dataset_name = "TIGER-Lab/MMEB-train"
    subset_name = MMEB_TRAIN_SUBSETS
    dataset_split = "original"
    num_sample_per_subset = 50000
    image_dir: str = MMEB_TRAIN_IMAGE_DIR
