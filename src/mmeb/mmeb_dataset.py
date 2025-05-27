#  https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/src/dataset.py
import logging
from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from PIL import Image
import os

from tqdm import tqdm


class MMEBTrainDataset(Dataset):
    def __init__(self, data_args, txt_tokenizer, image_processor, txt_modifier_left, txt_modifier_right, do_tokenize_txt=True, cc3m=None):
        self.data_args = data_args
        self.txt_tokenizer = txt_tokenizer
        self.image_processor = image_processor
        self.txt_modifier_left = txt_modifier_left
        self.txt_modifier_right = txt_modifier_right
        self.do_tokenize_txt = do_tokenize_txt

        train_data = []
        logging.info(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")

        for subset in tqdm(data_args.subset_name):
            subset_data = load_dataset(
                self.data_args.dataset_name,
                subset,
                split=f"{self.data_args.dataset_split}[:{data_args.num_sample_per_subset}]",
            )
            train_data.append(subset_data)
        self.train_data = concatenate_datasets(train_data)

        self.cc3m = cc3m


    def __len__(self):
        n = len(self.train_data)
        if self.cc3m is not None:
            n += len(self.cc3m.captions)
        return n

    def _process_image(self, image):
        if image is None:
            return None
        image = self.image_processor(image)
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        return image

    def _get_subset(self, qry_img_path, pos_img_path):
        if qry_img_path == "":
            return pos_img_path.split("/")[1]
        return qry_img_path.split("/")[1]

    def __getitem__(self, item) -> Tuple:
        if item < len(self.train_data):
            # get mmeb sample
            qry, qry_image_path, pos_text, pos_image_path = (
                self.train_data[item]["qry"], self.train_data[item]["qry_image_path"],
                self.train_data[item]["pos_text"], self.train_data[item]["pos_image_path"],
            )

            subset = self._get_subset(qry_image_path, pos_image_path)
            qry = self.txt_modifier_left(qry, subset=subset)
            pos_text = self.txt_modifier_right(pos_text, subset=subset)
            qry_image = self._process_image(self._get_image(qry_image_path))
            pos_image = self._process_image(self._get_image(pos_image_path))
        else:
            # get cc3m sample
            item = item % (len(self.train_data) - 1)
            qry_image, pos_text = self.cc3m[item]
            qry = ""
            pos_image = None

        if self.do_tokenize_txt:
            qry, pos_text = self.txt_tokenizer(qry).squeeze(), self.txt_tokenizer(pos_text).squeeze()

        return (qry, qry_image, pos_text, pos_image)


class EvalDataset(Dataset):
    def __init__(self, data_args, subset, text_field, img_path_field, image_processor, txt_modifier):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.image_processor = image_processor
        self.txt_modifier = txt_modifier
        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        self.paired_data = self.get_paired_data(text_field, img_path_field, subset=subset)
        self.subset = subset

        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })


    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text = self.paired_dataset[item]["text"]
        img_path = self.paired_dataset[item]["img_path"]
        return self.txt_modifier(text, self.subset), self._get_image(img_path), img_path

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        image = self.image_processor(image)
        return image

    def get_paired_data(self, text_field, img_path_field, subset):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif isinstance(row[text_field], List):
                assert isinstance(row[img_path_field], List) and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        if subset == "ImageNet-1K":
            # for imagenet, some targets contain multiple words, separated by comma
            paired_data = [{"text": text.split(",")[0], "img_path": img_path} for text, img_path in unique_pair]
        else:
            paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]

        return paired_data