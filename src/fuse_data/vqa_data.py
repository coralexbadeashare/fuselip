import os
from typing import Tuple

import pandas as pd
from PIL import Image

from config import CC3M_VQA_PATH, CC12M_VQA_PATH
from fuse_clip.fuse_clip_utils import set_seed


class VQAData:
    def __init__(self, txt_tokenizer, image_processor, source, do_tokenize=True):
        self.txt_tokenizer = txt_tokenizer
        self.image_processor = image_processor
        if source == "cc3m-vqa":
            self._load_generated_data(source)
        elif source == "cc12m-vqa":
            self._load_generated_data(source)
        self.do_tokenize = do_tokenize

    def _load_generated_data(self, source):
        if source == "cc3m-vqa":
            self.data_path = os.path.dirname(CC3M_VQA_PATH)
            df_path = CC3M_VQA_PATH
        elif source == "cc12m-vqa":
            self.data_path = os.path.dirname(CC12M_VQA_PATH)
            df_path = CC12M_VQA_PATH
        df = pd.read_csv(df_path, keep_default_na=False)
        assert not df.isna().any().any()
        self.images_left = df["path"].tolist()
        self.txt_left = df["question"].tolist()
        self.txt_right = df["answer"].tolist()


    def _process_image(self, image):
        image = self.image_processor(image)
        return image

    def __len__(self):
        return len(self.txt_left)

    def __getitem__(self, item) -> Tuple:
        image_left_path = os.path.join(self.data_path, self.images_left[item])
        with Image.open(image_left_path) as image_left:
            image_left = self._process_image(image_left)

        txt_left = self.txt_left[item]
        txt_right = self.txt_right[item]

        if self.do_tokenize:
            txt_left = self.txt_tokenizer(txt_left).squeeze()
            txt_right = self.txt_tokenizer(txt_right).squeeze()

        return txt_left, image_left, txt_right, None
