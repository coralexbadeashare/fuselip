import json
import os

from PIL import Image

from config import MTCIR_PATH



class MTCIR:
    def __init__(self, txt_tokenizer, image_processor, do_tokenize=True):
        self.txt_tokenizer = txt_tokenizer
        self.image_processor = image_processor
        self.do_tokenize = do_tokenize

        with open(MTCIR_PATH) as f:
            self.data = [json.loads(line) for line in f]

        # TODO implement combined sampling

    def _process_image(self, image):
        image = self.image_processor(image) if image is not None else None
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        qry_img_name = sample["image"]
        target_img_name = sample["target_image"]
        modifications = sample["modifications"]

        # load images
        qry_img_path = os.path.join(os.path.dirname(MTCIR_PATH), qry_img_name)
        target_img_path = os.path.join(os.path.dirname(MTCIR_PATH), target_img_name)
        with Image.open(qry_img_path) as qry_img:
            qry_img = self._process_image(qry_img)
        with Image.open(target_img_path) as target_img:
            target_img = self._process_image(target_img)

        # build text
        qry_txt = " ".join(modifications)
        target_txt = ""
        if self.do_tokenize:
            qry_txt = self.txt_tokenizer(qry_txt).squeeze()
            target_txt = self.txt_tokenizer(target_txt).squeeze()

        return qry_txt, qry_img, target_txt, target_img
