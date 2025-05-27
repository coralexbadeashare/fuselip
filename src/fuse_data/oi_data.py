import json
import os

from PIL import Image, ImageDraw

from config import OI_PATH, OI_POSITIONAL_DATA_PATH, OI_CROP_DATA_PATH


class OpenImages:
    def __init__(self, txt_tokenizer, image_processor, mask_target, mode="crop", do_tokenize=True):
        self.txt_tokenizer = txt_tokenizer
        self.image_processor = image_processor
        self.mode = mode

        if self.mode == "crop":
            data = json.load(open(OI_CROP_DATA_PATH))
            self.subset = data["meta"]["subset"]
            self.data = data["data"]
        elif self.mode == "pos":
            data = json.load(open(OI_POSITIONAL_DATA_PATH))
            self.subset = data["meta"]["subset"]
            # flatten {label: [img_id, bbox, query], .. } to [[img_id, bbox, query], ..]
            self.data = [el for lst in data["data"].values() for el in lst]
        elif self.mode == "vqa":
            raise NotImplementedError()
        else:
            raise ValueError

        # self.prompt = "crop to "
        self.prompt = ""
        self.mask_target = mask_target
        self.do_tokenize = do_tokenize

    def _process_image(self, image):
        image = self.image_processor(image) if image is not None else None
        return image

    def _crop_to_region(self, image, bbox):
        image = image.crop(
            (bbox["XMin"], bbox["YMin"],
             bbox["XMax"], bbox["YMax"])
        )
        return image

    def _get_img_path(self, img_id):
        return os.path.join(OI_PATH, self.subset, img_id)

    def _mask_target(self, image, bbox):
        # mask target bbox with black in image
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(
            (bbox["XMin"], bbox["YMin"], bbox["XMax"], bbox["YMax"]),
            fill=255
        )
        image.paste(0, mask=mask)
        return image

    def __len__(self):
        return len(self.data)

    def _getitem_vqa(self, item):
        sample = self.data[item]
        # get query image
        img_ref_path = self._get_img_path(sample["query_img"])
        img_ref = Image.open(img_ref_path).convert("RGB")
        # draw bbox
        bbox = sample["bbox"]
        x1, x2, y1, y2 = bbox
        draw = ImageDraw.Draw(img_ref)
        draw.rectangle(
            [x1, y1, x2, y2], outline="red", fill="red",
            width=10
        )
        # process
        img_ref = self._process_image(img_ref)
        # get query text
        query_txt = sample["query_txt"]
        # get retrieval texts
        ret_txts = sample["retrieval_txts"]
        if self.do_tokenize:
            query_txt = self.txt_tokenizer(query_txt).squeeze()
            ret_txts = [self.txt_tokenizer(txt).squeeze() for txt in ret_txts]
        retrieval_imgs = [None] * len(ret_txts)
        return query_txt, img_ref, ret_txts, retrieval_imgs

    def __getitem__(self, item):
        if self.mode == "vqa":
            return self._getitem_vqa(item)

        sample = self.data[item]
        # get query image
        img_ref_path = self._get_img_path(sample["query_img"])
        img_ref = Image.open(img_ref_path).convert("RGB")
        if self.mask_target:
            img_ref = self._mask_target(img_ref, sample["retrieval_samples"][0]["bbox"])
        # get retrieval pool
        retrieval_imgs = []
        for el in sample["retrieval_samples"]:
            img_path = self._get_img_path(el["ImageID"])
            img = Image.open(img_path).convert("RGB")
            # crop
            img = self._crop_to_region(img, el["bbox"])
            retrieval_imgs.append(img)
        # process images
        img_ref = self._process_image(img_ref)
        retrieval_imgs = [self._process_image(img) for img in retrieval_imgs]
        # get text
        query_text = self.prompt + sample["query_target"]
        retrieval_text = [""] * len(retrieval_imgs)
        if self.do_tokenize:
            query_text = self.txt_tokenizer(query_text).squeeze()
            retrieval_text = [self.txt_tokenizer(txt).squeeze() for txt in retrieval_text]

        return query_text, img_ref, retrieval_text, retrieval_imgs
