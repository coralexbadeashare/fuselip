import json
import os
from typing import Tuple

from PIL import Image

from config import VG_DATA_PATH
from fuse_clip.fuse_clip_utils import compute_iou


class VisualGenome:
    def __init__(
        self, txt_tokenizer, image_processor, mode, is_train, do_tokenize=True, iou_thresh=None, prune_small=False
    ):
        # mode: "crop" or "vqa"
        data_path = VG_DATA_PATH

        self.regions = None
        self.vqa = None
        if mode in ["crop", "position"]:
            regions_file = "region_descriptions_train.json" if is_train else "region_descriptions_test.json"
            regions = json.load(open(os.path.join(data_path, regions_file)))
            if prune_small:
                regions = self.prune_small_regions(regions, min_pixels=32)
            if iou_thresh is not None:
                regions = self.get_non_overlapping_regions(regions, iou_thresh)
            self.regions_per_image = regions
            # flatten s.t. each element is a region
            self.regions = [r for img_id in regions for r in img_id["regions"]]
            # group into groups of 4
            self.combined_items = [list(range(i, i+4)) for i in range(0, len(self.regions), 4)]
            self.combined_items = self.combined_items[:len(self.regions) // 4]
        if mode == "vqa":
            vqa = json.load(open(os.path.join(data_path, "question_answers.json")))
            # flatten s.t. each element is a question
            self.vqa = [qa for img_id in vqa for qa in img_id["qas"]]
            # group into groups of 4
            self.combined_items = [list(range(i, i + 4)) for i in range(0, len(self.vqa), 4)]
            self.combined_items = self.combined_items[:len(self.vqa) // 4]

        self.txt_tokenizer = txt_tokenizer
        self.image_processor = image_processor
        self.mode = mode
        self.do_tokenize = do_tokenize

    def get_non_overlapping_regions(self, regions_per_image, iou_thresh):
        # for each image, get a set of non-overlapping regions
        regions_per_image_pruned = []
        for regions in regions_per_image:
            id = regions["id"]
            regions = regions["regions"]
            order = range(len(regions))
            keep = []
            suppressed = set()
            for i in order:
                if i in suppressed:
                    continue
                keep.append(i)
                # Compare this box with the rest
                for j in order:
                    if j in suppressed or j == i:
                        continue
                    iou_val = compute_iou(regions[i], regions[j])
                    if iou_val > iou_thresh:
                        suppressed.add(j)
            regions = [regions[i] for i in keep]
            regions_per_image_pruned.append({"id": id, "regions": regions})
        return regions_per_image_pruned

    def prune_small_regions(self, regions_per_image, min_pixels):
        regions_per_image_pruned = []
        for regions in regions_per_image:
            id = regions["id"]
            regions = regions["regions"]
            regions = [region for region in regions if region["width"] > min_pixels and region["height"] > min_pixels]
            regions_per_image_pruned.append({"id": id, "regions": regions})
        return regions_per_image_pruned

    def _process_image(self, image):
        image = self.image_processor(image) if image is not None else None
        return image

    def _crop_to_region(self, image, region):
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        return image.crop((x, y, x+w, y+h))

    def _determine_position(self, image, region):
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        # get center of region
        x_center = x + w // 2
        y_center = y + h // 2
        # get text description
        if 0 <= y_center < image.height // 3:
            y_str = "upper"
        elif image.height // 3 <= y_center < 2 * image.height // 3:
            y_str = "center"
        else:
            y_str = "lower"
        if 0 <= x_center < image.width // 3:
            x_str = "left"
        elif image.width // 3 <= x_center < 2 * image.width // 3:
            x_str = "center"
        else:
            x_str = "right"
        return f"{y_str} {x_str}"


    def __len__(self):
        if self.mode in ["crop", "position"]:
            return len(self.regions)
        elif self.mode == "vqa":
            return len(self.vqa)
        else:
            raise ValueError

    def _get_img_path(self, item):
        if self.mode in ["crop", "position"]:
            img_id = self.regions[item]["image_id"]
        elif self.mode == "vqa":
            img_id = self.vqa[item]["image_id"]
        else:
            raise ValueError
        return os.path.join(VG_DATA_PATH, "images", f"{img_id}.jpg")

    def __getitem__(self, item) -> Tuple:
        img_path = self._get_img_path(item)
        with Image.open(img_path) as image_left:
            if self.mode == "crop":
                region = self.regions[item]
                image_right = self._crop_to_region(image_left, region)
                txt_left = region["phrase"]
                txt_right = ""
            elif self.mode == "position":
                region = self.regions[item]
                txt_left = f"Where in the image can this information be found? {region['phrase']}"
                txt_right = self._determine_position(image_left, region)
                image_right = None
            elif self.mode == "vqa":
                qa = self.vqa[item]
                txt_left = qa["question"]
                txt_right = qa["answer"].rstrip(".")  # remove trailing dot
                image_right = None
            else:
                raise ValueError

            # process images and text
            image_left = self._process_image(image_left)
            image_right = self._process_image(image_right)
            if self.do_tokenize:
                txt_left = self.txt_tokenizer(txt_left).squeeze()
                txt_right = self.txt_tokenizer(txt_right).squeeze()

        return txt_left, image_left, txt_right, image_right
