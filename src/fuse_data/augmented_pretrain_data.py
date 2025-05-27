import logging
import os
import random
import textwrap

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop

from config import CC3M_TRAIN_CSV_PATH
from open_clip.transform import _convert_to_rgb

LEFT_TEXT_IDENTITY = ""
_CROP_LOCATIONS = [
        "upper left", "upper center", "upper right",
        "center left", "center center", "center right",
        "lower left", "lower center", "lower right"
    ]
_ROTATION_ANGLES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
_ROTATION_DIRECTIONS = ["clockwise", "counter-clockwise"]
_ROTATION_ANGLES_DIRECTIONS = [(angle, direction) for angle in _ROTATION_ANGLES for direction in _ROTATION_DIRECTIONS]

def _identity(img, caption, **kwargs):
    return None, LEFT_TEXT_IDENTITY

    if len(caption) < 20:
        return None, "Classification"
    else:
        return None, "Captioning"

def _centercrop(img, **kwargs):
    orig_size = img.size
    img = F.center_crop(img, orig_size[0]//2)
    img = F.resize(img, orig_size, interpolation=F.InterpolationMode.BICUBIC)
    return img, "center crop"

def _randomcrop(img, imgsize, crop_location=None, **kwargs):
    orig_size = img.size
    # todo could make random, have to adapt positions then
    cropped_size = min(orig_size[0]//2, 256, orig_size[0], orig_size[1])
    top_str_to_pos = {"upper": 0, "center": (orig_size[1]-cropped_size)//2, "lower": orig_size[1]-cropped_size}
    left_str_to_pos = {"left": 0, "center": (orig_size[0]-cropped_size)//2, "right": orig_size[0]-cropped_size}

    crop_location = random.choice(_CROP_LOCATIONS) if crop_location is None else crop_location
    top_str, left_str = crop_location.split()
    top, left = top_str_to_pos[top_str], left_str_to_pos[left_str]
    img = F.resized_crop(img, top, left, cropped_size, cropped_size, imgsize, interpolation=F.InterpolationMode.BICUBIC)

    return img, "crop to " + crop_location

def _horizontalflip(img, **kwargs):
    img = F.hflip(img)
    return img, "horizontal flip"

def _verticalflip(img, **kwargs):
    img = F.vflip(img)
    return img, "vertical flip"

def _rotate(img, rotation_angle=None, rotation_direction=None, **kwargs):
    angle = random.choice(_ROTATION_ANGLES) if rotation_angle is None else rotation_angle
    direction = random.choice(_ROTATION_DIRECTIONS) if rotation_direction is None else rotation_direction
    assert direction in _ROTATION_DIRECTIONS
    if direction == "clockwise":
        angle = -angle
    img = F.rotate(img, angle)
    return img, f"rotate {np.abs(angle)} degrees {direction}"

def _colorjitter(img, **kwargs):
    img = transforms.ColorJitter(
        brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2
    )(img)
    return img, f"color jittered"

def _colorjitterv2(img, **kwargs):
    txt = ""
    for t in [F.adjust_brightness, F.adjust_contrast, F.adjust_saturation]:
        factor = round(random.uniform(0.3, 2.0), 1)
        if factor == 1:
            factor = round(factor + 0.1, 1)
        prefix = "increase" if factor > 1 else "decrease"
        txt += f"{prefix} {t.__name__.split('_')[-1]} by factor {factor} and "
        img = t(img, factor)
    txt = txt[:-len(" and ")]
    return img, txt

def _grayscale(img, **kwargs):
    img = F.to_grayscale(img, num_output_channels=3)
    return img, "grayscale"

def _colorize(img, **kwargs):
    img = F.to_grayscale(img, num_output_channels=3)
    return img, "colorize"

def _copy_text(text):
    if isinstance(text, torch.Tensor):
        text = text.clone()
    elif isinstance(text, str):
        text = text
    else:
        raise ValueError(f"Text type {type(text)} not supported.")
    return text


_AUG_WEIGHTS = {  # will be normalized
    "randomcrop": 2,
    "rotate": 1,
    "colorjitterv2": 1,
    "horizontalflip": 1,
    "verticalflip": 1,
    "grayscale": 1,
    "colorize": 1,
    "flip": 2,
    "colorize_grayscale": 1,
}


class AugmentedCsvDataset(Dataset):
    def __init__(self, input_filename, preprocess, img_key, caption_key, unimodal_prob, sep=",", tokenizer=None,
                 augmentations=None, eval_mode=False, multiaug=True, unpad_image=False, do_tokenize=True,
                 crop_from_large_resolution=False,
                 is_master=True):
        # multiaug: return several augmentations for each left image,
        # so that we can evaluate the model on the same left image with different augmentations
        # eval_mode: deterministic crop (is it necessary?)
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()

        # get rid of normalize and totensor transform
        normalize = lambda x: x
        if (preprocess is not None):
            if isinstance(preprocess.transforms[-1], transforms.Normalize):
                normalize = preprocess.transforms[-1]
                preprocess = transforms.Compose(preprocess.transforms[:-1])
            if isinstance(preprocess.transforms[-1], transforms.ToTensor):
                preprocess = transforms.Compose(preprocess.transforms[:-1])
            # make sure there are no normalization and to tensor transforms left
            for t in preprocess.transforms:
                assert not isinstance(t, (transforms.Normalize, transforms.ToTensor))

        self.preprocess = preprocess
        self.maybe_to_tensor = lambda x: normalize(transforms.ToTensor()(x)) if x is not None else None
        if preprocess is not None:
            self.img_size = preprocess.transforms[0].size
        else:
            print("WARNING: preprocess is None, using default img size 256")
            self.img_size = (256, 256)


        self.tokenize = lambda x: tokenizer(x).squeeze() if do_tokenize else x
        self.data_path = os.path.dirname(input_filename)

        self.aug_fn_dict = {
            "identity": self.get_id_augs,
            "randomcrop": self.get_crop_augs,
            "rotate": self.get_rotate_augs,
            "colorjitterv2": self.get_colorjitter_augs,
            "flip": self.get_flip_augs,
            "colorize_grayscale": self.get_colorize_grayscale_augs,
        }

        if augmentations is None:
            augmentations = ["randomcrop", "rotate", "colorjitterv2",
                             "flip", "colorize_grayscale"]
        self.augmentations = augmentations

        self.eval_mode = eval_mode
        self.multiaug = multiaug
        self.unpad_image = unpad_image
        self.crop_from_large_resolution = crop_from_large_resolution

        if multiaug and not self.eval_mode:
            self.n_augs = {  # number of augmentations returned per sample for each augmentation function
                self.get_crop_augs: 9,
                self.get_rotate_augs: 3,
                self.get_colorjitter_augs: 3,
                self.get_flip_augs: 4,
                self.get_colorize_grayscale_augs: 2,
            }
        elif multiaug and self.eval_mode:
            self.n_augs = {
                self.get_crop_augs: 9,
                self.get_rotate_augs: 18,
                self.get_colorjitter_augs: 10,
                self.get_flip_augs: 4,
                self.get_colorize_grayscale_augs: 2,
            }
        elif not multiaug:
            assert not self.eval_mode
            self.n_augs = {
                self.get_crop_augs: 1,
                self.get_rotate_augs: 1,
                self.get_colorjitter_augs: 1,
                self.get_flip_augs: 1,
                self.get_colorize_grayscale_augs: 1,
            }

        weights = [_AUG_WEIGHTS[a] / self.n_augs[self.aug_fn_dict[a]] for a in self.augmentations]
        self.weights = np.array(weights) / np.sum(weights)
        # convert prob, so that unimodal_prob fraction of samples are not augmented
        avg_n_augmented = np.average(
            [self.n_augs[self.aug_fn_dict[a]] for a in self.augmentations],
            weights=self.weights
            )
        self.prob = unimodal_prob * avg_n_augmented / (1 + unimodal_prob * (avg_n_augmented - 1))

        if is_master:
            logging.info(f'Unimodal probability: {unimodal_prob} (converted to {self.prob})')
            logging.info(f"Augmentations: {', '.join(self.augmentations)}")
            logging.info(f'left-text-identity: {LEFT_TEXT_IDENTITY}')

        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def _get_image(self, idx, preprocessor):
        img_path = os.path.join(self.data_path, str(self.images[idx]))
        with Image.open(img_path) as image:
            if self.unpad_image:
                print("WARNING: unpadding image")
                # unpad image (cc3m-val images are padded)
                img_np = np.array(image)
                mask = (img_np <= [242, 242, 242]).any(axis=-1)
                coords = np.argwhere(mask)
                if coords.size > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    image = image.crop((x_min+5, y_min+5, x_max-5, y_max-5))
            if preprocessor is not None:
                image = preprocessor(image)
            else:
                image = image.copy()

        return image

    def _preprocess_crop(self, img):
        if self.crop_from_large_resolution:
            # crop, s.t. image is square (min_size x min_size)
            min_size = min(img.size)
            if self.eval_mode:
                img = F.center_crop(img, min_size)  # deterministic crop
            else:
                img = RandomCrop(min_size)(img)  # random crop
        else:
            # resize to square
            img = F.resize(img, self.img_size, interpolation=F.InterpolationMode.BICUBIC)
        img = _convert_to_rgb(img)
        return img

    def _get_aug_and_prepocessor(self):
        if random.random() < self.prob:
            aug = "identity"
        else:
            aug = np.random.choice(self.augmentations, p=self.weights)
        aug = self.aug_fn_dict[aug]
        if aug in [_randomcrop, self.get_crop_augs]:
            preprocess = self._preprocess_crop
        else:
            preprocess = self.preprocess
        return aug, preprocess

    def get_id_augs(self, image, caption):
        # identity
        image_right, text_left = _identity(image, caption=caption)
        text_left = self.tokenize(text_left)
        text_right = self.tokenize(caption)
        return [text_left], [self.maybe_to_tensor(image)], [text_right], [self.maybe_to_tensor(image_right)]


    def get_crop_augs(self, image, caption):
        crop_locations = np.random.permutation(_CROP_LOCATIONS.copy())
        text_right = self.tokenize("")
        images_left, images_right = [], []
        texts_left, texts_right = [], []
        for i in range(self.n_augs[self.get_crop_augs]):
            crop_location = crop_locations[i]
            image_right, text_left = _randomcrop(
                image, caption=caption, imgsize=self.img_size, crop_location=crop_location,
            )
            text_left = self.tokenize(text_left)
            images_right.append(self.maybe_to_tensor(image_right))
            texts_left.append(text_left)
            texts_right.append(_copy_text(text_right))
        if image.size != self.img_size:
            # since we might crop from larger resolution, now we have to resize
            image = F.resize(image, self.img_size, interpolation=F.InterpolationMode.BICUBIC)
        image = self.maybe_to_tensor(image)
        images_left = [image.clone() for _ in range(self.n_augs[self.get_crop_augs])]
        return texts_left, images_left, texts_right, images_right

    def get_rotate_augs(self, image, caption):
        rotation_angles_directions = np.random.permutation(_ROTATION_ANGLES_DIRECTIONS.copy())
        text_right = self.tokenize("")
        images_left, images_right = [], []
        texts_left, texts_right = [], []
        for i in range(self.n_augs[self.get_rotate_augs]):
            rotation_angle, rotation_direction = rotation_angles_directions[i]
            image_right, text_left = _rotate(
                image, caption=caption, imgsize=self.img_size,
                rotation_angle=int(rotation_angle), rotation_direction=rotation_direction
            )
            text_left = self.tokenize(text_left)
            images_right.append(self.maybe_to_tensor(image_right))
            texts_left.append(text_left)
            texts_right.append(_copy_text(text_right))
        image = self.maybe_to_tensor(image)
        images_left = [image.clone() for _ in range(self.n_augs[self.get_rotate_augs])]
        return texts_left, images_left, texts_right, images_right

    def get_colorjitter_augs(self, image, caption):
        text_right = self.tokenize("")
        images_left, images_right = [], []
        texts_left, texts_right = [], []
        for i in range(self.n_augs[self.get_colorjitter_augs]):
            image_right, text_left = _colorjitterv2(
                image, caption=caption, imgsize=self.img_size,
            )
            text_left = self.tokenize(text_left)
            images_right.append(self.maybe_to_tensor(image_right))
            texts_left.append(text_left)
            texts_right.append(_copy_text(text_right))
        image = self.maybe_to_tensor(image)
        images_left = [image.clone() for _ in range(self.n_augs[self.get_colorjitter_augs])]
        return texts_left, images_left, texts_right, images_right

    def get_flip_augs(self, image, caption):
        text_right = self.tokenize("")
        images_left, images_right = [], []
        texts_left, texts_right = [], []
        for i, aug_fn in enumerate(np.random.permutation([_horizontalflip, _verticalflip])):
            if i >= self.n_augs[self.get_flip_augs]:
                break
            image_right, text_left = aug_fn(image, caption=caption, imgsize=self.img_size)
            text_left = self.tokenize(text_left)
            images_right.append(self.maybe_to_tensor(image_right))
            texts_left.append(_copy_text(text_left))
            texts_right.append(_copy_text(text_right))
            images_left.append(self.maybe_to_tensor(image))
            if self.multiaug:
                # we put also the other direction into the batch
                images_left.append(self.maybe_to_tensor(image_right))
                images_right.append(self.maybe_to_tensor(image))
                texts_left.append(_copy_text(text_left))
                texts_right.append(_copy_text(text_right))

        return texts_left, images_left, texts_right, images_right

    def get_colorize_grayscale_augs(self, image, caption):
        text_right = self.tokenize("")
        images_left, images_right = [], []
        texts_left, texts_right = [], []
        for i, aug_fn in enumerate(np.random.permutation([_colorize, _grayscale])):
            if i >= self.n_augs[self.get_colorize_grayscale_augs]:
                break
            image_right, text_left = aug_fn(image, caption=caption, imgsize=self.img_size)
            if aug_fn == _colorize:
                # swap image_left and image_right
                image, image_right = image_right, image
            text_left = self.tokenize(text_left)
            images_left.append(self.maybe_to_tensor(image))
            images_right.append(self.maybe_to_tensor(image_right))
            texts_left.append(_copy_text(text_left))
            texts_right.append(_copy_text(text_right))
        return texts_left, images_left, texts_right, images_right


    def __getitem__(self, idx):
        aug_fn, preprocess = self._get_aug_and_prepocessor()

        caption = str(self.captions[idx])
        image_left = self._get_image(idx, preprocess)

        if aug_fn in [
            self.get_id_augs,
            self.get_crop_augs,
            self.get_rotate_augs,
            self.get_colorjitter_augs,
            self.get_flip_augs,
            self.get_colorize_grayscale_augs,
        ]:
            texts_left, images_left, texts_right, images_right = aug_fn(image_left, caption)
        else:
            raise ValueError(f"Augmentation {aug_fn.__name__} not supported.")

        return texts_left, images_left, texts_right, images_right
