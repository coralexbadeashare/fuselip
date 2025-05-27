from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader

from fuse_clip.fuse_clip_utils import fused_train_collator, decode_tokens, set_seed


class HQEdit:
    def __init__(self, txt_tokenizer, image_processor, do_tokenize=True, use_original_only=False):
        self.data = load_dataset("UCSC-VLAA/HQ-Edit")["train"]
        self.txt_tokenizer = txt_tokenizer
        self.image_processor = image_processor
        self.do_tokenize = do_tokenize

        self.combined_items = [[idx, len(self.data) + idx] for idx in range(len(self.data))]
        self.combined_items.extend([[2 * len(self.data) + idx] for idx in range(len(self.data))])

        self.use_original_only = use_original_only

    def _process_image(self, image):
        if image is None:
            return None
        image = self.image_processor(image)
        return image

    def __len__(self):
        if not self.use_original_only:
            return 3*len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, item) -> Tuple:
        if item < len(self.data):
            el = self.data[item]
            input_image = el["input_image"]
            edit = el["edit"]
            output_image = el["output_image"]
            right_txt = ""
        elif item < 2*len(self.data):
            el = self.data[item % len(self.data)]
            input_image = el["output_image"]
            edit = el["inverse_edit"]
            output_image = el["input_image"]
            right_txt = ""
        else:
            el = self.data[item % len(self.data)]
            input_image = el["input_image"]
            edit = el["edit"]
            output_image = el["output_image"]
            right_txt = el["output"]

        if self.do_tokenize:
            edit = self.txt_tokenizer(edit).squeeze()
            right_txt = self.txt_tokenizer(right_txt).squeeze()

        input_image = self._process_image(input_image)
        output_image = self._process_image(output_image)
        return edit, input_image, right_txt, output_image
