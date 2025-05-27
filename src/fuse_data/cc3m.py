from PIL import Image
import webdataset as wds
import io
import os
from torch.utils.data import IterableDataset


class CC3MWDS(IterableDataset):

    def __init__(
        self, data_path, split, transform=None, tokenizer=None,
        return_multimodal_format=False):
        
        self.transform = transform
        self.tokenizer = tokenizer
        if split == 'train':
            tars = 'cc3m-train-{0000..0575}.tar'
        elif split == 'validation':
            tars = 'cc3m-validation-{0000..0015}.tar'
        data = os.path.join(data_path, 'cc3m_v2', tars)
        self.dataset = wds.WebDataset(data, shardshuffle=split != 'train').map(self.preprocess)
        self.return_multimodal_format = return_multimodal_format
        self.length = 2905954

    def preprocess(self, sample):
        img_data = sample["jpg"]
        txt = sample['txt']
        img = Image.open(io.BytesIO(img_data)).convert("RGB")  # Convert binary to PIL image
        if self.transform:
            img = self.transform(img)
        if self.tokenizer:
            txt = self.tokenizer([str(txt)])[0]
        if self.return_multimodal_format:
            return self.tokenizer("")[0], img, txt, None
        return img, txt

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.length


def get_cc3m_wds(
        data_path, split, transform=None, tokenizer=None,
        return_multimodal_format=False):

    def _preprocess(sample):
        img_data = sample["jpg"]
        txt = sample['txt']
        img = Image.open(io.BytesIO(img_data)).convert("RGB")  # Convert binary to PIL image
        if transform:
            img = transform(img)
        if tokenizer:
            txt = tokenizer([str(txt)])[0]
        if return_multimodal_format:
            return tokenizer("")[0], img, txt, None
        return img, txt
    
    if split == 'train':
            tars = 'cc3m-train-{0000..0575}.tar'
    elif split == 'validation':
        tars = 'cc3m-validation-{0000..0015}.tar'
    data = os.path.join(data_path, 'cc3m_v2', tars)
    dataset = wds.WebDataset(data, shardshuffle=split != 'train').map(_preprocess)

    return dataset