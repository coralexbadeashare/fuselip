import torch
from PIL import Image
from torch.utils.data import Dataset
from os.path import join
import pandas as pd


class HatefulMemes(Dataset):

    def __init__(self, data_dir, split, preprocess_fn=None, tokenizer=None) -> None:
        super().__init__()

        self._get_split(data_dir, split)
        self.split = split
        self.data_dir = data_dir
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer
    
    def _get_split(self, data_dir, split):
        if split == 'train':
            self.df = pd.read_csv(join(data_dir, 'filtered_train.csv'))
        elif split == 'test':
            df1 = pd.read_csv(join(data_dir, 'filtered_test_seen.csv'))
            df2 = pd.read_csv(join(data_dir, 'filtered_test_unseen.csv'))
            self.df = pd.concat([df1, df2], ignore_index=True)
        elif split == 'val':
            df1 = pd.read_csv(join(data_dir, 'filtered_dev_seen.csv'))
            df2 = pd.read_csv(join(data_dir, 'filtered_dev_unseen.csv'))
            self.df = pd.concat([df1, df2], ignore_index=True)

    def __len__(self,):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        txt = row['text']
        img = Image.open(join(self.data_dir, row['img']))
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)
        if self.tokenizer is not None:
            txt = self.tokenizer(txt)[0]
        label = row['label'].item()
        return img, txt, label
