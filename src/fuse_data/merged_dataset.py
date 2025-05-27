import logging
from typing import Tuple, List

import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import CC3M_TRAIN_CSV_PATH, CC12M_TRAIN_CSV_PATH, CC3M_VAL_CSV_PATH, DATA_DIR
from fuse_clip.fuse_clip_utils import fused_train_collator
from fuse_data.samplers import MergedBatchSampler, MergedSampler, DistributedMergedSampler
from open_clip_train.data import DataInfo
from open_clip_train.distributed import is_master


DOWNSAMPLE_FACTORS_CC3M = {
    "cc3m": 1.0,
    "cc12m": 1.0,
    "cc3m-wds": 1.0,
    "cc3m-aug": 1.0,
    "cc12m-aug": 1.0,
    "hq-edit": 1.0,
    "pix2pix": 1.0,
    "cc3m-vqa": 1.0,
    "llava": 1.0,
    "vg-crop": 1.0,
    "vg-vqa": 0.5,
    "mt-cir": 1.0,
}

DOWNSAMPLE_FACTORS_CC12M = {  # cc12m
    "cc3m": 1.0,
    "cc12m": 1.0,
    "cc3m-wds": 1.0,
    "cc3m-aug": 1.0,
    "cc12m-aug": 1.0,
    "hq-edit": 1.0,
    "pix2pix": 1.0,
    "cc3m-vqa": 1.0,
    "cc12m-vqa": 1.0,
    "llava": 1.0,
    "vg-crop": 1.0,
    "vg-vqa": 1.0,
    "mt-cir": 1.0,
}

class MultiDataloader:
    def __init__(self, datainfos, is_master=True):
        self.datainfos = datainfos
        self.dataloaders = [di.dataloader for di in datainfos]
        self.num_samples = sum(dl.num_samples for dl in self.dataloaders)
        self.dataloader = self
        self.epoch = 0
        self.seed = 0
        self.sample_dataloader = True
        if is_master:
            logging.info(f"Created MultiDataloader with {len(self.dataloaders)} dataloaders")
            logging.info(f"sample dataloader: {self.sample_dataloader}")

    @property
    def num_batches(self):
        if self.sample_dataloader:
            return sum(dl.num_batches for dl in self.dataloaders)
        else:
            return sum(dl.num_batches for dl in self.dataloaders)

    def __iter__(self):
        if not self.sample_dataloader:
            return self.iter_deterministic()
        else:
            return self.iter_sampled()

    def iter_deterministic(self):
        dl_iters = [iter(dl) for dl in self.dataloaders]
        for i in range(self.num_batches // len(self.dataloaders)):
            for dl_idx in range(len(self.dataloaders)):
                try:
                    yield next(dl_iters[dl_idx])
                except StopIteration:
                    dl_iters[dl_idx] = iter(self.dataloaders[dl_idx])

    def iter_sampled(self):
        g = np.random.default_rng(seed=self.seed + self.epoch)
        dl_iters = [iter(dl) for dl in self.dataloaders]
        weights = [dl.num_samples / self.num_samples for dl in self.dataloaders]
        for i in range(self.num_batches):
            dl_idx = g.choice(len(dl_iters), p=weights)
            try:
                yield next(dl_iters[dl_idx])
            except StopIteration:
                # We re-create a new iterator from the original dataloader
                dl_iters[dl_idx] = iter(self.dataloaders[dl_idx])
                yield next(dl_iters[dl_idx])

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        for di in self.datainfos:
            di.set_epoch(epoch)


class MergedDataset(Dataset):
    def __init__(self, datasets, random_subset=False, downsample_factors=None, use_combined_items=True):
        self.datasets = datasets
        self.n_datasets = len(self.datasets)

        if use_combined_items:
            self.combined_items_dict = {
                name : datasets[name].combined_items if hasattr(datasets[name], "combined_items")
                else [[idx] for idx in range(len(datasets[name]))] for name in datasets.keys()
            }
        else:
            self.combined_items_dict = {
                name: [[idx] for idx in range(len(datasets[name]))] for name in datasets.keys()
            }

        # downsample
        if downsample_factors:
            self._downsample(downsample_factors)

        if random_subset:
            self._random_subset(random_subset)

        # flatten combined items, [[i1, i2], [j1], ...]
        self.combined_items = []
        self.global_to_local = {}
        for name, comb in self.combined_items_dict.items():
            cutoff = self.cutoffs[name]
            self.combined_items.extend(
                [[idx + cutoff for idx in group] for group in comb]
            )
            self.global_to_local.update(
                {idx + cutoff: (name, idx) for group in comb for idx in group}
            )

    @property
    def cutoffs(self):
        cutoffs = {}
        cutoff_cur = 0
        for name, ds in self.datasets.items():
            cutoffs[name] = cutoff_cur
            cutoff_cur += len(ds)
        return cutoffs

    def _downsample(self, fractions):
        # downsample datasets
        # fractions is a dict {dataset_name fraction}

        # Downsample each dataset if fraction provided
        for name in self.datasets.keys():
            frac = fractions.get(name, 1.0)
            if frac == 1.0:
                continue
            count = int(frac * len(self.combined_items_dict[name]))
            # we use a fixed generator, so that we get the same data for each process
            generator = np.random.default_rng(seed=0)
            # Random choice of indices, shuffled
            chosen = generator.choice(len(self.combined_items_dict[name]), count, replace=False)
            self.combined_items_dict[name] = [self.combined_items_dict[name][c] for c in chosen]

    def _random_subset(self, n):
        assert len(self.datasets) == 1  # otherwise cutoff has to be adjusted
        # get random subset of combined items, so the sampler is aware
        for name, comb in self.combined_items_dict.items():
            if len(comb) <= n:  # no need to sample if we have less than n items
                self.combined_items_dict[name] = comb
                logging.info(f"Using all {len(comb)} items from {name}")
                continue
            random_group_idcs = np.random.choice(len(comb), n, replace=False)
            flat_len = 0
            new_comb = []
            for idx in random_group_idcs:
                current_group = comb[idx]
                needed = n - flat_len
                if len(current_group) <= needed:
                    new_comb.append(current_group)
                    flat_len += len(current_group)
                else:
                    # partial slice
                    new_comb.append(current_group[:needed])
                    flat_len += needed
                if flat_len >= n:
                    break
            self.combined_items_dict[name] = new_comb

    def __len__(self):
        return len(self.global_to_local)

    def __getitem__(self, item: int) -> Tuple:
        name, local_idx = self.global_to_local[item]
        try:
            sample = self.datasets[name][local_idx]
        except Exception as ex:
            logging.error(f"Exception while getting item {item}: {ex}")
            logging.error(f"dataset_name: {name}, local_index: {local_idx}")
            raise ex
        return sample

def get_merged_dataset(
    args, dataset_names: List, preprocess_fn, is_train, epoch=0, tokenizer=None, random_subset=False,
    **kwargs
):
    # assert is_train
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split(",")
    use_combined_items = args.combined_sampling
    datasets = {}
    downsample_factors = {}
    is_cc12m = "cc12m" in dataset_names or "cc12m-aug" in dataset_names
    DOWNSAMPLE_FACTORS = DOWNSAMPLE_FACTORS_CC12M if is_cc12m else DOWNSAMPLE_FACTORS_CC3M

    for dataset_name in dataset_names:
        if dataset_name in ["cc3m", "cc12m"]:
            from open_clip_train.data import CsvDataset
            if dataset_name == "cc3m" and is_train:
                data_path_cur = CC3M_TRAIN_CSV_PATH
            elif dataset_name == "cc3m" and not is_train:
                data_path_cur = CC3M_VAL_CSV_PATH
            elif dataset_name == "cc12m":
                data_path_cur = CC12M_TRAIN_CSV_PATH
            ds_ = CsvDataset(
                data_path_cur,
                preprocess_fn,
                img_key="path",
                caption_key="caption",
                sep=",",
                tokenizer=tokenizer,
                return_multimodal_format=True,
                is_arrow_dataset=False, #DATA_DIR is not None,  # cc3m in a different format
                dataset_cache_dir=DATA_DIR,
                cc_text_aug_prob = args.cc_text_aug_prob if is_train else 0.,  # Use only for training.
            )
        elif dataset_name == "cc3m-wds":
            from fuse_data.cc3m import get_cc3m_wds, CC3MWDS
            # ds_ = get_cc3m_wds(
            #     DATA_DIR,
            #     split='train' if is_train else 'validation',
            #     transform=preprocess_fn,
            #     tokenizer=tokenizer,
            #     return_multimodal_format=True,
            # )
            ds_ = CC3MWDS(
                DATA_DIR,
                split='train' if is_train else 'validation',
                transform=preprocess_fn,
                tokenizer=tokenizer,
                return_multimodal_format=True,
            )
        elif dataset_name in ["cc3m-aug", "cc12m-aug"]:
            from fuse_data.augmented_pretrain_data import AugmentedCsvDataset
            if dataset_name == "cc3m-aug":
                data_path_cur = CC3M_TRAIN_CSV_PATH if is_train else CC3M_VAL_CSV_PATH
                unimodal_prob = kwargs.get("unimodal_prob")
                if unimodal_prob is None:
                    unimodal_prob = 0.9 if is_train else 0.0
            elif dataset_name == "cc12m-aug":
                data_path_cur = CC12M_TRAIN_CSV_PATH
                unimodal_prob = kwargs.get("unimodal_prob")
                if unimodal_prob is None:
                    unimodal_prob = 0.971 if is_train else 0.0
            ds_ = AugmentedCsvDataset(
                data_path_cur,
                preprocess_fn,
                img_key="path",
                caption_key="caption",
                sep=",",
                tokenizer=tokenizer,
                unimodal_prob=unimodal_prob,
                is_master=is_master(args),
                multiaug=use_combined_items,
            )
        elif dataset_name == "hq-edit":
            from fuse_data.hq_edit_data import HQEdit
            ds_ = HQEdit(
                txt_tokenizer=tokenizer,
                image_processor=preprocess_fn
            )
        elif dataset_name == "pix2pix":
            from fuse_data.pix2pix_data import Pix2Pix
            ds_ = Pix2Pix(
                txt_tokenizer=tokenizer,
                image_processor=preprocess_fn
            )
        elif dataset_name in  ["llava", "cc3m-vqa", "cc12m-vqa"]:
            from fuse_data.vqa_data import VQAData
            ds_ = VQAData(
                txt_tokenizer=tokenizer,
                image_processor=preprocess_fn,
                source=dataset_name,
            )
        elif dataset_name == "vg-crop":
            from fuse_data.visual_genome import VisualGenome
            ds_ = VisualGenome(
                txt_tokenizer=tokenizer,
                image_processor=preprocess_fn,
                mode="crop",
                is_train=is_train
            )
        elif dataset_name == "vg-pos":
            from fuse_data.visual_genome import VisualGenome
            ds_ = VisualGenome(
                txt_tokenizer=tokenizer,
                image_processor=preprocess_fn,
                mode="position",
                is_train=is_train
            )
        elif dataset_name == "vg-vqa":
            from fuse_data.visual_genome import VisualGenome
            ds_ = VisualGenome(
                txt_tokenizer=tokenizer,
                image_processor=preprocess_fn,
                mode="vqa",
                is_train=is_train
            )
        elif dataset_name == "mt-cir":
            from fuse_data.mtcir_data import MTCIR
            ds_ = MTCIR(
                txt_tokenizer=tokenizer,
                image_processor=preprocess_fn,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        datasets[dataset_name] = ds_
        downsample_factors[dataset_name] = DOWNSAMPLE_FACTORS[dataset_name] if is_train else 1.0


    ds = MergedDataset(
        datasets,
        downsample_factors=downsample_factors,
        random_subset=random_subset,
        use_combined_items=use_combined_items
    )

    if is_train and is_master(args):
        logging.info(f"train on merged dataset with {len(datasets)} datasets, and {len(ds)} total samples")
        logging.info(f"downsample factors: {downsample_factors}")
        logging.info(f"use combined items: {use_combined_items}")

    num_samples = len(ds)
    batch_size = args.batch_size
    shuffle = True  # since we use random_subset for eval, we should always get the same samples anyway

    sampler = DistributedMergedSampler(ds, shuffle=shuffle) if args.distributed and is_train \
        else MergedSampler(ds, shuffle=shuffle)


    batch_sampler = MergedBatchSampler(
        sampler, batch_size, drop_last=is_train, enforce_exact_n_batches=(args.distributed and is_train)
    )

    # todo: cc3m-aug will yield larger batches. for train we truncate them in the training script
    # todo: but for validation we do not. thus the effective batch size is larger for cc3m-aug validation

    dataloader = DataLoader(
        ds,
        collate_fn=fused_train_collator,
        num_workers=args.workers,
        pin_memory=True,
        batch_sampler=batch_sampler,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler=sampler)


def get_multi_dataloader(args, dataset_names: List, preprocess_fn, is_train, epoch=0,
                         tokenizer=None, random_subset=False, **kwargs):
    datainfos = [
        get_merged_dataset(args=args, dataset_names=[name], preprocess_fn=preprocess_fn,
                            is_train=is_train, epoch=epoch, tokenizer=tokenizer, random_subset=random_subset,
                            **kwargs)
        for name in dataset_names
    ]
    return MultiDataloader(datainfos, is_master=is_master(args))



if __name__ == '__main__':
    import textwrap
    from argparse import Namespace
    import matplotlib.pyplot as plt
    import random
    from open_clip import get_tokenizer
    from fuse_clip.fuse_clip_preprocess import get_fuse_clip_image_preprocess
    from torch.utils.data import DataLoader

    # train_data ="cc3m-aug,hq-edit,llava,vg-crop,vg-vqa"
    train_data = "cc3m-aug"

    # set log level
    logging.basicConfig(level=logging.INFO)

    # Arguments for dataset and DataLoader
    args = Namespace(
        train_data=train_data,
        batch_size=100,
        workers=2,
        distributed=False,
        rank=0,
    )

    # Preprocessing functions and tokenizer
    preprocess_fn = get_fuse_clip_image_preprocess(train=True)
    txt_tokenizer = get_tokenizer("fuse-clip-titok", context_length=180)
    dataloader = get_merged_dataset(
        args=args, dataset_names=train_data.split(","), preprocess_fn=preprocess_fn, is_train=True,
        tokenizer=txt_tokenizer,
        #random_subset=120
    ).dataloader


    # Helper function to decode text tokens
    def decode_tokens(txt_tokenizer, txt):
        txt = txt[txt != 0][1:-1].tolist()  # remove pad, start and end token
        return txt_tokenizer.decode(txt)


    # plot examples
    nrows = 30
    fig, axs = plt.subplots(nrows, 4, figsize=(10, 3 * nrows))
    for batch in dataloader:
        texts_left = batch["texts_left"]
        images_left = batch["images_left"]
        texts_right = batch["texts_right"]
        images_right = batch["images_right"]
        for i in range(2 * nrows):
            row = i // 2
            col = 0 if i % 2 == 0 else 2
            edit = decode_tokens(txt_tokenizer, texts_left[i])
            right_txt = decode_tokens(txt_tokenizer, texts_right[i])
            input_image = images_left[i]
            output_image = images_right[i]
            axs[row, col].imshow(input_image.permute(1, 2, 0)) if input_image is not None else None
            axs[row, col].set_title(textwrap.fill(edit[:120], 40), fontsize=8)
            axs[row, col + 1].imshow(output_image.permute(1, 2, 0)) if output_image is not None else None
            axs[row, col + 1].set_title(textwrap.fill(right_txt[:120], 40), fontsize=8)
        [ax.axis("off") for ax in axs.flatten()]
        plt.show()
        break
    print("done")