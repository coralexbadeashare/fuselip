import ast
import copy
import json
import logging
import math
import os
import random
import sys
from functools import partial

import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from fuse_data.samplers import DistributedMergedSampler
from mmeb.mmeb_dataset import MMEBTrainDataset
from mmeb.mmeb_utils import TrainDataArguments, TxtModifier, PREFIXES_RIGHT
from fuse_clip.fuse_clip_utils import fused_train_collator
from open_clip import SimpleTokenizer
from open_clip_train.distributed import is_master

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None,
                 return_multimodal_format=False, is_arrow_dataset=False, dataset_cache_dir=None,
                 cc_text_aug_prob=0):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        from config import DATA_DIR
        self.data_path = os.path.join(DATA_DIR, "cc3m")

        self.return_multimodal_format = return_multimodal_format

        # Downloaded like this from HF, probably it's not a great solution.
        self.is_arrow_dataset = is_arrow_dataset
        if is_arrow_dataset:
            import datasets
            split = 'validation' if 'validation' in input_filename else 'train'
            assert dataset_cache_dir is not None
            self.dataset = datasets.load_dataset(
                "pixparse/cc3m-wds",
                cache_dir=dataset_cache_dir,
                #download_config=datasets.DownloadConfig(resume_download=True),
                )[split]
            self.keys = self.dataset['__key__']

        # For text guided augmentations (for CC3M at the moment).
        self.cc_text_aug_prob = cc_text_aug_prob
        if cc_text_aug_prob > 0:
            assert self.return_multimodal_format, 'Only works with multimodal format.'
            from fuse_data.text_augmentations import all_modifications
            self.all_text_augmentations = all_modifications
            print(f'{len(all_modifications)} possible text augmentations found,'
                  f' used with probability {self.cc_text_aug_prob}.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):

        if self.is_arrow_dataset:
            #mapped_idx = self.keys.index(self.images[idx].split('/')[-1].replace('.jpg', ''))
            mapped_idx = idx
            images = self.transforms(self.dataset[mapped_idx]['jpg'])
            texts = self.tokenize([str(self.dataset[mapped_idx]['txt'])])[0]
        else:
            img_path = os.path.join(self.data_path, str(self.images[idx]))
            images = self.transforms(Image.open(img_path))
            caption = str(self.captions[idx])
            img_aug = ""
            # Sample text-based augmtation (add texts to image, caption or both).
            if self.cc_text_aug_prob > 0:
                if random.random() < self.cc_text_aug_prob:
                    img_aug, txt_aug = random.choice(self.all_text_augmentations)
                    caption = caption + ((' ' + txt_aug) if txt_aug != "" else "")
            texts = self.tokenize([caption])[0]
        if not self.return_multimodal_format:
            return images, texts
        else:
            return self.tokenize(img_aug)[0], images, texts, None


class CsvMultimodalDataset(Dataset):
    def __init__(
            self, input_filename, transforms, img_left_key, text_left_key, img_right_key, text_right_key,
            sep="\t", tokenizer=None
    ):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images_left = df[img_left_key].tolist() if img_left_key else None
        self.captions_left = df[text_left_key].tolist() if text_left_key else None
        self.images_right = df[img_right_key].tolist() if img_right_key else None
        self.captions_right = df[text_right_key].tolist() if text_right_key else None
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        for el in [self.images_left, self.captions_left, self.images_right, self.captions_right]:
            if el is not None:
                return len(el)

    def __getitem__(self, idx):
        images_left = self.transforms(Image.open(str(self.images_left[idx]))) if self.images_left else torch.empty(0)
        texts_left = self.tokenize([str(self.captions_left[idx])])[0] if self.captions_left else torch.empty(0, dtype=torch.int64)
        images_right = self.transforms(Image.open(str(self.images_right[idx]))) if self.images_right else torch.empty(0)
        texts_right = self.tokenize([str(self.captions_right[idx])])[0] if self.captions_right else torch.empty(0, dtype=torch.int64)
        return texts_left, images_left, texts_right, images_right


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, (DistributedSampler, DistributedMergedSampler)):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size*1,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample or 'left.txt' in sample or 'right.txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample
                 or 'left.jpg' in sample or 'right.jpg' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if "fname" not in filesample:
            logging.warning(f"Skipping filesample without fname: {filesample}")
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
    ])

    if not args.multimodalize_wds:
        pipeline.extend([
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train)
        ])
    else:
        collation_fn = partial(wds.filters.default_collation_fn, combine_tensors=False)
        def adapt_sample(x):
            pad_id = 0  # todo (simple tokenizer has no pad id attribute)
            assert isinstance(tokenizer, SimpleTokenizer)
            if random.random() >= 0.1:
                return (x[0], None, torch.full_like(x[0], pad_id), x[3])  # (txt) - (img)
            else:
                return (x[0], None, x[2], x[3])  # (txt) - (txt, img)

        pipeline.extend([
            wds.to_tuple("text", "image", "text", "image"),
            wds.map(adapt_sample),
            wds.batched(args.batch_size, partial=not is_train, collation_fn=collation_fn),
        ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_fused_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    assert is_train
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    txt_modifier = TxtModifier(model_name=args.model, is_train=is_train)

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.'
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                wds.tarfile_to_samples(handler=log_and_continue),  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )

    def adapt_sample(x):
        # convert samples from non-fused dataset (cc3m) to fused format
        if "cc3m" in x["__url__"] or "cc12m" in x["__url__"]:
            image_key = next((k for k in  ["jpg", "png", "jpeg", "webp"] if k in x), None)
            # rename keys: jpg -> left.jpg, txt -> right.txt
            x = {"left.jpg": x[image_key], "left.txt": "", "right.txt": x["txt"]}

        # add missing keys
        if "left.jpg" not in x:
            x["left.jpg"] = None
        if "right.jpg" not in x:
            x["right.jpg"] = None

        return x

    def maybe_preprocess_img(img):
        return preprocess_img(img) if img is not None else None

    def preprocess_text(text):
        #try:
        text = txt_modifier(text)
        # except:
        #     text = txt_modifier(text, subset='')
        return tokenizer(text)[0]

    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.map(adapt_sample),
            wds.map_dict(**{
                "left.jpg": maybe_preprocess_img, "right.jpg": maybe_preprocess_img,
                "left.txt": preprocess_text, "right.txt": preprocess_text,
            }),
        ]
    )


    collation_fn = partial(wds.filters.default_collation_fn, combine_tensors=False)
    pipeline.extend(
        [
            wds.to_tuple("left.txt", "left.jpg", "right.txt", "right.jpg", none_is_error=False),
            wds.batched(args.batch_size, partial=not is_train, collation_fn=collation_fn),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )



    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, is_multimodal=False, **kwargs):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if not is_multimodal:
        dataset = CsvDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    else:
        dataset = CsvMultimodalDataset(
            input_filename,
            preprocess_fn,
            img_left_key="input_image",
            text_left_key="edit",
            img_right_key="output_image",
            text_right_key=None,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_mmeb_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    assert is_train, "val not implemented"
    data_args = TrainDataArguments(image_dir=args.train_data)
    txt_modifier_left = TxtModifier(model_name=args.model, is_train=is_train, empty_prompt=True)
    txt_modifier_right = TxtModifier(
        model_name=args.model, is_train=is_train, empty_prompt=True, prompt_prefixes=PREFIXES_RIGHT
        )

    if args.dataset_type == "mmeb-cc3m":
        cc3m_path = "/mnt/datasets/cc3m-hf/train.csv"  # todo improve
        cc3m = CsvDataset(
            cc3m_path,
            transforms=preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=lambda x: [x]
        )
    else:
        cc3m = None

    dataset = MMEBTrainDataset(
        data_args,
        txt_tokenizer=tokenizer,
        image_processor=preprocess_fn,
        txt_modifier_left=txt_modifier_left,
        txt_modifier_right=txt_modifier_right,
        cc3m=cc3m,
    )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        collate_fn=fused_train_collator,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "csv-multimodal":
        return partial(get_csv_dataset, is_multimodal=True)
    elif dataset_type == "fused-wds":
        return get_fused_wds_dataset
    elif dataset_type == "merged":
        from fuse_data.merged_dataset import get_merged_dataset
        return partial(get_merged_dataset, dataset_names=data_path)
    elif dataset_type == "multi":
        from fuse_data.merged_dataset import get_multi_dataloader
        return partial(get_multi_dataloader, dataset_names=data_path)
    elif dataset_type in ["mmeb", "mmeb-cc3m"]:
        return get_mmeb_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    elif dataset_type == 'finetuning':
        # from fuse_data.ft_datasets import get_ft_dataset
        return partial(get_ft_dataset, dataset_name=data_path)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        # try:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args=args, preprocess_fn=preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
        # except TypeError:
        #     data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
        #         args=args, preprocess_img=preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data and is_master(args):
        logging.info(f"Loading validation data")
        ds_type_ = args.dataset_type if args.dataset_type not in ["multi", "mmeb"] else "merged"  # no multi for eval
        data["val"] = {}
        for data_name in args.val_data.split(","):
            # fix batch size, so that loss is comparable
            args_eval = copy.deepcopy(args)
            args_eval.batch_size = 256
            # we split datasets, so that we can compute metrics on them individually
            data_val = get_dataset_fn(data_name, ds_type_)(
                args=args_eval, preprocess_fn=preprocess_val, is_train=False, tokenizer=tokenizer,
                random_subset=1000
            )
            data["val"].update({data_name: data_val})
        logging.info(
            "validation: " +
            ", ".join([f"{name} : {len(data_val.dataloader.dataset)}" for name, data_val in data["val"].items()])
        )

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data


def get_ft_dataset(
    args, dataset_name: str, preprocess_fn, is_train, epoch=0, tokenizer=None, random_subset=False,
    **kwargs
):
    from os.path import join
    from config import FT_DATA_DIR
    
    if dataset_name == "imagenet":
        from fuse_data.imagenet import ImageNet
        _data_dir = join(FT_DATA_DIR, 'imagenet/train') if is_train else join(FT_DATA_DIR, 'imagenet/val')
        ds_ = ImageNet(
            transforms_fn=preprocess_fn,
            tokenizer=tokenizer,
            return_multimodal_format=True,
            data_dir=_data_dir,
        )
    elif dataset_name == 'hateful_memes':
        from fuse_data.hateful_memes import HatefulMemes
        _data_dir = join(FT_DATA_DIR, 'hateful_memes')
        ds_ = HatefulMemes(
            preprocess_fn=preprocess_fn,
            tokenizer=tokenizer,
            data_dir=_data_dir,
            split='train' if is_train else 'test'
        )
    elif dataset_name == 'cifar10':
        from fuse_data.cifar import CIFAR10_mm
        _data_dir = join(FT_DATA_DIR, 'CIFAR10')
        ds_ = CIFAR10_mm(
            preprocess_fn=preprocess_fn,
            tokenizer=tokenizer,
            data_dir=_data_dir,
            is_train=is_train,
        )
    else:
        raise ValueError(f'Unknown dataset {dataset_name}.')

    num_samples = len(ds_)
    sampler = DistributedSampler(ds_) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        ds_,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)