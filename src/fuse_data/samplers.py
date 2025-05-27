import math
from typing import Optional, Iterator, List

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from fuse_clip.fuse_clip_utils import flatten_list


class MergedSampler(Sampler):
    def __init__(self, data_source, shuffle=True, generator=None):
        self.data_source = data_source
        self.combined_items = data_source.combined_items
        self.shuffle = shuffle
        self.generator = generator

    @property
    def flat_len(self):
        return len(flatten_list(self.combined_items))

    def __len__(self):
        return len(self.combined_items)

    def __iter__(self) -> Iterator[int]:
        generator = self.generator
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)

        if self.shuffle:
            idcs = torch.randperm(len(self.combined_items), generator=generator).tolist()
        else:
            idcs = list(range(len(self.combined_items)))
        for idx in idcs:
            yield self.combined_items[idx]


class DistributedMergedSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.combined_items = dataset.combined_items
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.combined_items) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_groups = math.ceil(
                (len(self.combined_items) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_groups = math.ceil(len(self.combined_items) / self.num_replicas)
        self.total_size = self.num_groups * self.num_replicas  # total num groups
        # print(
        #     f"num_groups = {self.num_groups}, total_size = {self.total_size},"
        #     f"avg_group_size = {avg_group_size}, expected_n_batches = {self.expected_n_batches}"
        #     f"num_replicas = {self.num_replicas}, rank = {self.rank}"
        # )
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.combined_items), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.combined_items)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            raise NotImplementedError
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_groups

        for idx in indices:
            yield self.combined_items[idx]

    def __len__(self) -> int:
        return self.num_groups

    @property
    def flat_len(self):
        return len(flatten_list(self.combined_items)) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class MergedBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last=True, enforce_exact_n_batches=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if enforce_exact_n_batches:
            # so that each rank yields the same number of batches
            self.num_batches = self.sampler.flat_len // self.batch_size
        else:
            self.num_batches = float("inf")

    def __len__(self):
        if self.num_batches < float("inf"):
            return self.num_batches
        else:
            n_batches =  self.sampler.flat_len / self.batch_size
            return math.floor(n_batches) if self.drop_last else math.ceil(n_batches)

    def __iter__(self) -> Iterator[List[int]]:
        n_yielded = 0
        batch = []
        sampler_iter = iter(self.sampler)

        while n_yielded < self.num_batches:
            try:
                samples = next(sampler_iter)
            except StopIteration:
                # If we've reached num_batches, break out even if the sampler is re-initialized.
                if n_yielded >= self.num_batches:
                    break
                if self.num_batches == float("inf"):
                    # no enforced exact #, so we are done
                    break
                # Otherwise re-init the sampler if we still need more batches
                else:
                    sampler_iter = iter(self.sampler)
                    samples = next(sampler_iter)

            batch.extend(samples)

            # If we have enough for a full batch, yield it.
            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size:] if len(batch) > self.batch_size else []
                n_yielded += 1
                if n_yielded >= self.num_batches:
                    break

        # yield leftover
        if (not self.drop_last) and (len(batch) > 0) and (n_yielded < self.num_batches):
            yield batch
