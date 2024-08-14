# https://github.com/pytorch/pytorch/blob/v1.11.0/torch/utils/data/distributed.py

import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class DistributedSampler(Sampler[T_co]):
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

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 1, seqs_per_video: int = 150,
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
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.seqs_per_video = seqs_per_video
        self.num_videos = len(self.dataset.videos)
        
        self.dataset_indices = torch.arange(len(self.dataset)).view(-1, self.seqs_per_video)
        assert self.dataset_indices.size(dim=0) == self.num_videos
        print("Number of videos = ", self.num_videos)
        print("Number of replicas = {} ; Batch size = {}".format(self.num_replicas, self.batch_size))
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.num_videos % (self.num_replicas * self.batch_size) != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_videos_per_replica = math.ceil(
                (self.num_videos - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
            self.num_videos_per_replica = self.num_videos_per_replica - self.num_videos_per_replica % self.batch_size
        else:
            self.num_videos_per_replica = math.ceil(self.num_videos / self.num_replicas)  # type: ignore[arg-type]
            if self.num_videos_per_replica % self.batch_size != 0:
                self.num_videos_per_replica = self.num_videos_per_replica + (self.batch_size - self.num_videos_per_replica % self.batch_size)

        assert self.num_videos_per_replica % self.batch_size == 0
        self.total_num_videos = self.num_videos_per_replica * self.num_replicas
        print("drop_last = ", self.drop_last)
        print("Total number of videos = ", self.total_num_videos)

        self.num_samples = self.num_videos_per_replica * self.seqs_per_video
        self.total_size = self.total_num_videos * self.seqs_per_video
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            video_indices = torch.randperm(self.num_videos, generator=g)
        else:
            video_indices = torch.arange(self.num_videos)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_num_videos - self.num_videos
            if padding_size <= self.num_videos:
                video_indices = torch.cat((video_indices, video_indices[:padding_size]))
            else:
                video_indices = torch.cat((video_indices, video_indices.repeat(math.ceil(padding_size / self.num_videos))[:padding_size]))
        else:
            # remove tail of data to make it evenly divisible.
            video_indices = video_indices[:self.total_num_videos]
        assert len(video_indices) == self.total_num_videos

        # subsample        
        indices = torch.split(self.dataset_indices[video_indices], self.batch_size * self.num_replicas, dim=0)
        indices = torch.cat([torch.reshape(torch.transpose(indice, 0, 1), (-1,)) for indice in indices], dim=0)
        indices = indices.tolist()
        assert len(indices) == self.total_size
            
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # seed
        seeds = torch.randperm(self.total_size)[:self.total_num_videos]
        seeds = torch.unsqueeze(seeds, 1)
        seeds = seeds.expand(-1, self.seqs_per_video)
        seeds = torch.split(seeds, self.batch_size * self.num_replicas, dim=0)
        seeds = torch.cat([torch.reshape(torch.transpose(seed, 0, 1), (-1,)) for seed in seeds], dim=0)
        seeds = seeds.tolist()
        assert len(seeds) == self.total_size

        seeds = seeds[self.rank:self.total_size:self.num_replicas]
        assert len(seeds) == self.num_samples

        return iter(zip(indices,seeds))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch