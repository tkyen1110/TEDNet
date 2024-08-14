# https://github.com/pytorch/pytorch/blob/v1.11.0/torch/utils/data/distributed.py

import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset


T_co = TypeVar('T_co', covariant=True)


class TestSampler(Sampler[T_co]):
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

    def __init__(self, dataset: Dataset, batch_size: int = 1, seqs_per_video: int = 1200,
                 ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seqs_per_video = seqs_per_video
        self.num_videos = len(self.dataset.videos)
        self.total_size = self.num_videos * self.seqs_per_video
        assert self.total_size == len(self.dataset)
        self.dataset_indices = torch.arange(len(self.dataset)).view(-1, self.seqs_per_video)
        assert self.dataset_indices.size(dim=0) == self.num_videos
        print("Number of videos = {} ; Batch size = {}".format(self.num_videos, self.batch_size))
        assert self.num_videos % self.batch_size == 0        

    def __iter__(self) -> Iterator[T_co]:
        # subsample
        indices = torch.split(self.dataset_indices, self.batch_size, dim=0)
        indices = torch.cat([torch.reshape(torch.transpose(indice, 0, 1), (-1,)) for indice in indices], dim=0)
        indices = indices.tolist()
        assert len(indices) == self.total_size

        return iter(indices)

    def __len__(self) -> int:
        return self.total_size
