from datasetops.dataset import Dataset
from datasetops.helpers import documents, monkeypatch, named
from datasetops.interfaces import (
    IDataset,
    ISampleProvider,
    Sample,
    ElemNameToIndex,
    ElemIndex,
    ElemStats,
)
from typing import Sequence, List, Dict, Union
from functools import reduce
from math import floor
from warnings import warn
import numpy as np


def collect_datasets(
    dataset: IDataset,
    dataset_or_datasets: Union[IDataset, Sequence[IDataset]],
    *rest_datasets: IDataset
) -> List[IDataset]:
    datasets = [dataset]
    if type(dataset_or_datasets) in {list, tuple}:
        datasets.extend(list(dataset_or_datasets))
    else:
        datasets.append(dataset_or_datasets)  # type: ignore

    datasets.extend(rest_datasets)
    return datasets


def zipped_elem_name2index(datasets: Sequence[IDataset]) -> ElemNameToIndex:
    if set.intersection(*[set(d.names.keys()) for d in datasets]):
        return {}
    else:
        elem_name2index = {}
        acc = 0
        for ds in datasets:
            for n, i in ds.names.items():
                elem_name2index[n] = int(i) + acc

            acc += len(ds.shape)  # type: ignore

        return elem_name2index


def zipped_stats(datasets: Sequence[IDataset]) -> Dict[ElemIndex, ElemStats]:
    if set.intersection(*[set(d.names.keys()) for d in datasets]):
        return {}
    else:
        stats = {}
        acc = 0
        for ds in datasets:
            for i in range(len(ds.shape)):  # type: ignore
                if i in ds._stats:  # type: ignore
                    stats[i] = int(i) + acc

            acc += len(ds.shape)  # type: ignore

        return stats


# ========= Datasets =========


class ZipProvider(ISampleProvider):
    def __init__(self, parents: Sequence[IDataset]):
        """ Compose datasets by zipping and flattening their items.
        The resulting dataset will have a length equal to the shortest
        of provided datasets.
        NB: Any class-specific information will be lost after this transformation,
        and methods such as classwise_subsampling will not work.

        Arguments:
            parents {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(parents) < 1:
            raise ValueError("Not enough datasets given to compose")

        self._parents = parents
        self._sample_ids = range(min([len(ds) for ds in self._parents]))

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> Sample:
        return tuple([elem for ds in self._parents for elem in ds[idx]])

    def trace(self):
        return {
            "operation_name": "zip",
            "operation_parameters": {},
            "parent": [p.trace() for p in self._parents],
        }


class CartesianProductProvider(ISampleProvider):
    def __init__(self, parents: Sequence[IDataset]):
        """Compose datasets with a cartesian product.

        This will produce a dataset containing all combinations of data.
        Example: For two sets [1,2], ['a','b'] it produces
        [(1,'a'), (2,'a'), (1,'b'), (2,'b'),].
        The resulting dataset will have a length equal to the product of
        the length of the parent datasets.

        NB: Any class-specific information will be lost after this transformation,
        and methods such as classwise_subsampling will not work.

        Arguments:
            parents {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(parents) < 1:
            raise ValueError("Not enough datasets given to compose")

        self._parents = parents
        self._parent_lengths: List[int] = [len(ds) for ds in parents]
        self._sample_ids = range(
            int(reduce(lambda acc, ds: acc * len(ds), self._parents, int(1)))
        )
        self._acc_len = reduce(
            lambda acc, ds: acc + [acc[-1] * len(ds)], self._parents[:-1], [int(1)],
        )

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> Sample:
        inds = [
            floor(idx / al) % l for al, l in zip(self._acc_len, self._parent_lengths)
        ]

        return tuple(
            [elem for i, ds in enumerate(self._parents) for elem in ds[inds[i]]]
        )

    def trace(self):
        return {
            "operation_name": "cartesian_product",
            "operation_parameters": {},
            "parent": [p.trace() for p in self._parents],
        }


class ConcatProvider(ISampleProvider):
    def __init__(self, parents: Sequence[IDataset]):
        """Compose datasets by concatenating them, placing one after the other.
        The resulting dataset will have a length equal to the sum of datasets.

        Arguments:
            parents {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(parents) < 1:
            raise ValueError("Not enough datasets given to compose")

        for i in range(len(parents) - 1):
            if parents[i].shape != parents[i + 1].shape:  # type: ignore
                warn(
                    (
                        "Concatenating datasets with different element shapes "
                        "constitutes undefined behavior"
                    )
                )

        self._parents = parents
        self._sample_ids = range(sum([len(ds) for ds in self._parents]))
        self._acc_idx_range = reduce(
            lambda acc, ds: acc + [len(ds) + acc[-1]], self._parents, [0]
        )

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> Sample:
        dataset_index = (
            np.where(np.array(self._acc_idx_range) > idx)[0][0] - 1  # type:ignore
        )
        index_in_dataset = idx - self._acc_idx_range[dataset_index]
        return self._parents[dataset_index][index_in_dataset]

    def trace(self):
        return {
            "operation_name": "concat",
            "operation_parameters": {},
            "parent": [p.trace() for p in self._parents],
        }


@documents(IDataset)
@monkeypatch(Dataset)
@named("zip")
def zipped(
    self: Dataset,
    dataset_or_datasets: Union[IDataset, Sequence[IDataset]],
    *rest_datasets: IDataset
) -> IDataset:
    """ Compose datasets by zipping and flattening their items.

    The resulting dataset will have a length equal to the shortest of provided datasets.
    If there are name collisionts between the dataset, names will be discarded

    Arguments:
        dataset_or_datasets {Union[IDataset, Sequence[IDataset]]} -- Either a single dataset or a list of them
        rest_dataset {IDataset} -- remaining arguments may be additional datasets

    Returns:
        IDataset
    """
    datasets = collect_datasets(self, dataset_or_datasets, *rest_datasets)
    compose = ZipProvider(datasets)
    return Dataset(
        parent=compose,
        sample_ids=range(len(compose)),
        operation_name="zip_wrapper",
        elem_name2index=zipped_elem_name2index(datasets),
        stats=zipped_stats(datasets),
    )


@documents(IDataset)
@monkeypatch(Dataset)
def concat(
    self: Dataset,
    dataset_or_datasets: Union[IDataset, Sequence[IDataset]],
    *rest_datasets: IDataset
) -> IDataset:
    """ Compose datasets by concatenating their items (like for lists).

    The resulting dataset will have a length equal the sum of dataset lengths.
    Datasets must have the same number of elements.
    The names of the first dataset (self) are keps.
    Any computed statistics are discarded.

    Arguments:
        dataset_or_datasets {Union[IDataset, Sequence[IDataset]]} -- Either a single dataset or a list of them
        rest_dataset {IDataset} -- remaining arguments may be additional datasets

    Returns:
        IDataset
    """
    datasets = collect_datasets(self, dataset_or_datasets, *rest_datasets)
    compose = ConcatProvider(datasets)
    return Dataset(
        parent=compose,
        sample_ids=range(len(compose)),
        operation_name="concat_wrapper",
        elem_name2index=self._elem_name2index,
    )


@documents(IDataset)
@monkeypatch(Dataset)
def __add__(self, other: IDataset) -> IDataset:
    return self.concat(other)


@documents(IDataset)
@monkeypatch(Dataset)
def cartesian_product(
    self: Dataset,
    dataset_or_datasets: Union[IDataset, Sequence[IDataset]],
    *rest_datasets: IDataset
) -> IDataset:
    """ Compose datasets with a cartesian product.

    This will produce a dataset containing all combinations of data.
    Example: For two sets [1,2], ['a','b'] it produces
    [(1,'a'), (2,'a'), (1,'b'), (2,'b'),].

    The resulting dataset will have a length equal to the product of
    the length of the parent datasets.
    Datasets must have the same number of elements.
    The names of the first dataset (self) are keps.
    Any computed statistics are discarded.

    Arguments:
        dataset_or_datasets {Union[IDataset, Sequence[IDataset]]} -- Either a single dataset or a list of them
        rest_dataset {IDataset} -- remaining arguments may be additional datasets

    Returns:
        IDataset
    """
    datasets = collect_datasets(self, dataset_or_datasets, *rest_datasets)
    compose = CartesianProductProvider(datasets)
    return Dataset(
        parent=compose,
        sample_ids=range(len(compose)),
        operation_name="cartesian_product_wrapper",
        elem_name2index=zipped_elem_name2index(datasets),
        stats=zipped_stats(datasets),
    )
