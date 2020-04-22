from datasetops.xtypes import AbstractDataset
import numpy as np
import functools
import math
import warnings
from typing import Tuple


# ========= Utils =========


def _zipped_item_names(*datasets: AbstractDataset):
    if set.intersection(*[set(d.names) for d in datasets]):  # type:ignore
        return None
    else:
        return {
            n: i
            for i, n in enumerate(
                [n for d in datasets for n in d.names]  # type:ignore
            )
        }


# ========= Datasets =========


class ZipDataset(AbstractDataset):
    def __init__(self, *parents: AbstractDataset):
        """ Compose datasets by zipping and flattening their items.
        The resulting dataset will have a length equal to the shortest
        of provided datasets.
        NB: Any class-specific information will be lost after this transformation,
        and methods such as classwise_subsampling will not work.

        Arguments:
            parents {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(parents) == 0:
            raise ValueError("No datasets given to compose")
        self._parents = parents
        self._ids = list(range(self.__len__()))
        self.name = "zipped{}".format([ds.name for ds in self._parents])
        self._item_names = _zipped_item_names(*parents)
        self._cacheable = all(ds._cacheable for ds in self._parents)
        self._origin = list(
            map(
                lambda ds: {"dataset": ds, "operation": {"name": "zip"}}, self._parents,
            )
        )

    def __len__(self) -> int:
        return min([len(ds) for ds in self._parents])

    def __getitem__(self, idx: int) -> Tuple:
        return tuple([elem for ds in self._parents for elem in ds[idx]])


class CartesianProductDataset(AbstractDataset):
    def __init__(self, *parents: AbstractDataset):
        """Compose datasets with a cartesian product.

        This will produce a dataset containing all combinations of data.
        Example: For two sets [1,2], ['a','b'] it produces
        [(1,'a'), (2,'a'), (1,'b'), (2,'b'),].
        The resulting dataset will have a length equal to the product of
        the length of the downstream datasets.

        NB: Any class-specific information will be lost after this transformation,
        and methods such as classwise_subsampling will not work.

        Arguments:
            parents {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(parents) == 0:
            raise ValueError("No datasets given to compose")
        self._parents = parents
        self._downstream_lengths = [len(ds) for ds in parents]
        self._ids = list(range(self.__len__()))
        self.name = "cartesian_product{}".format([ds.name for ds in self._parents])
        self._item_names = _zipped_item_names(*parents)
        self._cacheable = all(ds._cacheable for ds in self._parents)
        self._origin = list(
            map(
                lambda ds: {"dataset": ds, "operation": {"name": "cartesian_product"}},
                self._parents,
            )
        )

    def __len__(self) -> int:
        return int(
            functools.reduce(lambda acc, ds: acc * len(ds), self._parents, int(1))
        )

    def __getitem__(self, idx: int) -> Tuple:
        acc_len = functools.reduce(
            lambda acc, ds: acc + [acc[-1] * len(ds)], self._parents[:-1], [int(1)],
        )
        inds = [
            math.floor(idx / al) % l for al, l in zip(acc_len, self._downstream_lengths)
        ]

        return tuple(
            [elem for i, ds in enumerate(self._parents) for elem in ds[inds[i]]]
        )


class ConcatDataset(AbstractDataset):
    def __init__(self, *parents: AbstractDataset):
        """Compose datasets by concatenating them, placing one after the other.
        The resulting dataset will have a length equal to the sum of datasets.

        Arguments:
            parents {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(parents) == 0:
            raise ValueError("No datasets given to compose")

        for i in range(len(parents) - 1):
            if parents[i].shape != parents[i + 1].shape:
                warnings.warn(
                    (
                        "Concatenating datasets with different element shapes "
                        "constitutes undefined behavior"
                    )
                )

        self._parents = parents
        self._ids = list(range(self.__len__()))
        self.name = "concat{}".format(
            [ds.name for ds in self._parents]  # type:ignore
        )
        self._acc_idx_range = functools.reduce(
            lambda acc, ds: acc + [len(ds) + acc[-1]], self._parents, [0]
        )
        self._cacheable = all(ds._cacheable for ds in self._parents)
        self._origin = list(
            map(
                lambda ds: {"dataset": ds, "operation": {"name": "concat"}},
                self._parents,
            )
        )

    def __len__(self) -> int:
        return sum([len(ds) for ds in self._parents])

    def __getitem__(self, idx: int) -> Tuple:
        dataset_index = (
            np.where(np.array(self._acc_idx_range) > idx)[0][0] - 1  # type:ignore
        )
        index_in_dataset = idx - self._acc_idx_range[dataset_index]
        return self._parents[dataset_index][index_in_dataset]
