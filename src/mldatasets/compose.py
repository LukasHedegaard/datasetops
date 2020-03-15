from mldatasets.abstract import AbstractDataset
from mldatasets.types import *
import numpy as np
from typing import Union
import functools
import math 
import warnings


class ZipDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        """ Compose datasets by zipping and flattening their items. 
            The resulting dataset will have a length equal to the shortest of provided datasets
            NB: Any class-specific information will be lost after this transformation, and methods such as classwise_subsampling will not work.
        
        Arguments:
            downstream_datasets {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(downstream_datasets) == 0:
            raise ValueError("No datasets given to compose")
        self._downstream_datasets = downstream_datasets
        self._ids = list(range(self.__len__()))
        self._classwise_id_inds = {None: self._ids}
        self.name = "zipped{}".format([ds.name for ds in self._downstream_datasets ])


    def __len__(self) -> int:
        return min([len(ds) for ds in self._downstream_datasets])

    
    def __getitem__(self, idx:int) -> Tuple:
        return tuple([ elem 
            for ds in self._downstream_datasets 
                for elem in ds[idx] ])



class CartesianProductDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        """ Compose datasets with a cartesian product. 
            This will produce a dataset containing all combinations of data. 
            Example: For two sets [1,2], ['a','b'] it produces [(1,'a'), (2,'a'), (1,'b'), (2,'b'),]. 
            The resulting dataset will have a length equal to the product of the length of the downstream datasets.
            NB: Any class-specific information will be lost after this transformation, and methods such as classwise_subsampling will not work.
        
        Arguments:
            downstream_datasets {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(downstream_datasets) == 0:
            raise ValueError("No datasets given to compose")
        self._downstream_datasets = downstream_datasets
        self._downstream_lengths = [len(ds) for ds in downstream_datasets]
        self._ids = list(range(self.__len__()))
        self._classwise_id_inds = {None: self._ids}
        self.name = "cartesian_product{}".format([ds.name for ds in self._downstream_datasets ])


    def __len__(self) -> int:
        return int(functools.reduce(lambda acc,ds: acc*len(ds), self._downstream_datasets, int(1)))

    
    def __getitem__(self, idx:int) -> Tuple:
        acc_len = functools.reduce(
            lambda acc, ds: acc+[acc[-1]*len(ds)], 
            self._downstream_datasets[:-1], 
            [int(1)]
        )
        inds = [ 
            math.floor(idx/al) % l
            for al, l in zip(acc_len, self._downstream_lengths)
        ]

        return tuple([ elem 
            for i, ds in enumerate(self._downstream_datasets)
                for elem in ds[inds[i]] ])



class ConcatDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        """ Compose datasets by concatenating them, placing one after the other. 
            The resulting dataset will have a length equal to the sum of datasets.
        
        Arguments:
            downstream_datasets {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(downstream_datasets) == 0:
            raise ValueError("No datasets given to compose")

        for i in range(len(downstream_datasets)-1):
            if downstream_datasets[i].shape != downstream_datasets[i+1].shape: # type:ignore
                warnings.warn('Concatenating datasets with different element shapes constitutes undefined behavior')

        self._downstream_datasets = downstream_datasets

        self._ids = list(range(self.__len__()))

        self._acc_idx_range = functools.reduce(
            lambda acc, ds: acc + [len(ds)+acc[-1]],
            self._downstream_datasets,
            [0]
        )

        # make class-wise ids
        cwids = {}
        for i, ds in enumerate(self._downstream_datasets):
            new_cwids = {
                key:list(np.array(vals) + self._acc_idx_range[i]) 
                for key, vals in ds._classwise_id_inds.items() #type: ignore
            } 
            for key, vals in new_cwids.items():
                if not key in cwids:
                    cwids[key] = vals
                else:
                    cwids[key].extend(vals)

        self._classwise_id_inds = cwids

        self.name = "concat{}".format([ds.name for ds in self._downstream_datasets ])


    def __len__(self) -> int:
        return sum([len(ds) for ds in self._downstream_datasets])

    
    def __getitem__(self, idx:int) -> Tuple:
        dataset_index = np.where(np.array(self._acc_idx_range) > idx)[0][0]-1 #type:ignore
        index_in_dataset = idx - self._acc_idx_range[dataset_index]
        return self._downstream_datasets[dataset_index][index_in_dataset]



class InterleaveDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        pass # pragma: no cover

