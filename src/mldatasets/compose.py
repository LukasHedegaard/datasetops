from mldatasets.abstract import AbstractDataset
from mldatasets.types import *
import numpy as np
from typing import Union
import functools
import math 


class ZipDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        """ Compose datasets by zipping and flattening their items. 
            The resulting dataset will have a length equal to the shortest of provided datasets
        
        Arguments:
            downstream_datasets {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(downstream_datasets) == 0:
            raise ValueError("No datasets given to zip")
        self._downstream_datasets = downstream_datasets
        self._downstream_item_shapes = [ds.shape for ds in downstream_datasets]
        self._ids = list(range(self.__len__()))
        self._classwise_id_inds = {None: list(range(self.__len__()))}
        self.name = "zipped{}".format([ds.name for ds in self._downstream_datasets ])


    def __len__(self) -> int:
        return min([len(ds) for ds in self._downstream_datasets])

    
    def __getitem__(self, idx:int) -> Tuple:
        return tuple([ elem 
            for ds in self._downstream_datasets 
                for elem in ds[idx] ])

    
    # @property
    # def shape(self) -> Sequence[int]:
    #     return tuple([ dim 
    #         for s in self._downstream_item_shapes 
    #             for dim in s ])



class CartesianProductDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        """ Compose datasets with a cartesian product. 
            This will produce a dataset containing all combinations of data. 
            For two sets [1,2], ['a','b'] it produces [(1,'a'), (1,'b'), (2,'a'), (2,'b'),]. 
            The resulting dataset will have a length equal to the product of the length of the downstream datasets.
        
        Arguments:
            downstream_datasets {[AbstractDataset]} -- Comma-separated datasets
        """
        if len(downstream_datasets) == 0:
            raise ValueError("No datasets given to zip")
        self._downstream_datasets = downstream_datasets
        self._downstream_item_shapes = [ds.shape for ds in downstream_datasets]
        self._downstream_lengths = [len(ds) for ds in downstream_datasets]
        self._ids = list(range(self.__len__()))
        self._classwise_id_inds = {None: list(range(self.__len__()))}
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

        # i0 = math.floor(idx/1) % len(ds0)
        # i1 = math.floor(idx/len(ds0)) % len(ds1)
        # i2 = math.floor(idx/(len(ds1) * len(ds1))) % len(ds2)

        return tuple([ elem 
            for i, ds in enumerate(self._downstream_datasets)
                for elem in ds[inds[i]] ])

    
    # @property
    # def shape(self) -> Sequence[int]:
    #     return tuple([ dim 
    #         for s in self._downstream_item_shapes 
    #             for dim in s ])



class ConcatDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        """ Compose datasets by concatenating them, placing one after the other. 
            The resulting dataset will have a length equal to the sum of datasets
        
        Arguments:
            downstream_datasets {[AbstractDataset]} -- Comma-separated datasets
        """
        pass # pragma: no cover


class InterleaveDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        pass # pragma: no cover

