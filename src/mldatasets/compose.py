from mldatasets.abstract import AbstractDataset
from mldatasets.types import *
import numpy as np
from typing import Union


class ZipDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        """ Compose datasets by zipping and flattening their items. 
            The resulting dataset will have a length equal to the shortest of provided datasets
        
        Arguments:
            downstream_datasets {[AbstractDataset]} -- Comma-separated datasets
        """
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

    
    @property
    def shape(self) -> Sequence[int]:
        return tuple([ dim 
            for s in self._downstream_item_shapes 
                for dim in s ])



class CartesianProductDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        pass



class ConcatDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        pass



class InterleaveDataset(AbstractDataset):
    def __init__(self, *downstream_datasets:AbstractDataset):
        pass

