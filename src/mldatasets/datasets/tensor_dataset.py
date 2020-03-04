from mldatasets.datasets.abstract import ItemGetter
from mldatasets.datasets.dataset import Dataset, Ids, IdIndexSet
from typing import List, Callable, Union

class TensorDataset(Dataset):
    """ Dataset containing tensor information. 
        It extends the Dataset class with functions relating to scaling (standardization etc.)
        as well as noise addition
    """
    def __init__(self, downstream_getter:ItemGetter, ids:Ids=None, classwise_id_refs:IdIndexSet=None, item_transform_fn:Callable=None):
        super().__init__(downstream_getter, ids, classwise_id_refs, item_transform_fn)

    def scale(self, scaler):
        pass

    def add_noise(self, noise):
        pass