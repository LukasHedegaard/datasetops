from datasetops.abstract import ItemGetter
from datasetops.dataset import Dataset
from datasetops.types import *
import numpy as np


class FunctionDataset(Dataset):
    def __init__(
        self, getdata: Callable[[Any], Any], name: str = None,
    ):
        if not callable(getdata):
            raise TypeError("get_data should be callable")

        class Getter(ItemGetter):
            def __getitem__(self, i: int):
                return getdata(i)

        super().__init__(downstream_getter=Getter(), name=name)
    def _append(self, identifier: Data):
        self._ids.append(identifier)
    def _extend(self, ids: Union[List[Data], np.ndarray]):
        self._ids.extend(list(ids))
