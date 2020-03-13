from mldatasets.abstract import ItemGetter
from mldatasets.dataset import Dataset
from mldatasets.types import *
import numpy as np

class FunctionDataset(Dataset):

    def __init__(self, 
        getdata:Callable[[Any], Any], 
        name:str=None,
    ):
        if not callable(getdata):
            raise TypeError("get_data should be callable")

        class Getter(ItemGetter):
            def __getitem__(self, i:int):
                return getdata(i)

        super().__init__(downstream_getter=Getter(), name=name)


    def _append(self, identifier:Data, label:Optional[str]=None):
        i_new = len(self._ids)

        self._ids.append(identifier)

        if not label in self._classwise_id_inds:
            self._classwise_id_inds[label] = [i_new]
        else:
            self._classwise_id_inds[label].append(i_new)


    def _extend(self, ids:Union[List[Data], np.ndarray], label:Optional[str]=None):
        i_lo = len(self._ids)
        i_hi = i_lo + len(ids)
        l_new = list(range(i_lo, i_hi))

        self._ids.extend(list(ids))

        if not label in self._classwise_id_inds:
            self._classwise_id_inds[label] = l_new
        else:
            self._classwise_id_inds[label].extend(l_new)

