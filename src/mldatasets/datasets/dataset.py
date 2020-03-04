import random
from mldatasets.datasets.base import ItemGetter, BaseDataset
from typing import Callable, Dict, Union, Any, Optional, List, Tuple, Type, TypeVar

# types
IdIndex = int
Id = int
Data = Any
IdIndexSet = Dict[Any, List[IdIndex]]

class Dataset(BaseDataset):
    """ Contains information on how to access the raw data, and performs sampling and splitting related operations
        Tricky implementation notes:
            self._ids contains the identifiers that are use to grab the downstream data
            self._classwise_id_inds are the classwise sorted indexes of self._ids (not the ids themselves)
    """

    def __init__(self, downstream_getter:ItemGetter, ids:List[Id]=None, classwise_id_refs:IdIndexSet=None):
        """Initialise
        
        Keyword Arguments:
            downstream_getter {ItemGetter} -- Any object which implements the __getitem__ function (default: {None})
            ids {List[Id]} -- List of ids used in the downstream_getter (default: {None})
            classwise_id_refs {IdIndexSet} -- Classwise sorted indexes of the ids, NB: not the ids, but their indexes (default: {None})
        """
        self._downstream_getter = downstream_getter
        self._ids = []
        self._classwise_id_inds = {}

        if ids and classwise_id_refs:
            self._ids:List[Id] = ids
            self._classwise_id_inds:IdIndexSet = classwise_id_refs


    def __len__(self):
        return len(self._ids)


    def __getitem__(self, i: int):
        return self._downstream_getter[self._ids[i]] 

    
    # def _set_ids(self, ids: List[Id], classwise_id_refs: IdIndexSet):
    #     self._ids = ids
    #     self._classwise_id_inds = classwise_id_refs
    

    def _append(self, identifier:Data, label:Optional[str]=None):
        i_new = len(self._ids)

        self._ids.append(identifier)

        if not label in self._classwise_id_inds:
            self._classwise_id_inds[label] = [i_new]
        else:
            self._classwise_id_inds[label].append(i_new)


    def _extend(self, ids:List[Data], label:Optional[str]=None):
        i_lo = len(self._ids)
        i_hi = i_lo + len(ids)
        l_new = list(range(i_lo, i_hi))

        self._ids.extend(ids)

        if not label in self._classwise_id_inds:
            self._classwise_id_inds[label] = l_new
        else:
            self._classwise_id_inds[label].extend(l_new)


    @staticmethod
    def _label2name(label:Any) -> str:
        return str(label)


    def class_names(self) -> List[str]:
        return [self._label2name(k) for k in self._classwise_id_inds.keys()]


    def class_counts(self) -> Dict[str, int]:
        return {
            self._label2name(k): len(v)
            for k,v in self._classwise_id_inds.items()
        }


    def sample(self, num: int, seed:int=None):
        if seed:
            random.seed(seed)

        new_ids = random.sample(range(len(self)), num)

        # create list of ranges corresponding to classes
        ranges = {}
        prev = 0
        for k, v in self._classwise_id_inds.items():
            ranges[k] = (prev, prev+len(v))
            prev = len(v)

        # group the new samples items into class-buckets
        new_classwise_id_inds = {k:[] for k in self._classwise_id_inds.keys()}
        for i in enumerate(new_ids):
            for k, r in ranges.items():
                if r[0] <= i < r[1]:
                    new_classwise_id_inds[k].append(i)

        return Dataset(downstream_getter=self, ids=new_ids, classwise_id_refs=new_classwise_id_inds)


    def sample_classwise(self, num_per_class: int, seed:int=None):
        if seed:
            random.seed(seed)

        new_ids = []
        new_classwise_id_inds = {}
        prev_idx = 0
        for k, v in self._classwise_id_inds.items():
            next_idx = prev_idx+num_per_class
            ss = random.sample(v, num_per_class)
            new_ids.extend(ss)
            new_classwise_id_inds[k] = list(range(prev_idx, next_idx))
            prev_idx = next_idx

        return Dataset(downstream_getter=self, ids=new_ids, classwise_id_refs=new_classwise_id_inds)

