import random
from mldatasets.datasets.abstract import ItemGetter, AbstractDataset
from typing import Callable, Dict, Sequence, Union, Any, Optional, List, Tuple, Type, TypeVar
import numpy as np
import warnings
import functools

# types
Shape = Sequence[int]
IdIndex = int
Id = int
Ids = List[Id]
Data = Any
IdIndexSet = Dict[Any, List[IdIndex]]
TransformFn = Callable[[int, AbstractDataset], AbstractDataset] 

_DEFAULT_SHAPE = (1,)

class Dataset(AbstractDataset):
    """ Contains information on how to access the raw data, and performs sampling and splitting related operations.
        Notes on internal data representation:
            self._ids contains the identifiers that are use to grab the downstream data
            self._classwise_id_inds are the classwise sorted indexes of self._ids (not the ids themselves)
    """

    def __init__(self, downstream_getter:ItemGetter, ids:Ids=None, classwise_id_refs:IdIndexSet=None, item_transform_fn:Callable=None, name:str=None):
        """Initialise
        
        Keyword Arguments:
            downstream_getter {ItemGetter} -- Any object which implements the __getitem__ function (default: {None})
            ids {Ids} -- List of ids used in the downstream_getter (default: {None})
            classwise_id_refs {IdIndexSet} -- Classwise sorted indexes of the ids, NB: not the ids, but their indexes (default: {None})
        """
        self._downstream_getter = downstream_getter
        self._ids = []
        self._classwise_id_inds = {}
        self._item_transform_fn = item_transform_fn or (lambda x: x)
        self._shape = None
        self.name = name
        self._item_names = None 

        if ids and classwise_id_refs:
            self._ids:Ids = ids
            self._classwise_id_inds:IdIndexSet = classwise_id_refs


    def __len__(self):
        return len(self._ids)


    def __getitem__(self, i: int):
        return self._item_transform_fn(self._downstream_getter[self._ids[i]])


    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


    @property
    def shape(self):
        if not self._shape:
            item = self.__getitem__(0)
            if hasattr(item, '__getitem__'):
                item_shape = tuple([getattr(s, "shape", _DEFAULT_SHAPE) for s in item])
            else:
                item_shape = _DEFAULT_SHAPE
            self._shape = item_shape

        return self._shape
    

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

        new_classwise_id_inds = self._make_classwise_id_ids(self._classwise_id_inds, new_ids)

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


    def shuffle(self, seed=None):
        if seed:
            random.seed(seed)

        new_ids = list(range(len(self)))
        random.shuffle(new_ids)

        new_classwise_id_inds = self._make_classwise_id_ids(self._classwise_id_inds, new_ids)

        return Dataset(downstream_getter=self, ids=new_ids, classwise_id_refs=new_classwise_id_inds)

    
    def split(self, fractions:List[float], seed:int=None): 
        """ Split dataset into multiple datasets, determined by the fractions given.
            A wildcard (-1) may be given at a single position, to fill in the rest.
            If fractions don't add up, the last fraction in the list receives the remainding data.
        
        Arguments:
            fractions {List[float]} -- a list or tuple of floats i the interval ]0,1[. One of the items may be a -1 wildcard.
        
        Keyword Arguments:
            seed {int} -- Random seed (default: {None})
        
        Returns:
            List[Dataset] -- Datasets with the number of samples corresponding to the fractions given
        """
        if seed:
            random.seed(seed)

        # max one entry is '-1'
        assert(len(list(filter(lambda x: x==-1, fractions))) <=1)

        # replace wildcard with proper fraction
        if -1 in fractions:
            wildcard = -sum(fractions)
            fractions = [x if x != -1 else wildcard for x in fractions]

        # create shuffled list
        new_ids = list(range(len(self)))
        random.shuffle(new_ids)

        # split according to fractions
        split_ids = [[] for _ in fractions]
        last_ind = 0
        for i, f in enumerate(fractions):
            next_last_ind = last_ind + round(f*len(new_ids)) 
            if i != len(fractions)-1:
                split_ids[i].extend(new_ids[last_ind:next_last_ind])
                last_ind = next_last_ind
            else:
                split_ids[i].extend(new_ids[last_ind:])

        # create datasets corresponding to each split
        datasets = [
            Dataset(
                downstream_getter=self, 
                ids=new_ids, 
                classwise_id_refs=self._make_classwise_id_ids(self._classwise_id_inds, new_ids)
            )
            for new_ids in split_ids
        ]

        return datasets


    def _set_item_transform(self, transform_fn: Callable):
        self._item_transform_fn = transform_fn
    

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


    @staticmethod
    def _label2name(label:Any) -> str:
        return str(label)


    @staticmethod
    def _make_classwise_id_ids(old_classwise_id_inds: IdIndexSet, new_ids: List[IdIndex]) -> IdIndexSet:
        # create list of ranges corresponding to classes
        ranges = {}
        prev = 0
        for k, v in old_classwise_id_inds.items():
            ranges[k] = (prev, prev+len(v))
            prev = len(v)

        # group the new samples items into class-buckets
        new_classwise_id_inds = {k:[] for k in old_classwise_id_inds.keys()}
        for i, v in enumerate(new_ids):
            for k, r in ranges.items():
                if r[0] <= v < r[1]:
                    new_classwise_id_inds[k].append(i)

        return new_classwise_id_inds


    def set_item_names(self, *names: str):
        assert(len(names) <= len(self.shape))
        self._item_names = {
            n: i
            for i, n in enumerate(names)
        }


    def _itemname2ind(self, name:str) -> int:
        if not self._item_names:
            raise ValueError("Items cannot be identified by name when no names are given. Hint: Use `Dataset.set_item_names('name1', 'name2', ...)`")
        return self._item_names[name]
    

    ########## Methods relating to numpy data #########################

    def transform(self, *fns: TransformFn, **kwfns: TransformFn):

        new_dataset: AbstractDataset = self

        for i, f in enumerate(fns):
            new_dataset = f(i, new_dataset)

        for k, f in kwfns.items():
            new_dataset = f(self._itemname2ind(k), new_dataset)

        return new_dataset

    
    def reshape(self, *new_shapes:Optional[Shape]):
        """ Reshape the data
        
        Raises:
            ValueError: If no new_shapes are given
            ValueError: If too many new shapes are given
            ValueError: If shapes cannot be matched
        
        Returns:
            TensorDataset -- Dataset with reshaped elements
        """
        if len(new_shapes) > len(self.shape):
            raise ValueError("Cannot reshape dataset with shape '{}' to shape '{}'. Too many input shapes given".format(self.shape, new_shapes))

        if len(new_shapes) == 0:
            raise ValueError("Cannot reshape dataset with shape '{}' to shape '{}'. No target shape given".format(self.shape, new_shapes))

        transform_fns = []

        for old_shape, new_shape in zip(self.shape, new_shapes):
            if new_shape is None:
                transform_fns.append(lambda x: x)
                continue
            
            if np.prod(old_shape) != np.prod(new_shape) and not (-1 in new_shape): #type:ignore
                raise ValueError("Cannot reshape dataset with shape '{}' to shape '{}'. Dimensions cannot be matched".format(old_shape, new_shape))
            
            transform_fns.append(functools.partial(np.reshape, newshape=new_shape))

        # ensure that len(transform_fns) == len(self.shape)
        for _ in range(len(self.shape)-len(transform_fns)):
            transform_fns.append(lambda x: x)

        def item_transform_fn(item: Any):
            nonlocal transform_fns
            transformed_item = tuple([
                fn(x) for fn, x in zip(transform_fns, item)
            ])
            return transformed_item

        return Dataset(
            downstream_getter=self, 
            ids=self._ids, 
            classwise_id_refs=self._classwise_id_inds, 
            item_transform_fn=item_transform_fn,
            name=self.name
        )


    def scale(self, scaler):
        pass


    def add_noise(self, noise):
        pass

    ########## Methods below assume data is an image ##########

    def im_transform(self, transform):
        pass

    def im_filter(self, filter_fn):
        pass