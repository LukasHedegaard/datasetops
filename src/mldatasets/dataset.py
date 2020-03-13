import random
from mldatasets.abstract import ItemGetter, AbstractDataset
from mldatasets.types import *
# import mldatasets.transforms as tfm
import numpy as np
from PIL import Image
import warnings
import functools
from inspect import signature
from typing import Callable, Dict, Sequence, Union, Any, Optional, List, Tuple, Type, TypeVar
from mldatasets.abstract import AbstractDataset
from pathlib import Path

########## Types ####################
Shape = Sequence[int]
IdIndex = int
Id = int
Ids = List[Id]
Data = Any
IdIndexSet = Dict[Any, List[IdIndex]]
ItemTransformFn = Callable[[Any],Any]
DatasetTransformFn = Callable[[int, AbstractDataset], AbstractDataset] 
DatasetTransformFnCreator = Callable[[Any], DatasetTransformFn]
AnyPath = Union[str, Path]

########## Defaults ####################

_DEFAULT_SHAPE = tuple()


def warn_no_args(skip=0):
    def with_args(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if len(args) + len(kwargs) <= skip:
                warnings.warn('Too few args passed to {}'.format(fn.__code__))
            return fn(*args, **kwargs)
        return wrapped
    return with_args

########## Dataset ####################

class Dataset(AbstractDataset):
    """ Contains information on how to access the raw data, and performs sampling and splitting related operations.
        Notes on internal data representation:
            self._ids contains the identifiers that are use to grab the downstream data
            self._classwise_id_inds are the classwise sorted indexes of self._ids (not the ids themselves)
            naming: an `item` is a single datapoint, containing multiple `elements` (e.g. np.array and label)
    """

    def __init__(self, 
        downstream_getter:Union[ItemGetter,'Dataset'], 
        name:str=None, 
        ids:Ids=None, 
        classwise_id_refs:IdIndexSet=None, 
        item_transform_fn:ItemTransformFn=lambda x: x
    ):
        """Initialise
        
        Keyword Arguments:
            downstream_getter {ItemGetter} -- Any object which implements the __getitem__ function (default: {None})
            name {str} -- A name for the dataset
            ids {Ids} -- List of ids used in the downstream_getter (default: {None})
            classwise_id_refs {IdIndexSet} -- Classwise sorted indexes of the ids, NB: not the ids, but their indexes (default: {None})
            item_transform_fn: {Calleable} -- a function
        """
        self._downstream_getter = downstream_getter
        
        if type(downstream_getter) == Dataset or issubclass(type(downstream_getter), Dataset):
            self.name = self._downstream_getter.name                    #type: ignore
            self._ids = list(range(len(self._downstream_getter._ids)))  #type: ignore
            self._classwise_id_inds = self._make_classwise_id_ids(self._downstream_getter._classwise_id_inds, self._ids)  #type: ignore
        else:
            self.name = ''
            self._ids = []
            self._classwise_id_inds = {}

        if name:
            self.name = name

        self._item_transform_fn = item_transform_fn

        self._shape = None
        self._item_names: Optional[Dict[str,int]] = None 

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

        if len(self) == 0:
            return _DEFAULT_SHAPE
        
        item = self.__getitem__(0)
        if hasattr(item, '__getitem__'):
            item_shape = []
            for i in item:
                if hasattr(i, 'shape'): # numpy arrays
                    item_shape.append(i.shape)
                elif hasattr(i, 'size'): # PIL.Image.Image
                    item_shape.append(np.array(i).shape)
                else:
                    item_shape.append(_DEFAULT_SHAPE)

            return tuple(item_shape)

        return _DEFAULT_SHAPE

    

    def class_names(self) -> List[str]:
        return [self._label2name(k) for k in self._classwise_id_inds.keys()]


    def class_counts(self) -> Dict[str, int]:
        return {
            self._label2name(k): len(v)
            for k,v in self._classwise_id_inds.items()
        }


    def sample(self, num:int, seed:int=None):
        if seed:
            random.seed(seed)

        new_ids = random.sample(range(len(self)), num)

        new_classwise_id_inds = self._make_classwise_id_ids(self._classwise_id_inds, new_ids)

        return Dataset(downstream_getter=self, ids=new_ids, classwise_id_refs=new_classwise_id_inds)


    def sample_classwise(self, num_per_class:int, seed:int=None):
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


    def set_item_names(self, first:Union[str, Sequence[str]], *rest:str):
        
        names:List[str] = []

        if type(first) == str:
            names.append(first) #type: ignore
        else:
            assert(type(first) is list)
            assert(type(first[0]) is str)
            names = first #type: ignore

        names.extend(rest)

        assert(len(names) <= len(self.shape))
        self._item_names = {
            n: i
            for i, n in enumerate(names)
        }

    @property
    def item_names(self) -> List[str]:
        if self._item_names:
            return list(self._item_names.keys())
        else:
            return []


    def _itemname2ind(self, name:str) -> int:
        if not self._item_names:
            raise ValueError("Items cannot be identified by name when no names are given. Hint: Use `Dataset.set_item_names('name1', 'name2', ...)`")
        return self._item_names[name]
    
    @warn_no_args(skip=1)
    def transform(self, *fns:DatasetTransformFn, **kwfns:DatasetTransformFn):

        if len(fns) + len(kwfns) > len(self.shape): # type:ignore
            raise ValueError("More transforms ({}) given than can be performed on item with {} elements".format(len(fns) + len(kwfns), len(self.shape)))

        new_dataset: AbstractDataset = self

        for i, f in enumerate(fns): #type:ignore
            if f:
                # if user passed a function with a single argument, wrap it
                if len(signature(f).parameters) == 1:
                    f = custom(f) #type:ignore
                new_dataset = f(i, new_dataset)

        for k, f in kwfns.items():
            if f:
                new_dataset = f(self._itemname2ind(k), new_dataset)

        return new_dataset


    ########## Conversion methods #########################

    def as_image(self, *positional_flags:Any):
        if len(positional_flags) == 0:
            # convert all that can be converted
            positional_flags = []
            for elem in self.__getitem__(0):
                try:
                    _check_image_compatibility(elem)
                    positional_flags.append(True) # didn't raise error
                except:
                    positional_flags.append(False)
                   
        if any(positional_flags):
            return self._optional_argument_indexed_transform(transform_fn=as_image, args=positional_flags)       
        else: 
            warnings.warn('Conversion to image skipped. No elements were compatible')
            return self

    
    def as_numpy(self, *positional_flags:Any):
        if len(positional_flags) == 0:
            # convert all that can be converted
            positional_flags = []
            for elem in self.__getitem__(0):
                try:
                    _check_numpy_compatibility(elem)
                    positional_flags.append(True) # didn't raise error
                except:
                    positional_flags.append(False)
                   
        if any(positional_flags):
            return self._optional_argument_indexed_transform(transform_fn=as_numpy, args=positional_flags)       
        else: 
            warnings.warn('Conversion to numpy.array skipped. No elements were compatible')
            return self


    ########## Methods relating to numpy data #########################
    
    def _optional_argument_indexed_transform(self, transform_fn:DatasetTransformFnCreator, args:Tuple[Any]):
        if len(args) == 0:
            raise ValueError("Unable to perform transform: No arguments arguments given")
        if len(self.shape) < len(args):
            raise ValueError("Unable to perform transform: Too many arguments given")

        tfs = [
            transform_fn(a) if a else None 
            for a in args
        ]
        return self.transform(*tfs) #type:ignore


    def reshape(self, *new_shapes:Optional[Shape]):
        return self._optional_argument_indexed_transform(transform_fn=reshape, args=new_shapes) 


    # def scale(self, scaler):
    #     pass


    # def add_noise(self, noise):
    #     pass

    ########## Methods below assume data is an image ##########

    def img_resize(self, *new_sizes:Optional[Shape]):
        return self._optional_argument_indexed_transform(transform_fn=img_resize, args=new_sizes) 

    # def img_transform(self, transform):
    #     pass

    # def img_filter(self, filter_fn):
    #     pass


########## Handy decorators ####################

def _dataset_element_transforming(fn:Callable, check:Callable=None):
    """ Applies the function to dataset item elements """

    # @functools.wraps(fn)
    def wrapped(idx:int, ds:AbstractDataset) -> AbstractDataset:

        if check:
            # grab an item and check its elements
            for i, elem in enumerate(ds[0]):
                if i == idx:
                    check(elem)

        def item_transform_fn(item:Sequence):
            return tuple([
                fn(elem) if i == idx else elem
                for i, elem in enumerate(item)
            ])

        return Dataset(
            downstream_getter=ds, 
            item_transform_fn=item_transform_fn,
        )

    return wrapped


def _check_shape_compatibility(shape:Shape):
    def check(elem):
        if not hasattr(elem, 'shape'):
            raise ValueError('{} needs a shape attribute for shape compatibility to be checked'.format(elem))

        if (
            np.prod(elem.shape) != np.prod(shape) #type: ignore
            and not (-1 in shape) 
        ) or any([s > np.prod(elem.shape) for s in shape]) :
            raise ValueError("Cannot reshape dataset with shape '{}' to shape '{}'. Dimensions cannot be matched".format(elem.shape, shape))

    return check


def convert2img(elem:Union[Image.Image, str, Path, np.ndarray]) -> Image.Image:
    if issubclass(type(elem), Image.Image):
        return elem
    
    if type(elem) in [str, Path]:
        if Path(elem).is_file(): #type: ignore
            return Image.open(elem)

    if type(elem) == np.ndarray:
        if issubclass(elem.dtype.type, np.integer):     # type:ignore
            return Image.fromarray(np.uint8(elem))      # type:ignore
        elif issubclass(elem.dtype.type, np.floating):  # type:ignore
            return Image.fromarray(np.float32(elem))    # type:ignore

    raise ValueError("Unable to convert element {} to Image".format(elem))


def _check_image_compatibility(elem):
    # check if this raises an Exception
    convert2img(elem)


def _check_numpy_compatibility(elem):
    # skip simple datatypes such as int and float as well
    if type(elem) in [dict, str, int, float]:
        raise ValueError("Unable to convert element {} to numpy".format(elem))
    # check if this raises an Exception
    np.array(elem)


########## Transform implementations ####################

def reshape(new_shape:Shape) -> DatasetTransformFn:
    return _dataset_element_transforming(
        fn=functools.partial(np.reshape, newshape=(new_shape)), #type: ignore
        check=_check_shape_compatibility(new_shape)
    )


def custom(elem_transform_fn:Callable[[Any], Any], elem_check_fn:Callable[[Any],None]=None) -> DatasetTransformFn:
    """ Create a user defined transform
    
    Arguments:
        fn {Callable[[Any], Any]} -- A user defined function, which takes the element as only argumnet
    
    Keyword Arguments:
        check_fn {Callable[[Any]]} -- A function that raises an Exception if the elem is incompatible (default: {None})
    
    Returns:
        DatasetTransformFn -- [description]
    """
    return _dataset_element_transforming(
        fn=elem_transform_fn, 
        check=elem_check_fn
    )


def as_image(dummy_input=None) -> DatasetTransformFn:
    return _dataset_element_transforming(
        fn=convert2img,
        check=_check_image_compatibility
    )


def as_numpy(dummy_input=None) -> DatasetTransformFn:
    return _dataset_element_transforming(
        fn=np.array,
        check=_check_numpy_compatibility
    )


def img_resize(new_size:Shape, resample=Image.NEAREST) -> DatasetTransformFn:
    assert(len(new_size)==2)
    return _dataset_element_transforming(
        fn=lambda x: convert2img(x).resize(size=new_size, resample=resample),
        check=_check_image_compatibility
    )