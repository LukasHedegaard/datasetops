import random
from mldatasets.abstract import ItemGetter, AbstractDataset
from mldatasets.types import *
import mldatasets.compose as compose 
import numpy as np
from PIL import Image
import warnings
import functools
from inspect import signature
from typing import Callable, Dict, Sequence, Union, Any, Optional, List, Tuple, Type, TypeVar
from mldatasets.abstract import AbstractDataset
from pathlib import Path


########## Defaults ####################

_DEFAULT_SHAPE = tuple()


def _warn_no_args(skip=0):
    def with_args(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if len(args) + len(kwargs) <= skip:
                warnings.warn('Too few args passed to {}'.format(fn.__code__.co_name))
            return fn(*args, **kwargs)
        return wrapped
    return with_args


def _raise_no_args(skip=0):
    def with_args(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if len(args) + len(kwargs) <= skip:
                raise ValueError('Too few args passed to {}'.format(fn.__code__.co_name))
            return fn(*args, **kwargs)
        return wrapped
    return with_args


def _dummy_arg_receiving(fn):
    @functools.wraps(fn)
    def wrapped(dummy, *args, **kwargs):
        return fn(*args, **kwargs)
    return wrapped


########## Dataset ####################

class Dataset(AbstractDataset):
    """ Contains information on how to access the raw data, and performs sampling and splitting related operations.
    """

    def __init__(self, 
        downstream_getter:Union[ItemGetter,'Dataset'], 
        name:str=None, 
        ids:Ids=None, 
        item_transform_fn:ItemTransformFn=lambda x: x,
        item_names:Dict[str,int]=None,
    ):
        """Initialise
        
        Keyword Arguments:
            downstream_getter {ItemGetter} -- Any object which implements the __getitem__ function (default: {None})
            name {str} -- A name for the dataset
            ids {Ids} -- List of ids used in the downstream_getter (default: {None})
            item_transform_fn: {Calleable} -- a function
        """
        self._downstream_getter = downstream_getter
        
        if issubclass(type(downstream_getter), AbstractDataset):
            self.name = self._downstream_getter.name                    #type: ignore
            self._ids = list(range(len(self._downstream_getter._ids)))  #type: ignore
            self._item_names = getattr(downstream_getter, "_item_names", None)
        else:
            self.name = ''
            self._ids = []
            self._item_names: Optional[Dict[str,int]] = None 

        if name:
            self.name = name
        if item_names:
            self._item_names = item_names
        if ids:
            self._ids:Ids = ids

        self._item_transform_fn = item_transform_fn
            

    def __len__(self):
        return len(self._ids)


    def __getitem__(self, i: int) -> Tuple:
        return self._item_transform_fn(self._downstream_getter[self._ids[i]])


    @property
    def shape(self) -> Sequence[int]:
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


    @functools.lru_cache(4)
    @_warn_no_args(skip=1)
    def counts(self, *itemkeys:Key) -> List[Tuple[Any,int]]:
        """ Compute the counts of each unique item in the dataset
            Warning: this operation may be expensive for large datasets
        
        Arguments:
            itemkeys {Union[str, int]} -- The item keys (str) or indexes (int) to be checked for uniqueness. If no key is given, all item-parts must match for them to be considered equal

        Returns:
            List[Tuple[Any,int]] -- List of tuples, each containing the unique value and its number of occurences
        """
        inds:List[int] = [k if type(k) == int else self._itemname2ind(k) for k in itemkeys] # type: ignore

        selector =  (lambda item: item          ) if len(itemkeys) == 0   else \
                    (lambda item: item[inds[0]] ) if len(inds) == 1       else \
                    (lambda item: tuple([val for i,val in enumerate(item) if i in inds]) )

        unique_items, item_counts = {}, {}
        for item in iter(self):
            selected = selector(item)
            h = hash(str(selected))
            if not h in unique_items.keys():
                unique_items[h] = selected
                item_counts[h] = 1
            else:
                item_counts[h] += 1
        
        return [(unique_items[k], item_counts[k]) for k in unique_items.keys()]
        

    @_warn_no_args(skip=1)
    def unique(self, *itemkeys:Key) -> List[Any]:
        """ Compute a list of unique values in the dataset
            Warning: this operation may be expensive for large datasets
        
        Arguments:
            itemkeys {str} -- The item keys to be checked for uniqueness

        Returns:
            List[Any] -- List of the unique items
        """
        return [x[0] for x in self.counts(*itemkeys)] 


    def sample(self, num:int, seed:int=None):
        if seed:
            random.seed(seed)
        new_ids = random.sample(range(len(self)), num)
        return Dataset(downstream_getter=self, ids=new_ids)


    def _combine_conditions(self, 
        bulk:DataPredicate=None, 
        itemwise:Sequence[Optional[DataPredicate]]=[], 
        **kwpredicates:DataPredicate
    ) -> DataPredicate:
        assert(len(itemwise) <= len(self.shape))
        assert(all([k in self.item_names for k in kwpredicates.keys()]))

        # clean up predicates
        if not bulk:
            bulk = lambda x: True
        preds:List[DataPredicate] = [((lambda x:True) if p is None else p) for p in itemwise] 

        def condition(x:Any) -> bool:
            return (
                bulk(x)
                and
                all([pred(x[i]) for i, pred in enumerate(preds)]) 
                and 
                all([pred(x[self._itemname2ind(k)]) for k, pred in kwpredicates.items()]) 
            )

        return condition


    @_warn_no_args(skip=1)
    def filter(self, bulk:DataPredicate=None, itemwise:Sequence[Optional[DataPredicate]]=[], **kwpredicates:DataPredicate):
        condition = self._combine_conditions(bulk, itemwise, **kwpredicates)
        new_ids = list(filter(lambda i: condition(self.__getitem__(i)), self._ids))
        return Dataset(downstream_getter=self, ids=new_ids)

    
    @_raise_no_args(skip=1)
    def filter_split(self, bulk:DataPredicate=None, itemwise:Sequence[Optional[DataPredicate]]=[], **kwpredicates:DataPredicate):
        condition = self._combine_conditions(bulk, itemwise, **kwpredicates)
        ack, nack = [], []
        for i in self._ids:
            if condition(self.__getitem__(i)):
                ack.append(i)
            else:
                nack.append(i)

        return tuple([ 
            Dataset(downstream_getter=self, ids=new_ids) 
            for new_ids in [ack, nack] 
        ])


    def shuffle(self, seed=None):
        random.seed(seed)
        new_ids = list(range(len(self)))
        random.shuffle(new_ids)
        return Dataset(downstream_getter=self, ids=new_ids)

    
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
        return tuple([
            Dataset(
                downstream_getter=self, 
                ids=new_ids, 
            )
            for new_ids in split_ids
        ])


    def take(self, num:int):
        """ Take the first elements of a datatset
        
        Arguments:
            num {int} -- number of elements to take
        
        Returns:
            Dataset -- A dataset with only the first `num` elements
        """
        if num > len(self):
            raise ValueError("Can't take more elements than are available in dataset")

        new_ids = list(range(num))
        return Dataset(downstream_getter=self, ids=new_ids)

    
    def repeat(self, repeats=1, mode='itemwise'):
        """ Repeat the dataset elements
        
        Keyword Arguments:
            repeats {int} -- Number of repeats for each element (default: {1})
            mode {str} -- Repeat 'itemwise' (i.e. [1,1,2,2,3,3]) or as a 'whole' (i.e. [1,2,3,1,2,3]) (default: {'itemwise'})
        
        Returns:
            [type] -- [description]
        """
        new_ids = {
            'whole'    : lambda : [i for _ in range(repeats) for i in list(range(len(self)))],
            'itemwise' : lambda : [i for i in list(range(len(self))) for _ in range(repeats)]
        }[mode]()

        return Dataset(downstream_getter=self, ids=new_ids)


    def reorder(self, *new_inds:Union[int,str]):
        """ Reorder items in the dataset (similar to numpy.transpose)

        Arguments:
            new_inds {Union[int,str]} -- positioned item index or key (if item names were previously set) of item
            
        Returns:
            [type] -- [description]
        """
        if len(new_inds) == 0:
            warnings.warn('No indexes given in Dataset.reorder. The dataset remains unchanged')
            return self

        inds:List[int] = []
        for i in new_inds:
            if type(i)==str:
                inds.append(self._itemname2ind(str(i))) 
            else:
                inds.append(int(i))

        for i in inds:
            if i > len(self.shape):
                raise ValueError("reoder index {} is out of range (maximum allowed index is {})".format(i, len(self.shape)))

        def item_transform_fn(item:Tuple):
            return tuple([item[i] for i in inds])

        item_names = None
        if self._item_names:
            if len(set(new_inds)) < len(new_inds):
                warnings.warn("discarding item_names due to otherwise non-unique labels on transformed dataset")
            else:
                item_names = {k:inds[v] for k,v in self._item_names.items() if v < len(inds)}

        return Dataset(
            downstream_getter=self, 
            item_transform_fn=item_transform_fn,
            item_names=item_names
        )


    @staticmethod
    def _label2name(label:Any) -> str:
        return str(label)


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
        return self


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
    

    @_warn_no_args(skip=1)
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

    ########## Label transforming methods #########################

    def label(self, key:Key, mapping_fn:Callable[[Any], int]=None ):
        idx:int = self._itemname2ind(key) if type(key) == str else key #type:ignore
        args = [mapping_fn or True if i == idx else None for i in range(idx+1)]
        return self._optional_argument_indexed_transform(transform_fn=label, args=args) 


    def one_hot(self, key:Key, encoding_size:int=None, mapping_fn:Callable[[Any], int]=None, dtype='bool'):
        enc_size = encoding_size or len(self.unique(key))
        idx:int = self._itemname2ind(key) if type(key) == str else key #type:ignore
        args = [enc_size if i == idx else None for i in range(idx+1)]
        return self._optional_argument_indexed_transform(
            transform_fn=functools.partial(one_hot, mapping_fn=mapping_fn, dtype=dtype), 
            args=args
        ) 

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
            return self._optional_argument_indexed_transform(transform_fn=_dummy_arg_receiving(as_image), args=positional_flags)       
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
            return self._optional_argument_indexed_transform(transform_fn=_dummy_arg_receiving(as_numpy), args=positional_flags)       
        else: 
            warnings.warn('Conversion to numpy.array skipped. No elements were compatible')
            return self


    ########## Composition methods #########################

    def zip(self,*datasets):
        return zipped(self, *datasets)

    
    def cartesian_product(self,*datasets):
        return cartesian_product(self, *datasets)

    
    def concat(self,*datasets):
        return concat(self, *datasets)


    ########## Methods relating to numpy data #########################
    
    def _optional_argument_indexed_transform(self, transform_fn:DatasetTransformFnCreator, args:Sequence[Any]):
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


########## Predicate implementations ####################

def allow_unique(max_num_duplicates=1):
    mem_counts = {}

    def fn(x):
        nonlocal mem_counts
        h = hash(str(x))
        if not h in mem_counts.keys():
            mem_counts[h]=1
            return True
        if mem_counts[h]<max_num_duplicates:
            mem_counts[h] += 1
            return True
        return False

    return fn


########## Transform implementations ####################

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


def reshape(new_shape:Shape) -> DatasetTransformFn:
    return _dataset_element_transforming(
        fn=functools.partial(np.reshape, newshape=(new_shape)), #type: ignore
        check=_check_shape_compatibility(new_shape)
    )


def label(mapping_fn:Callable[[Any], int]=None) -> DatasetTransformFn:
    """ Transform data into an integer label
    
    Arguments:
        mapping_fn {Callable[[Any], int]} -- A function transforming the input data to the integer label. If not specified, labels are automatically inferred from the data.
    
    Returns:
        DatasetTransformFn -- A function to be passed to the Dataset.transform()
    """
    mem, maxcount = {}, -1
    def auto_label(x:Any) -> int:
        nonlocal mem, maxcount
        h = hash(str(x)) 
        if not h in mem.keys():
            maxcount += 1
            mem[h] = maxcount
        return mem[h]

    return _dataset_element_transforming(
        fn=mapping_fn if callable(mapping_fn) else auto_label # type: ignore
    )


def one_hot(encoding_size:int, mapping_fn:Callable[[Any], int]=None, dtype='bool') -> DatasetTransformFn:
    """ Transform data into a one-hot encoded label
    
    Arguments:
        encoding_size {int} -- The size of the encoding
        mapping_fn {Callable[[Any], int]} -- A function transforming the input data to an integer label. If not specified, labels are automatically inferred from the data.
    
    Returns:
        DatasetTransformFn -- A function to be passed to the Dataset.transform()
    """
    mem, maxcount = {}, -1
    def auto_label(x:Any) -> int:
        nonlocal mem, maxcount, encoding_size
        h = hash(str(x)) 
        if not h in mem.keys():
            maxcount += 1
            if maxcount > encoding_size:
                raise ValueError("More unique labels found than were specified by the encoding size ({} given)".format(encoding_size))
            mem[h] = maxcount
        return mem[h]

    def encode(x):
        nonlocal encoding_size, dtype
        o = np.zeros(encoding_size, dtype=dtype)
        o[auto_label(x)] = True
        return o

    return _dataset_element_transforming(fn=encode)


def as_numpy() -> DatasetTransformFn:
    return _dataset_element_transforming(
        fn=np.array,
        check=_check_numpy_compatibility
    )


def as_image() -> DatasetTransformFn:
    return _dataset_element_transforming(
        fn=convert2img,
        check=_check_image_compatibility
    )


def img_resize(new_size:Shape, resample=Image.NEAREST) -> DatasetTransformFn:
    assert(len(new_size)==2)
    return _dataset_element_transforming(
        fn=lambda x: convert2img(x).resize(size=new_size, resample=resample),
        check=_check_image_compatibility
    )



########## Compose functions ####################

@_warn_no_args(skip=1)
def zipped(*datasets:AbstractDataset):
    comp = compose.ZipDataset(*datasets)
    return Dataset(
        downstream_getter=comp,
        ids=comp._ids,
    )


@_warn_no_args(skip=1)
def cartesian_product(*datasets:AbstractDataset):
    comp = compose.CartesianProductDataset(*datasets)
    return Dataset(
        downstream_getter=comp,
        ids=comp._ids,
    )


@_warn_no_args(skip=1)
def concat(*datasets:AbstractDataset):
    comp = compose.ConcatDataset(*datasets)
    return Dataset(
        downstream_getter=comp,
        ids=comp._ids,
    )