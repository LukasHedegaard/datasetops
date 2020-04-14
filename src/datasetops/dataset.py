"""
Module defining operations which may be applied to transform the data of a single dataset.
The transforms are available as free functions or as ``extension`` methods defined on the dataset objects:

  >>> ds.shuffle(seed=0)
  >>> ds_s = shuffle(ds,seed=0)
  >>> ds.idx == ds_s.idx
  True

"""

import random
import warnings
import functools
from typing import overload, TypeVar
from pathlib import Path
from inspect import signature

import numpy as np
from PIL import Image

from datasetops.abstract import ItemGetter, AbstractDataset
from datasetops.scaler import ElemStats
from datasetops.types import *
import datasetops.compose as compose
from datasetops.abstract import AbstractDataset
from datasetops import scaler

########## Local Helpers ####################

_DEFAULT_SHAPE = tuple()


def _warn_no_args(skip=0):
    def with_args(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if len(args) + len(kwargs) <= skip:
                warnings.warn("Too few args passed to {}".format(fn.__code__.co_name))
            return fn(*args, **kwargs)

        return wrapped

    return with_args


def _raise_no_args(skip=0):
    def with_args(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if len(args) + len(kwargs) <= skip:
                raise ValueError(
                    "Too few args passed to {}".format(fn.__code__.co_name)
                )
            return fn(*args, **kwargs)

        return wrapped

    return with_args


# def _dummy_arg_receiving(fn):
#     @functools.wraps(fn)
#     def wrapped(dummy, *args, **kwargs):
#         return fn(*args, **kwargs)

#     return wrapped


def _key_index(item_names: ItemNames, key: Key) -> int:
    if type(key) == int:
        return int(key)
    else:
        if not item_names:
            raise ValueError(
                "Items cannot be identified by name when no names are given. Hint: Use `Dataset.named('name1', 'name2', ...)`"
            )
        return item_names[str(key)]


def _split_bulk_itemwise(
    l: Union[Optional[Callable], Sequence[Optional[Callable]]]
) -> Tuple[Optional[Callable], Sequence[Optional[Callable]]]:
    bulk: Optional[Callable] = None
    itemwise: Sequence[Optional[Callable]] = []
    if hasattr(l, "__getitem__"):
        itemwise = l  # type: ignore
    else:
        bulk = l  # type: ignore
    return bulk, itemwise


def _combine_conditions(
    item_names: ItemNames,
    shape: Sequence[Shape],
    predicates: Optional[
        Union[DataPredicate, Sequence[Optional[DataPredicate]]]
    ] = None,
    **kwpredicates: DataPredicate,
) -> DataPredicate:

    bulk, itemwise = _split_bulk_itemwise(predicates)  # type:ignore

    if len(itemwise) > len(shape):
        raise ValueError("Too many predicates given")

    for k in kwpredicates.keys():
        if not k in item_names:
            raise KeyError("Key {} is not an item name".format(k))

    # clean up predicates
    if not bulk:

        def bulk(x):
            return True

    preds: List[DataPredicate] = [
        ((lambda x: True) if p is None else p) for p in itemwise
    ]

    def condition(x: Any) -> bool:
        return (
            bulk(x)
            and all([pred(x[i]) for i, pred in enumerate(preds)])
            and all(
                [pred(x[_key_index(item_names, k)]) for k, pred in kwpredicates.items()]
            )
        )

    return condition


def _optional_argument_indexed_transform(
    shape: Union[Shape, Sequence[Shape]],
    ds_transform: Callable,
    transform_fn: DatasetTransformFnCreator,
    args: Sequence[Optional[Sequence[Any]]],
):
    if len(args) == 0:
        warnings.warn("Skipping transform: No arguments arguments given")
    if len(shape) < len(args):
        raise ValueError("Unable to perform transform: Too many arguments given")

    tfs = [transform_fn(*a) if (a != None) else None for a in args]
    return ds_transform(tfs)


def _keywise(item_names: Dict[str, int], l: Sequence, d: Dict):
    keywise = {i: v for i, v in enumerate(l)}
    keywise.update({_key_index(item_names, k): v for k, v in d.items()})
    return keywise


def _itemwise(item_names: Dict[str, int], l: Sequence, d: Dict):
    keywise = _keywise(item_names, l, d)
    keywise = {k: v for k, v in keywise.items() if v != None}
    if keywise:
        itemwise = [
            ([keywise[i]] if i in keywise else None)
            for i in range(max(keywise.keys()) + 1)
        ]
    else:
        itemwise = []
    return itemwise


def _keyarg2list(
    item_names, key_or_keys: Union[Key, Sequence[Key]], arg: Sequence[Any]
) -> List[Optional[Sequence[Any]]]:
    if type(key_or_keys) in (list, tuple):
        idxs: List[int] = [
            _key_index(item_names, k) for k in key_or_keys  # type:ignore
        ]
    else:
        idxs: List[int] = [_key_index(item_names, key_or_keys)]  # type: ignore

    args = [arg if i in idxs else None for i in range(max(idxs) + 1)]
    return args


########## Dataset ####################


class Dataset(AbstractDataset):
    """
    Contains information on how to access the raw data, and performs
    sampling and splitting related operations.
    """

    def __init__(
        self,
        downstream_getter: Union[ItemGetter, "Dataset"],
        name: str = None,
        ids: Ids = None,
        item_transform_fn: ItemTransformFn = lambda x: x,
        item_names: Dict[str, int] = None,
        stats: List[Optional[ElemStats]] = [],
    ):
        """Initialise.

        Keyword Arguments:
            downstream_getter {ItemGetter} -- Any object which implements the __getitem__ function (default: {None})
            name {str} -- A name for the dataset
            ids {Ids} -- List of ids used in the downstream_getter (default: {None})
            item_transform_fn: {Callable} -- a function
        """
        self._downstream_getter = downstream_getter

        if issubclass(type(downstream_getter), AbstractDataset):
            self.name = self._downstream_getter.name  # type: ignore
            # type: ignore
            self._ids = list(range(len(self._downstream_getter._ids)))
            self._item_names = getattr(downstream_getter, "_item_names", None)
        else:
            self.name = ""
            self._ids = []
            self._item_names: ItemNames = {}

        if name:
            self.name = name
        if item_names:
            self._item_names = item_names
        if ids:
            self._ids: Ids = ids

        self._item_transform_fn = item_transform_fn
        self._item_stats = stats
        self._shape = None

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, i: IdxSlice) -> Tuple:
        """Returns the element at the specified index or slice

        Arguments:
            i {IdxSlice} -- An single index or a slice

        Returns:
            Tuple -- the element(s) of the dataset specified by the index or slice
        """
        if isinstance(i, int):
            return self._item_transform_fn(self._downstream_getter[self._ids[i]])
        else:
            ids = self._ids[i]
            return tuple(
                self._item_transform_fn(self._downstream_getter[self._ids[ii]])
                for ii in ids
            )

    def item_stats(self, item_key: Key, axis=None) -> scaler.ElemStats:
        """Compute the statistics (mean, std, min, max) for an item element

        Arguments:
            item_key {Key} -- index of string identifyer for element on which the stats should be computed

        Keyword Arguments:
            axis {[type]} -- the axis on which to accumulate statistics (default: {None})

        Raises:
            TypeError: if statistics cannot be computed on the element type

        Returns:
            scaler.ElemStats -- Named tuple with (mean, std, min, max, axis)
        """
        idx = _key_index(self._item_names, item_key)
        elem = self.__getitem__(0)[idx]

        if not type(elem) in [int, float, np.ndarray] or issubclass(
            type(elem), Image.Image
        ):
            raise TypeError(
                "Cannot compute statistics for element of type {}".format(type(elem))
            )

        if not axis:
            axis = -1 if issubclass(type(elem), Image.Image) else 0

        if len(self._item_stats) <= idx:
            for _ in range(idx - len(self._item_stats) + 1):
                self._item_stats.append(None)

        if not self._item_stats[idx] or self._item_stats[idx].axis != axis:

            def iterable():
                for item in self:
                    yield np.array(item[idx])

            self._item_stats[idx] = scaler.fit(
                iterable(), shape=np.array(elem).shape, axis=axis
            )
        return self._item_stats[idx]  # type: ignore

    @property
    # @functools.lru_cache(1)
    def shape(self) -> Sequence[Shape]:
        """Get the shape of a dataset's items.

        The process for doing this is picking a single sample from the dataset.
        Each item in the sample is checked for the presence "shape" or "size"
        properties. If present they are added to the shape tuple otherwise an
        empty tuple "()" is added.

        Returns:
            Sequence[int] -- Item shapes
        """
        if len(self) == 0:
            return _DEFAULT_SHAPE

        if self._shape is not None:
            return self._shape

        item = self.__getitem__(0)
        if hasattr(item, "__getitem__"):
            item_shape = []
            for i in item:
                if hasattr(i, "shape"):  # numpy arrays
                    item_shape.append(i.shape)
                elif hasattr(i, "size"):  # PIL.Image.Image
                    item_shape.append(np.array(i).shape)
                else:
                    item_shape.append(_DEFAULT_SHAPE)

            shape = tuple(item_shape)
        else:
            shape = _DEFAULT_SHAPE

        self._shape = shape
        return shape

    @functools.lru_cache(4)
    @_warn_no_args(skip=1)
    def counts(self, *itemkeys: Key) -> List[Tuple[Any, int]]:
        """Compute the counts of each unique item in the dataset.

        Warning: this operation may be expensive for large datasets

        Arguments:
            itemkeys {Union[str, int]} -- The item keys (str) or indexes (int) to be checked for uniqueness. If no key is given, all item-parts must match for them to be considered equal

        Returns:
            List[Tuple[Any,int]] -- List of tuples, each containing the unique value and its number of occurences
        """
        inds: List[int] = [_key_index(self._item_names, k) for k in itemkeys]

        if len(itemkeys) == 0:

            def selector(item):
                return item

        elif len(inds) == 1:

            def selector(item):
                return item[inds[0]]

        else:

            def selector(item):
                return tuple([val for i, val in enumerate(item) if i in inds])

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
    def unique(self, *itemkeys: Key) -> List[Any]:
        """Compute a list of unique values in the dataset.

        Warning: this operation may be expensive for large datasets

        Arguments:
            itemkeys {str} -- The item keys to be checked for uniqueness

        Returns:
            List[Any] -- List of the unique items
        """
        return [x[0] for x in self.counts(*itemkeys)]

    def subsample(self, func, n_samples: int, cache_method="block"):
        """Divide each sample in the dataset into several sub-samples using a user-defined function.
        The function must take a single sample as an argument and must return a list of samples.

        Arguments:
            func {Callable} -- function defining how each sample should divided.
            n_samples {int} -- the number of sub-samples produced for each sample.
            cache_method {Any} -- defines the caching method used by the subsampling operation. Possible options are {None, "block"}

        Returns:
            Dataset -- a new dataset containing the subsamples.
        """
        return SubsampleDataset(self, func, n_samples, cache_method)

    def supersample(self, func, n_samples: int):
        """Combines several samples into a smaller number of samples using a user-defined function.
        The function is invoked with an iterable of and must return a single sample.

        Arguments:
            func {[type]} -- a function used to transform a number of samples into a single supersample
            n_samples {int} -- number of samples required to produce each supersample

        Returns:
            [Dataset] -- a new dataset containing the supersamples
        """
        return SupersampleDataset(self, func, n_samples)

    def sample(self, num: int, seed: int = None):
        """Sample data randomly from the dataset.

        Arguments:
            num {int} -- Number of samples. If the number of samples is larger than the dataset size, some samples may be samples multiple times

        Keyword Arguments:
            seed {int} -- Random seed (default: {None})

        Returns:
            [Dataset] -- Sampled dataset
        """
        if seed:
            random.seed(seed)
        l = self.__len__()
        if l >= num:
            new_ids = random.sample(range(l), num)
        else:
            # TODO: determine if we should warn of raise an error instead
            new_ids = random.sample(range(l), l) + random.sample(
                range(l), num - l
            )  # Supersample.
        return Dataset(downstream_getter=self, ids=new_ids)

    @_warn_no_args(skip=1)
    def filter(
        self,
        predicates: Optional[
            Union[DataPredicate, Sequence[Optional[DataPredicate]]]
        ] = None,
        **kwpredicates: DataPredicate,
    ):
        """Filter a dataset using a predicate function.

        Keyword Arguments:
            predicates {Union[DataPredicate, Sequence[Optional[DataPredicate]]]} -- either a single or a list of functions taking a single dataset item and returning a bool if a single function is passed, it is applied to the whole item, if a list is passed, the functions are applied itemwise element-wise predicates can also be passed, if item_names have been named.
            kwpredicates {DataPredicate} -- TODO

        Returns:
            [Dataset] -- A filtered Dataset
        """
        condition = _combine_conditions(
            self._item_names, self.shape, predicates, **kwpredicates
        )
        new_ids = list(filter(lambda i: condition(self.__getitem__(i)), self._ids))
        return Dataset(downstream_getter=self, ids=new_ids)

    @_raise_no_args(skip=1)
    def split_filter(
        self,
        predicates: Optional[
            Union[DataPredicate, Sequence[Optional[DataPredicate]]]
        ] = None,
        **kwpredicates: DataPredicate,
    ):
        """Split a dataset using a predicate function.

        Keyword Arguments:
            predicates {Union[DataPredicate, Sequence[Optional[DataPredicate]]]} -- either a single or a list of functions taking a single dataset item and returning a bool. if a single function is passed, it is applied to the whole item, if a list is passed, the functions are applied itemwise
            element-wise predicates can also be passed, if item_names have been named.

        Returns:
            [Dataset] -- Two datasets, one that passed the predicate and one that didn't
        """
        condition = _combine_conditions(
            self._item_names, self.shape, predicates, **kwpredicates
        )
        ack, nack = [], []
        for i in self._ids:
            if condition(self.__getitem__(i)):
                ack.append(i)
            else:
                nack.append(i)

        return tuple(
            [Dataset(downstream_getter=self, ids=new_ids) for new_ids in [ack, nack]]
        )

    def shuffle(self, seed: int = None):
        """Shuffle the items in a dataset.

        Keyword Arguments:
            seed {[int]} -- Random seed (default: {None})

        Returns:
            [Dataset] -- Dataset with shuffled items
        """
        random.seed(seed)
        new_ids = list(range(len(self)))
        random.shuffle(new_ids)
        return Dataset(downstream_getter=self, ids=new_ids)

    def split(self, fractions: List[float], seed: int = None):
        """Split dataset into multiple datasets, determined by the fractions
        given.

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
        assert len(list(filter(lambda x: x == -1, fractions))) <= 1

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
            next_last_ind = last_ind + round(f * len(new_ids))
            if i != len(fractions) - 1:
                split_ids[i].extend(new_ids[last_ind:next_last_ind])
                last_ind = next_last_ind
            else:
                split_ids[i].extend(new_ids[last_ind:])

        # create datasets corresponding to each split
        return tuple(
            [Dataset(downstream_getter=self, ids=new_ids,) for new_ids in split_ids]
        )

    def take(self, num: int):
        """Take the first elements of a dataset.

        Arguments:
            num {int} -- number of elements to take

        Returns:
            Dataset -- A dataset with only the first `num` elements
        """
        if num > len(self):
            raise ValueError("Can't take more elements than are available in dataset")

        new_ids = list(range(num))
        return Dataset(downstream_getter=self, ids=new_ids)

    def repeat(self, times=1, mode="itemwise"):
        """Repeat the dataset elements.

        Keyword Arguments:
            times {int} -- Number of times an element is repeated (default: {1})
            mode {str} -- Repeat 'itemwise' (i.e. [1,1,2,2,3,3]) or as a 'whole' (i.e. [1,2,3,1,2,3]) (default: {'itemwise'})

        Returns:
            [type] -- [description]
        """
        new_ids = {
            "whole": lambda: [i for _ in range(times) for i in list(range(len(self)))],
            "itemwise": lambda: [
                i for i in list(range(len(self))) for _ in range(times)
            ],
        }[mode]()

        return Dataset(downstream_getter=self, ids=new_ids)

    def reorder(self, *keys: Key):
        """Reorder items in the dataset (similar to numpy.transpose).

        Arguments:
            new_inds {Union[int,str]} -- positioned item index or key (if item names were previously set) of item

        Returns:
            [Dataset] -- Dataset with items whose elements have been reordered
        """
        if len(keys) == 0:
            warnings.warn(
                "No indexes given in Dataset.reorder. The dataset remains unchanged"
            )
            return self

        inds = [_key_index(self._item_names, k) for k in keys]

        for i in inds:
            if i > len(self.shape):
                raise ValueError(
                    "reorder index {} is out of range (maximum allowed index is {})".format(
                        i, len(self.shape)
                    )
                )

        def item_transform_fn(item: Tuple):
            return tuple([item[i] for i in inds])

        item_names = None
        if self._item_names:
            if len(set(keys)) < len(keys):
                warnings.warn(
                    "discarding item_names due to otherwise non-unique labels on transformed dataset"
                )
            else:
                item_names = {
                    k: inds[v] for k, v in self._item_names.items() if v < len(inds)
                }

        return Dataset(
            downstream_getter=self,
            item_transform_fn=item_transform_fn,
            item_names=item_names,
        )

    def named(self, first: Union[str, Sequence[str]], *rest: str):
        """Set the names associated with the elements of an item.

        Arguments:
            first {Union[str, Sequence[str]]} -- The new item name(s)

        Returns:
            [Dataset] -- A Dataset whose item elements can be accessed by name
        """
        names: List[str] = []

        if type(first) == str:
            names.append(first)  # type: ignore
        else:
            assert hasattr(first, "__len__")
            assert type(first[0]) is str
            names = list(first)  # type: ignore

        names.extend(rest)

        assert len(names) <= len(self.shape)
        self._item_names = {n: i for i, n in enumerate(names)}
        return self

    @property
    def names(self) -> List[str]:
        """Get the names of the elements in an item.

        Returns:
            List[str] -- A list of element names
        """
        if self._item_names:
            return [x[0] for x in sorted(self._item_names.items(), key=lambda x: x[1])]
        else:
            return []

    @_warn_no_args(skip=1)
    def transform(
        self,
        fns: Optional[
            Union[
                ItemTransformFn, Sequence[Union[ItemTransformFn, DatasetTransformFn]],
            ]
        ] = None,
        **kwfns: DatasetTransformFn,
    ):
        """Transform the items of a dataset according to some function (passed
        as argument).

        Arguments:
            If a single function taking one input given, e.g. transform(lambda x: x), it will be applied to the whole item.
            If a list of functions are given, e.g. transform([image(), one_hot()]) they will be applied to the elements of the item corresponding to the position.
            If key is used, e.g. transform(data=lambda x:-x), the item associated with the key i transformed.

        Raises:
            ValueError: If more functions are passed than there are elements in an item.
            KeyError: If a key doesn't match

        Returns:
            [Dataset] -- Dataset whose items are transformed
        """
        bulk, itemwise = _split_bulk_itemwise(fns)

        if bool(bulk) + len(itemwise) + len(kwfns) > len(self.shape):
            raise ValueError(
                "More transforms ({}) given than can be performed on item with {} elements".format(
                    bool(bulk) + len(itemwise) + len(kwfns), len(self.shape)
                )
            )
        new_dataset: AbstractDataset = self

        if bulk:
            return Dataset(downstream_getter=self, item_transform_fn=bulk)

        for k, v in list(enumerate(itemwise)) + list(kwfns.items()):  # type:ignore
            funcs = v if type(v) in [list, tuple] else [v]
            for f in funcs:
                if f:
                    if len(signature(f).parameters) == 1:
                        f = _custom(f)
                    new_dataset = f(_key_index(self._item_names, k), new_dataset)

        return new_dataset

    ########## Label transforming methods #########################

    def categorical(self, key: Key, mapping_fn: Callable[[Any], int] = None):
        """Transform elements into categorical categoricals (int).

        Arguments:
            key {Key} -- Index of name for the element to be transformed

        Keyword Arguments:
            mapping_fn {Callable[[Any], int]} -- User defined mapping function (default: {None})

        Returns:
            [Dataset] -- Dataset with items that have been transformed to categorical labels
        """
        idx: int = _key_index(self._item_names, key)
        mapping_fn = mapping_fn or categorical_template(self, key)
        args = [[mapping_fn] or [] if i == idx else None for i in range(idx + 1)]
        return _optional_argument_indexed_transform(
            self.shape, self.transform, transform_fn=categorical, args=args
        )

    def one_hot(
        self,
        key: Key,
        encoding_size: int = None,
        mapping_fn: Callable[[Any], int] = None,
        dtype="bool",
    ):
        """Transform elements into a categorical one-hot encoding.

        Arguments:
            key {Key} -- Index of name for the element to be transformed

        Keyword Arguments:
            encoding_size {int} -- The number of positions in the one-hot vector. If size it not provided, it we be automatically inferred (with a O(N) runtime cost) (default: {None})
            mapping_fn {Callable[[Any], int]} -- User defined mapping function (default: {None})
            dtype {str} -- Numpy datatype for the one-hot encoded data (default: {'bool'})

        Returns:
            [Dataset] -- Dataset with items that have been transformed to categorical labels
        """
        enc_size = encoding_size or len(self.unique(key))
        mapping_fn = mapping_fn or categorical_template(self, key)
        idx: int = _key_index(self._item_names, key)
        args = [[enc_size] if i == idx else None for i in range(idx + 1)]
        return _optional_argument_indexed_transform(
            self.shape,
            self.transform,
            transform_fn=functools.partial(one_hot, mapping_fn=mapping_fn, dtype=dtype),
            args=args,
        )

    ########## Conversion methods #########################

    # TODO: reconsider API
    def image(self, *positional_flags: Any):
        """Transforms item elements that are either numpy arrays or path
        strings into a PIL.Image.Image.

        Arguments:
            positional flags, e.g. (True, False) denoting which element should be converted. If no flags are supplied, all data that can be converted will be converted.

        Returns:
            [Dataset] -- Dataset with PIL.Image.Image elements
        """
        if len(positional_flags) == 0:
            # convert all that can be converted
            positional_flags = []
            for elem in self.__getitem__(0):
                try:
                    _check_image_compatibility(elem)
                    positional_flags.append([])  # didn't raise error
                except:
                    positional_flags.append(None)

        if any([f != None and f != False for f in positional_flags]):
            return _optional_argument_indexed_transform(
                self.shape, self.transform, transform_fn=image, args=positional_flags,  # type: ignore
            )
        else:
            warnings.warn("Conversion to image skipped. No elements were compatible")
            return self

    # TODO: reconsider API
    def numpy(self, *positional_flags: Any):
        """Transforms elements into numpy.ndarray.

        Arguments:
            positional flags, e.g. (True, False) denoting which element should be converted. If no flags are supplied, all data that can be converted will be converted.

        Returns:
            [Dataset] -- Dataset with np.ndarray elements
        """
        if len(positional_flags) == 0:
            # convert all that can be converted
            positional_flags = []
            for elem in self.__getitem__(0):
                try:
                    _check_numpy_compatibility()(elem)
                    positional_flags.append([])  # didn't raise error
                except:
                    positional_flags.append(None)

        if any([f != None and f != False for f in positional_flags]):
            return _optional_argument_indexed_transform(
                self.shape, self.transform, transform_fn=numpy, args=positional_flags,  # type: ignore
            )
        else:
            warnings.warn(
                "Conversion to numpy.array skipped. No elements were compatible"
            )
            return self

    ########## Composition methods #########################

    def zip(self, *datasets):
        return zipped(self, *datasets)

    def cartesian_product(self, *datasets):
        return cartesian_product(self, *datasets)

    def concat(self, *datasets):
        return concat(self, *datasets)

    ########## Methods relating to numpy data #########################

    def reshape(self, *new_shapes: Optional[Shape], **kwshapes: Optional[Shape]):
        return _optional_argument_indexed_transform(
            self.shape,
            self.transform,
            transform_fn=reshape,
            args=_itemwise(self._item_names, new_shapes, kwshapes),
        )

    # def add_noise(self, noise):
    #     pass

    ########## Methods below assume data is an image ##########

    def image_resize(self, *new_sizes: Optional[Shape], **kwsizes: Optional[Shape]):
        return _optional_argument_indexed_transform(
            self.shape,
            self.transform,
            transform_fn=image_resize,
            args=_itemwise(self._item_names, new_sizes, kwsizes),
        )

    # def img_transform(self, transform):
    #     pass

    # def img_filter(self, filter_fn):
    #     pass

    ########## Dataset scalers #########################

    def standardize(self, key_or_keys: Union[Key, Sequence[Key]], axis=0):
        """Standardize features by removing the mean and scaling to unit variance

        Arguments:
            key_or_keys {Union[Key, Sequence[Key]]} -- The keys on which the Max Abs scaling should be performed

        Keyword Arguments:
            axis {int} -- Axis on which to accumulate statistics (default: {0})

        Returns:
            Dataset -- Transformed dataset
        """
        return _optional_argument_indexed_transform(
            self.shape,
            self.transform,
            transform_fn=standardize,
            args=_keyarg2list(self._item_names, key_or_keys, [axis]),
        )

    def center(self, key_or_keys: Union[Key, Sequence[Key]], axis=0):
        """Centers features by removing the mean

        Arguments:
            key_or_keys {Union[Key, Sequence[Key]]} -- The keys on which the Max Abs scaling should be performed

        Keyword Arguments:
            axis {int} -- Axis on which to accumulate statistics (default: {0})

        Returns:
            Dataset -- Transformed dataset
        """
        return _optional_argument_indexed_transform(
            self.shape,
            self.transform,
            transform_fn=center,
            args=_keyarg2list(self._item_names, key_or_keys, [axis]),
        )

    # When people say normalize, they often mean either minmax or standardize.
    # This implementation follows the scikit-learn terminology
    # Not included in the library for now, because it is used very seldomly in practice
    # def normalize(self, key_or_keys: Union[Key, Sequence[Key]], axis=0, norm="l2"):
    #     """Normalize samples individually to unit norm.

    #     Arguments:
    #         key_or_keys {Union[Key, Sequence[Key]]} -- The keys on which the Max Abs scaling should be performed

    #     Keyword Arguments:
    #         axis {int} -- Axis on which to accumulate statistics (default: {0})

    #     Keyword Arguments:
    #         norm {str} -- "l1" or "l2"

    #     Returns:
    #         Dataset -- Transformed dataset
    #     """
    #     return _optional_argument_indexed_transform(
    #         self.shape,
    #         self.transform,
    #         transform_fn=normalize,
    #         args=_keyarg2list(self._item_names, key_or_keys, [axis]),
    #     )

    def minmax(
        self, key_or_keys: Union[Key, Sequence[Key]], axis=0, feature_range=(0, 1),
    ):
        """Transform features by scaling each feature to a given range.

        Arguments:
            key_or_keys {Union[Key, Sequence[Key]]} -- The keys on which the Max Abs scaling should be performed

        Keyword Arguments:
            axis {int} -- Axis on which to accumulate statistics (default: {0})

        Keyword Arguments:
            feature_range {Tuple[int, int]} -- Minimum and maximum bound to scale to

        Returns:
            Dataset -- Transformed dataset
        """
        return _optional_argument_indexed_transform(
            self.shape,
            self.transform,
            transform_fn=minmax,
            args=_keyarg2list(self._item_names, key_or_keys, [axis, feature_range],),
        )

    def maxabs(self, key_or_keys: Union[Key, Sequence[Key]], axis=0):
        """Scale each feature by its maximum absolute value.

        Arguments:
            key_or_keys {Union[Key, Sequence[Key]]} -- The keys on which the Max Abs scaling should be performed

        Keyword Arguments:
            axis {int} -- Axis on which to accumulate statistics (default: {0})

        Returns:
            Dataset -- Transformed dataset
        """
        return _optional_argument_indexed_transform(
            self.shape,
            self.transform,
            transform_fn=maxabs,
            args=_keyarg2list(self._item_names, key_or_keys, [axis]),
        )

    ########## Framework converters #########################

    def to_tensorflow(self):
        return to_tensorflow(self)

    def to_pytorch(self):
        return to_pytorch(self)


class SubsampleDataset(Dataset):
    def __init__(self, dataset: Dataset, subsample_func, n_ss: int, cache_method=None):

        if n_ss < 1:
            raise ValueError(
                "Unable to perform subsampling, value of n_ss should be greater than one."
            )

        valid_cache_methods = {"block", None}

        if cache_method not in valid_cache_methods:
            raise ValueError(
                "Unable to perform subsampling, cache method: {cache_methods} is invalid, possible values are {valid_cache_methods}"
            )

        new_ids = list(range(0, len(dataset) * n_ss))

        # super().__init__(self, ids=new_ids) # TODO why does every class deriving from Dataset have to define _ids?
        super().__init__(dataset, ids=new_ids)

        self.cached = {}
        self.subsample_func = subsample_func
        self.n_ss = n_ss
        self.dataset = dataset
        self.cache_method = cache_method
        self.last_downstream_idx = None

    def __getitem__(self, ss_idx):
        """Gets the subsample corresponding to the

        Arguments:
            ss_idx {[type]} -- index of the subsample

        Returns:
            [Any] -- the subsample corresponding to the specified index
        """

        ds_idx = self._get_downstream_idx(ss_idx)

        if self._is_subsample_cached(ss_idx):
            return self._get_cached_subsample(ss_idx)
        else:

            ds_sample = self.dataset[ds_idx]
            ss = self.subsample_func(ds_sample)
            n_actual = None
            # ensure that subsampling function has returned the correct value of subsamples
            try:
                n_actual = len(ss)
            except Exception as e:
                raise RuntimeError(
                    f"subsampling function returned: {n_actual}, this should be an iterable"
                )

            if n_actual != self.n_ss:
                raise RuntimeError(
                    f"subsampling function returned {n_actual} subsamples, which is different than the expected: {self.n_ss}"
                )

            self._do_cache_for(ds_idx, ss)
            ss_relative_idx = ss_idx % self.n_ss
            return ss[ss_relative_idx]

    def _get_downstream_idx(self, ss_idx):
        return ss_idx // self.n_ss

    def _is_subsample_cached(self, ss_idx):
        return self._get_downstream_idx(ss_idx) in self.cached

    def _get_cached_subsample(self, ss_idx):
        assert self._is_subsample_cached(ss_idx)

        ds_idx = self._get_downstream_idx(ss_idx)
        ss_relative_idx = ss_idx % self.n_ss
        return self.cached[ds_idx][ss_relative_idx]

    def _do_cache_for(self, ds_idx, ss):
        """Caches the values read from the specified index of the downstream data set.

        Arguments:
            ds_idx {Idx} -- index of the last read downstream sample
            ss {Tuple[Any]} -- the items produced by subsampling the downstream dataset at the specified index.
        """

        if self.cache_method == None:
            return
        elif self.cache_method == "block":
            if (
                ds_idx != self.last_downstream_idx
                and self.last_downstream_idx is not None
            ):
                del self.cached[self.last_downstream_idx]

            self.cached[ds_idx] = ss
            self.last_downstream_idx = ds_idx


class SupersampleDataset(AbstractDataset):
    def __init__(
        self, dataset, func, sampling_ratio: int, excess_samples_policy="discard"
    ):
        """Performs supersampling on the provided dataset.

        Arguments:
            dataset {AbstractDataset} -- the dataset which the supersampling is applied to
            func {Callable} -- function used to combine several samples into a single supersample.
            sampling_ratio {int} -- the number of samples used to produce a each supersample.

        Keyword Arguments:
            excess_samples_policy {str} -- defines how left over samples should be treated. Possible values are {"discard","error"} (default: {"discard"})

        """

        extra_sample_policy_options = {"discard", "error"}

        if excess_samples_policy not in extra_sample_policy_options:
            raise ValueError(
                f"Illegal value for argument excess_samples_policy: {excess_samples_policy}, possible options are {excess_samples_policy_options}."
            )

        if sampling_ratio < 1:
            raise ValueError(
                f"Illegal value for argument sampling_ratio: {sampling_ratio}, this must be 1 or greater."
            )

        excess_samples = len(dataset) % sampling_ratio
        if excess_samples_policy == "error" and (excess_samples != 0):
            raise ValueError(
                f"The specified excess sample policy: {excess_samples} does not permit left over samples, of which: {excess_samples} would exist."
            )

        self.sampling_ratio = sampling_ratio
        self.func = func
        self.dataset = dataset

    def __getitem__(self, idx):
        start = idx * self.sampling_ratio
        end = start + self.sampling_ratio
        ss = self.dataset[start:end]

        return self.func(ss)

    def __len__(self):
        return len(self.dataset) // self.sampling_ratio


########## Handy decorators ####################


def _make_dataset_element_transforming(
    make_fn: Callable[[AbstractDataset, Optional[int]], Callable],
    check: Callable = None,
    maintain_stats=False,
) -> DatasetTransformFn:
    def wrapped(idx: int, ds: AbstractDataset) -> AbstractDataset:

        fn = make_fn(ds, idx)

        if check:
            check(ds[0][idx])

        def item_transform_fn(item: Sequence):
            return tuple(
                [fn(elem) if i == idx else elem for i, elem in enumerate(item)]
            )

        stats: List[Optional[ElemStats]] = []
        if maintain_stats and hasattr(ds, "_item_stats"):
            # maintain stats on other elements, and optionally on this one
            stats = [
                (s if i != idx else (s if maintain_stats else None))
                for i, s in enumerate(ds._item_stats)  # type:ignore
            ]

        return Dataset(
            downstream_getter=ds, item_transform_fn=item_transform_fn, stats=stats,
        )

    return wrapped


def _dataset_element_transforming(
    fn: Callable, check: Callable = None, maintain_stats=False
) -> DatasetTransformFn:
    """Applies the function to dataset item elements."""

    def wrapped(idx: int, ds: AbstractDataset) -> AbstractDataset:

        if check:
            check(ds[0][idx])

        def item_transform_fn(item: Sequence):
            return tuple(
                [fn(elem) if i == idx else elem for i, elem in enumerate(item)]
            )

        stats: List[Optional[ElemStats]] = []
        if maintain_stats and hasattr(ds, "_item_stats"):
            # maintain stats on other elements, and optionally on this one
            stats = [
                (s if i != idx else (s if maintain_stats else None))
                for i, s in enumerate(ds._item_stats)  # type:ignore
            ]

        return Dataset(
            downstream_getter=ds, item_transform_fn=item_transform_fn, stats=stats,
        )

    return wrapped


def _check_shape_compatibility(shape: Shape):
    def check(elem):
        if not hasattr(elem, "shape"):
            raise ValueError(
                "{} needs a shape attribute for shape compatibility to be checked".format(
                    elem
                )
            )

        if (
            np.prod(elem.shape) != np.prod(shape)  # type: ignore
            and not (-1 in shape)
        ) or any([s > np.prod(elem.shape) for s in shape]):
            raise ValueError(
                "Cannot reshape dataset with shape '{}' to shape '{}'. Dimensions cannot be matched".format(
                    elem.shape, shape
                )
            )

    return check


def convert2img(elem: Union[Image.Image, str, Path, np.ndarray]) -> Image.Image:
    if issubclass(type(elem), Image.Image):
        return elem

    if type(elem) in [str, Path]:
        if Path(elem).is_file():  # type: ignore
            return Image.open(elem)

    if type(elem) == np.ndarray:
        if issubclass(elem.dtype.type, np.integer):  # type:ignore
            return Image.fromarray(np.uint8(elem))  # type:ignore
        elif issubclass(elem.dtype.type, np.floating):  # type:ignore
            return Image.fromarray(np.float32(elem))  # type:ignore

    raise ValueError("Unable to convert element {} to Image".format(elem))


def _check_image_compatibility(elem):
    # check if this raises an Exception
    convert2img(elem)


def _check_numpy_compatibility(allow_scalars=False):
    allowed = [dict, str]
    if not allow_scalars:
        allowed.extend([int, float])

    def fn(elem):
        if type(elem) in allowed:
            raise ValueError("Unable to convert element {} to numpy".format(elem))
        # check if this raises an Exception
        np.array(elem)

    return fn


########## Predicate implementations ####################


def allow_unique(max_num_duplicates=1) -> Callable[[Any], bool]:
    """Predicate used for filtering/sampling a dataset classwise.

    Keyword Arguments:
        max_num_duplicates {int} -- max number of samples to take that share the same value (default: {1})

    Returns:
        Callable[[Any], bool] -- Predicate function
    """
    mem_counts = {}

    def fn(x):
        nonlocal mem_counts
        h = hash(str(x))
        if not h in mem_counts.keys():
            mem_counts[h] = 1
            return True
        if mem_counts[h] < max_num_duplicates:
            mem_counts[h] += 1
            return True
        return False

    return fn


########## Transform implementations ####################


def _custom(
    elem_transform_fn: Callable[[Any], Any],
    elem_check_fn: Callable[[Any], None] = None,
) -> DatasetTransformFn:
    """Create a user defined transform.

    Arguments:
        fn {Callable[[Any], Any]} -- A user defined function, which takes the element as only argument

    Keyword Arguments:
        check_fn {Callable[[Any]]} -- A function that raises an Exception if the elem is incompatible (default: {None})

    Returns:
        DatasetTransformFn -- [description]
    """
    return _dataset_element_transforming(fn=elem_transform_fn, check=elem_check_fn)


def reshape(new_shape: Shape) -> DatasetTransformFn:
    return _dataset_element_transforming(
        fn=lambda x: np.reshape(np.array(x), newshape=tuple(new_shape)),
        check=_check_shape_compatibility(new_shape),
    )


def categorical(mapping_fn: Callable[[Any], int] = None) -> DatasetTransformFn:
    """Transform data into a categorical int label.

    Arguments:
        mapping_fn {Callable[[Any], int]} -- A function transforming the input data to the integer label. If not specified, labels are automatically inferred from the data.

    Returns:
        DatasetTransformFn -- A function to be passed to the Dataset.transform()
    """
    mem, maxcount = {}, -1

    def auto_label(x: Any) -> int:
        nonlocal mem, maxcount
        h = hash(str(x))
        if not h in mem.keys():
            maxcount += 1
            mem[h] = maxcount
        return mem[h]

    return _dataset_element_transforming(
        fn=mapping_fn if callable(mapping_fn) else auto_label  # type: ignore
    )


def categorical_template(ds: Dataset, key: Key) -> Callable[[Any], int]:
    """Creates a template mapping function to be with one_hot.

    Arguments:
        ds {Dataset} -- Dataset from which to create a template for one_hot coding
        key {Key} -- Dataset key (name or item index) on the one_hot coding is made

    Returns:
        {Callable[[Any],int]} -- mapping_fn for one_hot
    """
    unq = ds.unique(key)
    if type(unq[0]) == np.ndarray:

        def mapper(x):
            return x.tobytes()

        unq = [mapper(x) for x in unq]
    else:

        def mapper(x):
            return x

    d = {k: i for i, k in enumerate(sorted(unq))}

    def fn(i):
        return d[mapper(i)]

    return fn


def one_hot(
    encoding_size: int, mapping_fn: Callable[[Any], int] = None, dtype="bool"
) -> DatasetTransformFn:
    """Transform data into a one-hot encoded label.

    Arguments:
        encoding_size {int} -- The size of the encoding
        mapping_fn {Callable[[Any], int]} -- A function transforming the input data to an integer label. If not specified, labels are automatically inferred from the data.

    Returns:
        DatasetTransformFn -- A function to be passed to the Dataset.transform()
    """
    mem, maxcount = {}, -1

    def auto_label(x: Any) -> int:
        nonlocal mem, maxcount, encoding_size
        h = hash(str(x))
        if not h in mem.keys():
            maxcount += 1
            if maxcount >= encoding_size:
                raise ValueError(
                    "More unique labels found than were specified by the encoding size ({} given)".format(
                        encoding_size
                    )
                )
            mem[h] = maxcount
        return mem[h]

    label_fn = mapping_fn or auto_label

    def encode(x):
        nonlocal encoding_size, dtype, label_fn
        o = np.zeros(encoding_size, dtype=dtype)
        o[label_fn(x)] = True
        return o

    return _dataset_element_transforming(fn=encode)


def numpy() -> DatasetTransformFn:
    return _dataset_element_transforming(
        fn=np.array, check=_check_numpy_compatibility(), maintain_stats=True
    )


def image() -> DatasetTransformFn:
    return _dataset_element_transforming(
        fn=convert2img, check=_check_image_compatibility, maintain_stats=True
    )


def image_resize(new_size: Shape, resample=Image.NEAREST) -> DatasetTransformFn:
    assert len(new_size) == 2
    return _dataset_element_transforming(
        fn=lambda x: convert2img(x).resize(size=new_size, resample=resample),
        check=_check_image_compatibility,
    )


########## Dataset scalers #########################


def standardize(axis=0) -> DatasetTransformFn:
    """Standardize features by removing the mean and scaling to unit variance

    Keyword Arguments:
        axis {int} -- Axis on which to accumulate statistics (default: {0})

    Returns:
        DatasetTransformFn -- Function to be passed to Datasets.transform
    """

    def make_fn(ds, idx) -> Callable:
        return scaler.standardize(shape=ds.shape[idx], stats=ds.item_stats(idx, axis))

    return _make_dataset_element_transforming(
        make_fn=make_fn,
        check=_check_numpy_compatibility(allow_scalars=True),
        maintain_stats=True,
    )


def center(axis=0) -> DatasetTransformFn:
    """Center features by removing the mean

    Keyword Arguments:
        axis {int} -- Axis on which to accumulate statistics (default: {0})

    Returns:
        DatasetTransformFn -- Function to be passed to Datasets.transform
    """

    def make_fn(ds, idx) -> Callable:
        return scaler.center(shape=ds.shape[idx], stats=ds.item_stats(idx, axis))

    return _make_dataset_element_transforming(
        make_fn=make_fn,
        check=_check_numpy_compatibility(allow_scalars=True),
        maintain_stats=True,
    )


# When people say normalize, they often mean either minmax or standardize.
# This implementation follows the scikit-learn terminology
# Not included in the library for now, because it is used very seldomly in practice
# def normalize(axis=0, norm="l2") -> DatasetTransformFn:
#     """Normalize samples individually to unit norm.

#     Keyword Arguments:
#         axis {int} -- Axis on which to accumulate statistics (default: {0})

#     Keyword Arguments:
#         norm {str} -- "l1" or "l2"

#     Returns:
#         DatasetTransformFn -- Function to be passed to Datasets.transform
#     """

#     def make_fn(ds, idx) -> Callable:
#         return scaler.normalize(shape=ds.shape[idx], axis=axis, norm=norm,)

#     return _make_dataset_element_transforming(
#         make_fn=make_fn, check=_check_numpy_compatibility(allow_scalars=True),
#     )


def minmax(axis=0, feature_range=(0, 1)) -> DatasetTransformFn:
    """Transform features by scaling each feature to a given range.

    Keyword Arguments:
        axis {int} -- Axis on which to accumulate statistics (default: {0})

    Keyword Arguments:
        feature_range {Tuple[int, int]} -- Minimum and maximum bound to scale to

    Returns:
        DatasetTransformFn -- Function to be passed to Datasets.transform
    """

    def make_fn(ds, idx) -> Callable:
        return scaler.minmax(
            shape=ds.shape[idx],
            stats=ds.item_stats(idx, axis),
            feature_range=feature_range,
        )

    return _make_dataset_element_transforming(
        make_fn=make_fn,
        check=_check_numpy_compatibility(allow_scalars=True),
        maintain_stats=True,
    )


def maxabs(axis=0) -> DatasetTransformFn:
    """Scale each feature by its maximum absolute value.

    Keyword Arguments:
        axis {int} -- Axis on which to accumulate statistics (default: {0})

    Returns:
        DatasetTransformFn -- Function to be passed to Datasets.transform
    """

    def make_fn(ds, idx) -> Callable:
        return scaler.maxabs(shape=ds.shape[idx], stats=ds.item_stats(idx, axis))

    return _make_dataset_element_transforming(
        make_fn=make_fn,
        check=_check_numpy_compatibility(allow_scalars=True),
        maintain_stats=True,
    )


########## Compose functions ####################


@_warn_no_args(skip=1)
def zipped(*datasets: AbstractDataset):
    comp = compose.ZipDataset(*datasets)
    return Dataset(downstream_getter=comp, ids=comp._ids,)


@_warn_no_args(skip=1)
def cartesian_product(*datasets: AbstractDataset):
    comp = compose.CartesianProductDataset(*datasets)
    return Dataset(downstream_getter=comp, ids=comp._ids,)


@_warn_no_args(skip=1)
def concat(*datasets: AbstractDataset):
    comp = compose.ConcatDataset(*datasets)
    return Dataset(downstream_getter=comp, ids=comp._ids,)


########## Sampling ####################
def subsample(dataset, func, n_samples: int, cache_method="block") -> Dataset:
    """Divide each sample in the dataset into several sub-samples using a user-defined function.
    The function must take a single sample as an argument and must return a list of samples.

    Arguments:
        dataset {[type]} -- dataset containing the samples which are sub-sampled.
        func {Callable} -- function defining how each sample should divided.
        n_samples {int} -- the number of sub-samples produced for each sample.
        cache_method {Any} -- defines the caching method used by the subsampling operation. Possible options are {None, "block"}

    Returns:
        Dataset -- a new dataset containing the subsamples.
    """

    return SubsampleDataset(dataset, func, n_samples, cache_method)


########## Converters ####################


def _tf_compute_type(item: Any):
    import tensorflow as tf  # type:ignore

    if type(item) in [list, tuple]:
        return tuple([_tf_compute_type(i) for i in item])

    if type(item) == dict:
        return {str(k): _tf_compute_type(v) for k, v in item.items()}

    return tf.convert_to_tensor(item).dtype


def _tf_compute_shape(item: Any):
    import tensorflow as tf  # type:ignore

    if type(item) in [list, tuple]:
        return tuple([_tf_compute_shape(i) for i in item])

    if type(item) == dict:
        return {str(k): _tf_compute_shape(v) for k, v in item.items()}

    return tf.convert_to_tensor(item).shape


def _tf_item_conversion(item: Any):
    if type(item) in [list, tuple]:
        return tuple([_tf_item_conversion(i) for i in item])

    if type(item) == dict:
        return {str(k): _tf_item_conversion(v) for k, v in item.items()}

    if issubclass(type(item), Image.Image):
        return np.array(item)

    return item


def to_tensorflow(dataset: Dataset):
    import tensorflow as tf  # type:ignore

    ds = Dataset(downstream_getter=dataset, item_transform_fn=_tf_item_conversion)
    item = ds[0]
    return tf.data.Dataset.from_generator(
        generator=dataset.generator,
        output_types=_tf_compute_type(item),
        output_shapes=_tf_compute_shape(item),
    )


def to_pytorch(dataset: Dataset):
    from torch.utils.data import Dataset as TorchDataset

    class PyTorchDataset(TorchDataset):
        def __init__(self, dataset: Dataset):
            self.dataset = dataset

        def __len__(self,):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.dataset[idx]

    return PyTorchDataset(dataset)
