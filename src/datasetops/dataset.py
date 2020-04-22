"""
Implementation of basic Dataset functionality.
"""

from datasetops.interfaces import (
    IDataset,
    ISampleProvider,
    SampleIds,
    SampleTransform,
    ElemNameToIndex,
    ElemStats,
    Sample,
    SampleShape,
    ElemKey,
    ElemTransform,
    ElemIndex,
)
import typing
from typing import Dict, List, Iterator, Union, Sequence, Tuple, Any, Optional
import numpy as np
from datasetops.helpers import documents, signature, index_from, inds_from, collect_keys
from warnings import warn
from functools import lru_cache

_DEFAULT_SHAPE = tuple()


class Dataset(IDataset):
    def __init__(
        self,
        parent: ISampleProvider,
        sample_ids: SampleIds = [],
        operation_name: str = "",
        operation_parameters: Dict = {},
        transform_func: SampleTransform = None,
        elem_name2index: Optional[ElemNameToIndex] = {},
        stats: Optional[Dict[ElemIndex, ElemStats]] = {},
    ):
        self._parent = parent
        self._sample_ids = sample_ids
        self._operation_name = operation_name
        self._operation_parameters = operation_parameters
        self._transform_func = transform_func or (lambda x: x)
        self._elem_name2index = elem_name2index or {}
        self._stats = stats or {}

        # if parent is a Dataset, we can automatically infer some attributes
        if type(parent) == Dataset:
            ds: Dataset = parent  # type: ignore
            self._sample_ids = sample_ids or range(len(ds))
            self._elem_name2index = elem_name2index or ds._elem_name2index
            self._stats = stats or ds._stats

        # allow override with empty fields
        if elem_name2index is None:
            self._elem_name2index = {}
        if stats is None:
            self._stats = {}

    def __len__(self):
        return len(self._sample_ids)

    @typing.overload
    def __getitem__(self, i: slice) -> List[Sample]:
        ...

    @typing.overload
    def __getitem__(self, i: int) -> Sample:
        ...

    def __getitem__(self, i):
        if type(i) == int:
            return self._transform_func(self._parent[self._sample_ids[i]])
        elif type(i) == slice:
            return [
                self._transform_func(self._parent[self._sample_ids[ii]])
                for ii in range(len(self))[i]  # type: ignore
            ]
        else:
            raise IndexError("Index of type {} is not supported".format(type(i)))

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    @property
    def generator(self):
        def g():
            for sample in self:
                yield sample

        return g

    @documents(IDataset)
    def named(self, name_or_names: Union[str, Sequence[str]], *rest: str) -> "IDataset":
        """Set the names associated with the elements of an item.

        Arguments:
            name_or_names {Union[str, Sequence[str]]} -- Either all the items names in a sequence, or a single item name
            rest {str} -- remaining item names

        Returns:
            {Dataset} -- A Dataset whose item elements can be accessed by name
        """
        names: List[str] = []

        if type(name_or_names) == str:
            names.append(name_or_names)  # type: ignore
        else:
            assert hasattr(name_or_names, "__len__")
            assert type(name_or_names[0]) is str
            names = list(name_or_names)  # type: ignore

        names.extend(rest)

        assert (len(names) <= len(self.shape)) or len(self) == 0
        elem_name2index = {n: i for i, n in enumerate(names)}
        return Dataset(
            parent=self,
            elem_name2index=elem_name2index,
            operation_name="named",
            operation_parameters={"names": names},
        )

    @property
    @documents(IDataset)
    def names(self) -> ElemNameToIndex:
        """Get the names associated with sample elements

        Returns:
            {List[str]} -- names
        """
        return self._elem_name2index

    @property
    @lru_cache(1)
    @documents(IDataset)
    def shape(self) -> SampleShape:
        """Get the shape of a sample.

        Returns:
            {Tuple[Tuple[int, ...], ...]} -- Sample shape
        """
        # if len(self) == 0:
        #     return _DEFAULT_SHAPE

        # if hasattr(self, "_shape"):
        #     return self._shape

        item = self.__getitem__(0)
        # if hasattr(item, "__getitem__"):
        sample_shape = []
        for i in item:
            if hasattr(i, "shape"):  # numpy arrays
                sample_shape.append(i.shape)
            elif hasattr(i, "size"):  # PIL.Image.Image
                sample_shape.append(np.array(i).shape)
            else:
                sample_shape.append(_DEFAULT_SHAPE)

        shape: SampleShape = tuple(sample_shape)
        # else:
        #     shape = _DEFAULT_SHAPE

        # self._shape = shape
        return shape

    @documents(IDataset)
    def trace(self) -> Dict[str, Any]:
        """Compute a trace of how this dataset was created

        Returns:
            Dict[str, Any] -- A recursively nested dict with creation info
        """
        return {
            "operation_name": self._operation_name,
            "operation_parameters": self._operation_parameters,
            "parent": self._parent.trace(),
        }

    @documents(IDataset)
    def reorder(
        self, key_or_list: Union[ElemKey, Sequence[ElemKey]], *rest_keys: ElemKey
    ):
        """Reorder items in the dataset (similar to numpy.transpose).

        Examples

        .. code-block::
            >>> ds = Dataset.from_iterable([(1,10),(2,20)])
            >>> ds.reorder(1,0) == [(10,1),(20,2)]
            >>> ds.reorder([1,0]) == [(10,1),(20,2)]
            >>> ds.reorder((1,0)) == [(10,1),(20,2)]
            >>> ds.reorder("two","one") == [(10,1),(20,2)]
            ...

        Arguments:
            key_or_list {Union[ElemKey, Sequence[ElemKey]]} -- first element can be a single elem key (int or str) or a sequence of elem keys
            rest_keys {ElemKey} -- remaining elements are elem keys

        Returns:
            [Dataset] -- Dataset with items whose elements have been reordered
        """
        inds = inds_from(self._elem_name2index, collect_keys(key_or_list, rest_keys))

        for i in inds:
            if i > len(self.shape):
                raise IndexError(
                    (
                        "reorder index {} is out of range"
                        "(maximum allowed index is {})"
                    ).format(i, len(self.shape))
                )

        def item_transform_fn(item: Tuple):
            return tuple([item[i] for i in inds])

        elem_name2index: ElemNameToIndex = {}
        if self._elem_name2index:
            if len(set(inds)) < len(inds):
                warn(
                    "discarding item_names due to otherwise non-unique labels on "
                    "transformed dataset"
                )
            else:
                elem_name2index = {
                    lbl: inds.index(idx)
                    for lbl, idx in self._elem_name2index.items()
                    if idx in inds
                }

        return Dataset(
            parent=self,
            transform_func=item_transform_fn,
            elem_name2index=elem_name2index,
            operation_name="reorder",
            operation_parameters={"keys": inds},
        )

    @documents(IDataset)
    def transform(
        self,
        key_or_sampletransform: Union[ElemKey, SampleTransform],
        elem_transform: ElemTransform = lambda e: True,
    ) -> "IDataset":
        """Transform sample elements using a user-defined function

        Examples:
        .. code-block::
            >>> ds = Dataset.from_iterable([(1,10),(2,20),(3,30)]).named("ones", "tens")
            >>> ds.transform(0, lambda elem: elem*100) == [(100,10),(200,20),(300,30)]
            >>> ds.transform("ones", lambda elem: elem*100) == [(100,10),(200,20),(300,30)]
            >>> ds.transform(lambda sample: tuple(sample[0]+sample[1])) == [(11),(22),(33)]

        Arguments:
            key_or_sampletransform {Union[ElemKey, SampleTransform]} -- either a key (string or index) or a function transforming the whole sample
            elem_transform {Optional[ElemTransform]} -- function that transforms the element selected by the key

        Returns:
            {Dataset} -- dataset
        """
        if callable(key_or_sampletransform):
            elem_idx = None
            transform_func: SampleTransform = key_or_sampletransform  # type:ignore
            operation_parameters = {"transform_func": signature(transform_func)}
        else:
            elem_idx = index_from(self._elem_name2index, key_or_sampletransform)
            transform_func: SampleTransform = lambda sample: tuple(
                (elem_transform(elem) if i == elem_idx else elem)
                for i, elem in enumerate(sample)
            )
            operation_parameters = {
                "key": elem_idx,
                "transform_func": signature(transform_func),
            }

        # reset - can we infer that it should be passed on?
        elem_name2index = self._elem_name2index if elem_idx else {}
        stats = self._stats if elem_idx else {}

        return Dataset(
            parent=self,
            transform_func=transform_func,
            operation_name="transform",
            operation_parameters=operation_parameters,
            elem_name2index=elem_name2index,
            stats=stats,
        )
