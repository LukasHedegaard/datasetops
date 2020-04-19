from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import typing

from datasetops.transformation_graph import TransformationGraph

Shape = Sequence[int]
IdIndex = int
Id = int
Ids = Sequence[Id]

"""Represents a single index or a slice"""
IdxSlice = Union[int, slice]

"""Represents a sample from a dataset"""
Sample = Tuple

Data = Any
ItemTransformFn = Callable[[Any], Any]
DatasetTransformFn = Callable[[int, "AbstractDataset"], "AbstractDataset"]

DatasetTransformFnCreator = Union[
    Callable[[], DatasetTransformFn], Callable[[Any], DatasetTransformFn]
]
AnyPath = Union[str, Path]
DataPredicate = Callable[[Any], bool]
Key = Union[int, str]

"""Something"""
ItemNames = Dict[str, int]


class ItemGetter(ABC):
    """Abstract base class implemented by classes that implement
    an index based get method
    """

    @abstractmethod
    def __getitem__(self, i: int) -> Tuple:
        pass  # pragma: no cover

    @abstractmethod
    def __len__(self):
        pass  # pragma: no cover


class AbstractDataset(ItemGetter):
    """Abstract base class defining a generic dataset interface."""

    def __init__(self):
        pass  # pragma: no cover

    name = ""
    shape = None
    _cacheable = False
    _origin = None

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of elements in the dataset."""
        pass  # pragma: no cover

    @typing.overload
    def __getitem__(self, idx: slice) -> List[Sample]:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        """Returns the element at the specified index.

        Arguments:
            idx {int} -- the index from which to read the sample.

        Returns:
            Tuple -- A tuple representing the sample
        """
        pass  # pragma: no cover

    def get_transformation_graph(self) -> TransformationGraph:
        """Returns TransformationGraph of current dataset
        """
        return TransformationGraph(self)

    def __iter__(self):
        for i in range(self.__len__()):

            yield self.__getitem__(i)

    @property
    def generator(self,):
        def g():
            for d in self:
                yield d

        return g
