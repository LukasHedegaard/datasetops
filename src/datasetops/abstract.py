"""
This module defines a generic interface for datasets.

Examples
--------
A simple dataset can be implemented as::

 >>> class DummyDataset(AbstractDataset):
 ...    def __init__(self, size):
 ...       self.size = size
 ...
 ...    def __len__(self):
 ...       return self.size
 ...
 ...    def __getitem__(self, key):
 ...       return list(range(self.size))[key]
 ...
 >>> ds = DummyDataset(10)
 >>> len(ds)
 10
 >>> ds[0]
 0
 >>> ds[0:5]
 [0, 1, 2, 3, 4]

"""

from abc import ABC, abstractmethod
from typing import Tuple
from datasetops.transformation_graph import TransformationGraph


class ItemGetter(ABC):
    """Abstract base class implemented by classes that implement
    an index based get method
    """

    @abstractmethod
    def __getitem__(self, i: int) -> Tuple:
        pass  # pragma: no cover


class AbstractDataset(ItemGetter):
    """Abstract base class defining a generic dataset interface."""

    def __init__(self):
        pass  # pragma: no cover

    name = ""
    cachable = False
    shape = None
    _origin = None

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of elements in the dataset."""
        pass  # pragma: no cover

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple:
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


class DummyDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return list(range(self.size))[key]


ds = DummyDataset(10)
