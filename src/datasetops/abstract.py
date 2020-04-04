from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Dict
from datasetops.transformation_graph import TransformationGraph


class ItemGetter(ABC):
    @abstractmethod
    def __getitem__(self, i: int) -> Tuple:
        pass  # pragma: no cover


class AbstractDataset(ItemGetter):
    """Abstract base class defining a generic dataset interface."""

    def __init__(self):
        pass  # pragma: no cover

    name = ""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of elements in the dataset."""
        pass  # pragma: no cover

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple:
        """Returns the element at the specified index.

        Parameters
        ----------
        idx : int
            the index from which to read the sample.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _get_origin(self) -> Union[List[Dict], Dict]:
        """Returns parent Dataset or list of parent Datasets and description
        of how it was made
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
