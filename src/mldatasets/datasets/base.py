from abc import ABC, abstractmethod

from typing import List, Tuple, Any, Union

Subscriptable = Union[List, Tuple]


class ItemGetter(ABC):
    @abstractmethod
    def __getitem__(self, i:int):
        pass


class BaseDataset(ItemGetter):
    """Abstract base class defining a generic dataset interface.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of elements in the dataset
        """
        pass


    @abstractmethod
    def __getitem__(self, idx : int) -> Any:
        """Returns the element at the specified index
        
        Parameters
        ----------
        idx : int
            the index from which to read the sample.
        """
        pass


    # @abstractmethod
    # def describe(self):
    #     """Returns a summary of the dataset.
    #     """
    #     pass


    @abstractmethod
    def sample(self, num):
        """Samples a number of samples from the dataset 
        
        Parameters
        ----------
        fractions : List{float}
            A list of real values determining the relative size of each split.
        """
        pass


    def sample_classwise(self, num_per_class: int):
        """Samples the dataset with a desired number of samples per class
        
        Parameters
        ----------
        fractions : num_per_class{int}
            Number of samples per class
        """
        pass


    # @abstractmethod
    # def split(self, fractions:Subscriptable[float]) -> List[BaseDataset]: #type: ignore
    #     """Splits the dataset using the specified fractions
        
    #     Parameters
    #     ----------
    #     fractions : {Subscriptable[float]}
    #         A list of real values determining the relative size of each split.
    #     """
    #     pass


    # @abstractmethod
    # def center(self, mean=None) -> Dataset: #type: ignore
    #     """ Whiten the dataset
    #     """
    #     pass


    # @abstractmethod
    # def normalize(self, mean=None, std=None) -> Dataset: #type: ignore
    #     """ Normalize the dataset
    #     """
    #     pass


    # @abstractmethod
    # def plot(self, idx : int = None, n_samples : int = 1):
    #     """Plots the data. The concrete behavior is determined by the type of data in the dataset.
        
    #     Parameters
    #     ----------
    #     idx : int, optional
    #         [description], by default None
    #     n_samples : int, optional
    #         [description], by default 1
    #     """
    #     pass