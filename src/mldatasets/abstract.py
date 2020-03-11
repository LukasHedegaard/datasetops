from abc import ABC, abstractmethod

from typing import List, Tuple, Any, Union


class ItemGetter(ABC):
    @abstractmethod
    def __getitem__(self, i:int):
        pass # pragma: no cover


class AbstractDataset(ItemGetter):
    """Abstract base class defining a generic dataset interface.
    """
    
    def __init__(self):
        pass # pragma: no cover
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of elements in the dataset
        """
        pass # pragma: no cover


    @abstractmethod
    def __getitem__(self, idx : int) -> Any:
        """Returns the element at the specified index
        
        Parameters
        ----------
        idx : int
            the index from which to read the sample.
        """
        pass # pragma: no cover


    @abstractmethod
    def __iter__(self) -> Any:
        pass # pragma: no cover


    # @abstractmethod
    # def describe(self):
    #     """Returns a summary of the dataset.
    #     """
    #     pass # pragma: no cover


    @abstractmethod
    def sample(self, num, seed:int=None):
        """Samples a number of samples from the dataset 
        
        Parameters
        ----------
        fractions : List{float}
            A list of real values determining the relative size of each split.
        """
        pass # pragma: no cover


    @abstractmethod
    def sample_classwise(self, num_per_class:int, seed:int=None):
        """Samples the dataset with a desired number of samples per class
        
        Parameters
        ----------
        fractions : num_per_class{int}
            Number of samples per class
        """
        pass # pragma: no cover


    @abstractmethod
    def split(self, fractions:List[float], seed:int=None): 
        """Splits the dataset using the specified fractions
        
        Parameters
        ----------
        fractions : {Subscriptable[float]}
            A list of real values determining the relative size of each split.
        """
        pass # pragma: no cover


    @abstractmethod
    def shuffle(self, seed:int=None): 
        """Splits the dataset using the specified fractions
        
        Parameters
        ----------
        fractions : {Subscriptable[float]}
            A list of real values determining the relative size of each split.
        """
        pass # pragma: no cover


    # @abstractmethod
    # def center(self, mean=None) -> Dataset: #type: ignore
    #     """ Whiten the dataset
    #     """
    #     pass # pragma: no cover


    # @abstractmethod
    # def normalize(self, mean=None, std=None) -> Dataset: #type: ignore
    #     """ Normalize the dataset
    #     """
    #     pass # pragma: no cover


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
    #     pass # pragma: no cover