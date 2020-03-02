"""This module defines the structure of datasets.
"""

from ABC import abc
from Types import List

class Dataset(abc):
    """Abstract base class defining a generic dataset interface.
    """

    def get(self, idx : int):
        """Returns the element at the specified index
        
        Parameters
        ----------
        idx : int
            the index from which to read the sample.
        """
        pass

    def split(self,fractions):
        """Splits the dataset using the specified fractions
        
        Parameters
        ----------
        fractions : List{float}
            A list of real values determining the relative size of each split.
        """
        pass

    def whiten(self) -> Dataset:
        """ Whiten the dataset
        """
        pass
    

    def describe(self):
        """Returns a summary of the dataset.
        """
        pass

    def plot(self, idx : int = None, n_samples : int = 1):
        """Plots the data. The concrete behavior is determined by the type of data in the dataset.
        
        Parameters
        ----------
        idx : int, optional
            [description], by default None
        n_samples : int, optional
            [description], by default 1
        """
        pass