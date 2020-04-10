"""
This module provides a number of commonly used datasets. 
These may be used as reference or to experiment with various preprocessing steps.
"""

from collections import namedtuple

import numpy as np

from datasetops.types import DatasetTransformFnCreator
from datasetops.abstract import AbstractDataset
from datasetops.dataset import Dataset
from datasetops.loaders import Loader

MnistSample = namedtuple("MnistSample", ["img", "lbl"])


class MNIST(Loader):
    """ Returns the dataset containing the data from the MNIST database of 70k hand written digits 0-9 and their associated labels.

    Each sample is a tuple of an image and a label.
    """

    def __init__(self):

        try:
            import mnist
        except Exception as e:
            raise RuntimeError(
                f"Unable to load MNIST dataset due to error: {e}")

        train_images = mnist.train_images()
        train_labels = mnist.train_labels()

        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        self.images = np.concatenate((train_images, test_images))
        self.labels = np.concatenate((train_labels, test_labels))
        n = self.images.shape[0]

        super().__init__(self._getdata, "MNIST")
        self.extend(np.arange(0, n-1))
        self.named("image", "label")

    def _getdata(self, idx):
        if(idx > len(self)-1):
            raise ValueError(
                f"Index: {idx} is out of range. The dataset only contains: {len(self)} samples.")

        img = self.images[idx, :]
        lbl = self.labels[idx]
        s = MnistSample(img, lbl)

        return s

    # def __len__(self):
    #     return self._len


def load_mnist() -> Dataset:
    """Returns the dataset containing the data from the
    MNIST database of 70k hand written digits 0-9 and their associated labels."""
    return MNIST()


if __name__ == "__main__":
    train, val = MNIST().shuffle().split([0.7, 0.3])
    s = train[0]
    img, lbl = s
    print(lbl)
    print(s.lbl)
