"""
Dataset Ops is a library that enables the loading and processing of
datasets stored in various formats.
It does so by providing::
1. Loaders for various storage formats
2. Transformations which may chained to transform the data into the desired form.



Finding The Documentation
-------------------------
Documentation is available online at:

https://datasetops.readthedocs.io/en/latest/


"""


from .dataset import Dataset  # noqa: F401

from .loaders import from_iterable  # noqa: F401
