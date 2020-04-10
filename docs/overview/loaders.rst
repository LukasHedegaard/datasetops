Loaders
=======

Dataset Ops provides a set of standard loaders that covers loading of the most frequently exchange formats.

PyTorch
-------

Tensorflow
----------

Comma-separated values (CSV)
----------------------------

CSV is a format commonly used by spreadsheet editor to store tabular data.
Consider the scenario where the data describes the correlation between speed and vibration
under some specific load for two different car models, referred to as *car1* and *car2*.

For two experiments the folder structure may look like:

.. code-block::

    cars_csv
    ├── car1
    │   ├── load_1000.csv
    │   └── load_2000.csv
    └── car2
        ├── load_1000.csv
        └── load_2000.csv

The contents of each file may look like:

.. code-block::

    speed,vibration
    1,0.5
    2,1
    3,1.5

The :func:`load_csv <datasetops.loaders.load_csv>` function allows either a single or multiple CSV files to be loaded.

.. note::

    CSV is not standardized, rather it refers to a *family* of related formats, each differing slightly and with their own quirks.
    Under the hood the framework relies on Pandas's `read_csv <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>`__ implementation.

Single File
~~~~~~~~~~~
To load a single CSV file the path of the file is passed to the function.

.. doctest::

    >>> ds = do.load_csv("car1/load_1000.csv")
    >>> len(ds)
    3
    >>> ds[0]
    Empty DataFrame
    Columns: []
    Index: []
    >>> ds[0].shape
    (1,2)

It is possible to specify whether a sample should be generated from each row, or only a single sample should be produced from the entire file.

.. doctest::

    >>> ds = do.load_csv("car1/load_1000.csv", single_sample=True)
    >>> len(ds)
    1
    >>> ds[0].shape
    (3,2)

Finally, it is possible to pass a function to transform the raw data into a sample.
The function must take the path and the raw data as argument and in turn return a new sample:

.. doctest::

    >>> def func(path,data):
    >>>     load = int(path.stem.split("_")[-1])
    >>>     return (data,load)
    >>> ds = do.load_csv("car1/load_1000.csv",func)
    >>> ds[0][1]
    1000

This useful for converting the data into other formats or to extract labels from the name of the CSV file.

Multiple Files
~~~~~~~~~~~~~~
The process of loading multiple files is similar. 
However, instead of specifying a single CSV file, a directory containing the CSV files must be specified instead.
This will search recursively for CSV files creating a sample for each file.

.. doctest::

    >>> ds = load_csv("cars_csv")

