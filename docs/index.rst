.. ml-datasets documentation master file, created by
   sphinx-quickstart on Thu Feb 27 11:02:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ML-Datasets's documentation!
=======================================
ML-Datasets is a library which provides a framework-independent way of loading and processing data for machine learning.
Its highligts are:

* Provides loaders for commonly used data formats.

* Supports Tensorflow, PyTorch and Keras out of the box.

* Provides methods for creating test and validation splits.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installing
----------

The library can be obtained either as a package through PyPI and Conda, or built from source.

To install through a package manager use:

.. code-block:: bash

   pip install mldatasets
   conda install mldatasets
   
The package can be installed from source by cloning the git repository:

.. code-block:: bash
   
   git clone https://github.com/LukasHedegaard/ml-datasets.git

And then running pip install in the root of the folder:

.. code-block:: bash
   
   pip install .

This will install the package in the current environment.

.. toctree::
   :maxdepth: 2


Examples
--------

.. code-block:: python

   from mldatasets.loaders import load_dataset
   
   ds = load_dataset("mydataset").to_torch()

