Getting started
===============

Before getting started with loading and processing datasets it is useful to have an overview of what the framework provides and its intended workflow.
As depicted in :numref:`fig_pipeline`, the framework provides a pipeline for processing the data by composing a chains of operations applied to the dataset.

.. _fig_pipeline:
.. figure:: ../pics/pipeline.svg
   :figwidth: 600
   :align: center
   :alt: Dataset Ops pipeline

   Dataset Ops Pipeline.

At the beginning of this chain is a *loader* which implements the process of reading a dataset stored in some specific file format.
Following this the raw data can then processed into a desired from by applying a number of transformations, independently of the underlying storage format.
After applying the transformations to the dataset it can be used as is or it can be converted into a type compatible with either PyTorch or TensorFlow.


An overview of the available loaders and transforms can be found in:

.. doctest::

   >>> from datasetops.external_datasets import load_mnist
   >>> ds = load_mnist()