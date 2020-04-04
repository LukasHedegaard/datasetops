.. datasetops documentation master file, created by
   sphinx-quickstart on Thu Feb 27 11:02:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dataset Ops documentation
=======================================
Friendly dataset operations for your data science needs.
Dataset Ops provides declarative loading, sampling, splitting and transformation operations for datasets, alongside export options for easy integration with Tensorflow and PyTorch.

.. image:: pics/pipeline.svg
   :width: 500
   :alt: Dataset Ops pipeline


.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installing
   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Loaders



.. toctree::
   :maxdepth: 2
   :caption: Transforms
   
   transforms_general.rst
   transforms_images.rst
   transforms_timeseries.rst

.. toctree::
   :maxdepth: 2
   :caption: How-to guides:

   howto_new_loader.rst
   howto_new_transform.rst

.. toctree::
   :maxdepth: 2
   :caption: Contributing:

   continuous_integration.rst
   documentation_structure.rst

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   autoapi/index