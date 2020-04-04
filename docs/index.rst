Dataset Ops documentation
=========================
Friendly dataset operations for your data science needs.
Dataset Ops provides declarative loading, sampling, splitting and transformation operations for datasets, alongside export options for easy integration with Tensorflow and PyTorch.

.. image:: pics/pipeline.svg
   :width: 1000
   :alt: Dataset Ops pipeline


First Steps
-----------
Are you looking for ways to install the framework 
or do you looking for inspiration to get started?

* **Installing**:
   :doc:`Installing <installing>`

* **Getting Started**:
   :doc:`Getting started <getting_started>`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started:

   installing
   getting_started

Loaders and Transforms
----------------------

Get an overview of the available loaders and transforms that can be used with your dataset.

* **Loaders**:
   :doc:`Standard loaders <loaders_standard>`

* **Transforms**:
   :doc:`General <transforms_common>` |
   :doc:`Image <transforms_images>` |
   :doc:`Time-series <transforms_timeseries>`

It is also possible to implement your own loaders and transforms.

* **User-Defined**:
   :doc:`Loaders <transforms_timeseries>`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Loaders and Transforms
   
   loaders_standard
   transforms_common
   transforms_images
   transforms_timeseries

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: How-to guides:

   howto_new_loader
   howto_new_transform

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Performance

   optimizations

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contributing:

   continuous_integration
   documentation_structure

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference:

   autoapi/index