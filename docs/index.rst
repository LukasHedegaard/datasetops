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

* **Installing**: :doc:`Installing <installing>`

* **Getting Started**: :doc:`Getting started <getting_started>`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started:

   installing
   getting_started

Loaders and Transforms
----------------------

Get an overview of the available loaders and transforms that can be used with your dataset.

* **Loaders**: :doc:`Standard loaders <loaders/standard>`

* **Transforms**: :doc:`General <transforms/common>` | :doc:`Image <transforms/images>` | :doc:`Time-series <transforms/timeseries>`

It is also possible to implement your own loaders and transforms.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Loaders and Transforms
   
   loaders/standard
   transforms/common
   transforms/images
   transforms/timeseries

Custom Loaders and Transforms
-----------------------------

Is your dataset structured in a way thats not compatible with any standard loaders? 
Or does your application require  very specific and complex transformations to be applied to the data?
The framework makes integration with custom loaders and transforms easy and clean.
For how-to guides on how to do this see:

* **User-Defined**: :doc:`Loaders <howto/custom_loader>` | :doc:`Transforms <howto/custom_transform>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: How-to guides:

   howto/custom_loader
   howto/custom_transform

Performance And Optimizations
-----------------------------

Are you looking for ways to reduce the time required to load and process big datasets?
The library provides several mechanisms that can drastically reduce the time required.

* **Increasing performance**: :doc:`Caching <optimization/caching>` | :doc:`Multiprocessing <optimization/multiprocessing>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Performance

   optimization/caching
   optimization/multiprocessing

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference:

   autoapi/index

Developer And Contributor Guide
--------------------------------

Are you looking to contribute to the project or are you already a developer?
Contributions of any size and form are always welcomed.
Information on how to the codebase is tested, how it is published, and how to add documentation can below:

* **Quality Assurance And CI:** :doc:`Testing <development/testing>` | :doc:`Git Workflow <development/git_workflow>` | :doc:`CI <development/ci>`
   
* **How To Contribute:** :doc:`Communication channels <development/communication>` | :doc:`Writing documentation <development/writing_docs>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contributing:

   development/communication
   development/codebase
   development/testing
   development/git_workflow
   development/ci
   development/writing_docs


