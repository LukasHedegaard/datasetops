Implementing A New Loader
=========================

In case the format of your dataset does not fit any of the standard loaders, it is possible to define your own custom loader.
By defining a custom loader you dataset can be integrated with the framework allowing transformations to be applied to its data, just like a standard loader.

To define a new loader a new class must be created that implements the interface declared by :class:`AbstractDataset <datasetops.abstract.AbstractDataset>`.
In the context of the library a dataset is an 

