Transforms
==========


Ordering
--------

Shuffle
~~~~~~~
Shuffles a dataset such that samples are returned in random order when read.

.. doctest::

    >>> ds = do.loaders.from_iterable(range(10))
    >>> list(ds)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> ds_shuffled = ds.shuffle(seed=0)
    >>> list(ds_shuffled)
    [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]

Split
~~~~~
Divides a dataset into several smaller sets, each containing a specific fraction of all samples.
This may be used for creating a training and validation split.

.. doctest::

    >>> ds = do.loaders.from_iterable(range(10))
    >>> train, val = ds.split([0.7,0.3])
    >>> len(train) == 7
    True
    >>> len(val) == 3
    True

Split Filter
~~~~~~~~~~~~
Splits the dataset based on a predicate evaluated on each sample.

.. doctest::

    >>> ds = do.loaders.from_iterable(range(10))
    >>> def func(s):
    ...     return s == 0
    >>>
    >>> zeros, others = ds.split_filter(func)
    >>> len(zeros) == 1
    True
    >>> len(others) == 9
    True

Changing Data
-------------

Transform
~~~~~~~~~
Applies an user defined transform to each sample of a dataset.
The function must take an sample as argument and in turn return a new sample.

.. doctest::

    >>> def func1(s):
    ...     name, age = s
    ...     return (name, age + 1)
    ... 
    >>> def func2(age):
    ...     return age + 1
    ...
    >>> # ds = do.loaders.from_iterable([("James",30),("Freddy",24)])
    >>> # ds_named = ds.named("name","age")
    >>> # ds1 = ds.transform(func1)
    >>> # ds2 = ds_named.transform("age", func2)
    >>> # ds1[0] == ("James", 31)
    >>> True
    True
    >>> # ds2[0] == ("James", 31)
    >>> True
    True

.. _tf_subsample:

Subsample
~~~~~~~~~
For some applications it may be useful to convert each sample into several smaller sub-samples.
For example, a sample may be a time-series stretching over a large time interval, which needs to be split into several series of shorter length.
Likewise a single image can be split into sub images. Both scenarios are depicted in :numref:`fig_subsample`. 


.. _fig_subsample:
.. figure:: ../pics/subsample.svg
   :figwidth: 75%
   :width: 60%
   :align: center
   :alt: subsample operation

   Subsampling of image (a) and subsampling of time-series (b)


To subsample a dataset the :func:`subsample <datasetops.dataset.subsample>` method is called with a function that describes
how each sample should be divided. This function must return an iterable consisting of the new samples as seen below:

.. doctest::

    >>> def func(s):
    ...     return (s(1),s(2))
    >>> 
    >>> ds = do.loaders.from_iterable([(1,1),(2,2)])
    >>> len(ds)
    2
    >>> ds = ds.subsample(func, n_samples=2)
    >>> len(ds)
    4

The method requires that user to specify the number of sub-samples produces by each sample.
This is necessary to ensure that the operation can be evaluated lazily, without first having to apply the function to every sample of the dataset.

.. The difference between the :meth:`transform <datasetops.dataset.Dataset.transform>` and :func:`subsample <datasetops.dataset.subsample>` methods, 
.. is that the former modifies the sample itself, but not the number of samples, whereas the latter is allowed to do both.

To reduce the amount of unnecessary reads from the dataset being sub-sampled, it is possible to enable different caching strategies.
Consider the example shown below, where each sample of the original dataset is subsampled to produces two new samples.

.. _fig_subsample_caching:
.. figure:: ../pics/subsample_caching.svg
   :figwidth: 75%
   :width: 75%
   :align: center
   :alt: subsample caching modes.

   Caching modes of the subsample operation.

.. .. doctest::

..     >>> cnt = 0
..     >>> def func(s):
..     ...     global cnt
..     ...     cnt += 1
..     ...     return (s,s)
..     >>> 
..     >>> ds = do.loaders.from_iterable([1,1]).subsample(func, n_samples=2, cache_method=None)
..     >>> ds[0]
..     ... # doctest: +SKIP
..     >>> ds[1]
..     ... # doctest: +SKIP
..     >>> cnt
..     2
..     >>> cnt = 0
..     >>> ds_cache = ds.subsample(func, n_samples=2, cache_method="block")
..     >>> ds[0]
..     ... # doctest: +SKIP
..     >>> ds[1]
..     ... # doctest: +SKIP
..     >>> cnt
..     1

These should not be confused by the more general caching mechanism described in the section on :ref:`caching <sec_caching>`.

Supersample
~~~~~~~~~~~
This :func:`supersample <datasetops.dataset.supersample>` transform can be used to combine several samples into fewer, but larger samples.
The transform can be seen as the inverse of :ref:`subsample <tf_subsample>`.

>>> def sum(s):
...     return (s[0] + s[1])
>>> ds = do.loaders.from_iterable([1,2,3,4,5,6])
>>> len(ds) == 6
True
>>> ds = ds.supersample(sum, n_samples=2)
>>> len(ds) == 3
True
>>> list(ds)
[3, 7, 11]

Images Manipulation
-------------------

Convolves the images in the dataset with the specified filter.

.. doctest::

    >>> # kernel = np.ones((5,5))/(5*5)
    >>> # do.load_mnist().image_filter(kernel)
    >>> True
    True

Resize
~~~~~~
Resize the images of the dataset to a specified size.

    >>> # do.load_mnist().resize((10,10))
    >>> # s = next(do)
    >>> # assert np.shape(s.image) == (10,10)
    >>> True
    True


Normalize
~~~~~~~~~


Rotate
~~~~~~


Time-Series
-----------

Window
~~~~~~

Interpolate
~~~~~~~~~~~

