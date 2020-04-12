Common Transforms
==================



Shuffle
-------
Shuffles a dataset such that samples are returned in random order when read.

.. doctest::

    >>> ds = do.load_mnist()
    >>> ds_s = ds.shuffle(seed=0)
    >>> ds.inds != ds.inds
    False


Split
-----
Divides a dataset into several smaller sets, each containing a specific fraction of all samples.
This may be used for creating a training and validation split.

.. doctest::

    >>> train, val = do.load_mnist().split([0.7,0.3])
    >>> len(train) == 1000
    True
    >>> len(val) == 300
    True

Split Filter
------------
Splits the dataset based on a predicate evaluated on each sample.
For example the MNIST dataset may be split into the samples corresponding to zero and all others.

.. doctest::

    >>> def func(s):
    >>>     return s.lbl == 0
    >>>
    >>> zeros, others = do.load_mnist().split_filter(func)
    >>> all([s.lbl == 0 for s in zeros])
    True

Transform
---------
Applies an user defined transform to each sample of the dataset.

.. doctest::

    >>> def func(s):
    >>>     return someFunc(s)
    >>>
    >>> train, val = do.load_mnist().transform(TODO)
    >>> TODO
    True
