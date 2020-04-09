Transforms
==========


Ordering
--------

Shuffle
~~~~~~~
Shuffles a dataset such that samples are returned in random order when read.

.. doctest::

    >>> ds = do.load_mnist()
    >>> ds_s = ds.shuffle(seed=0)
    >>> ds.inds != ds.inds
    False


Split
~~~~~
Divides a dataset into several smaller sets, each containing a specific fraction of all samples.
This may be used for creating a training and validation split.

.. doctest::

    >>> train, val = do.load_mnist().split([0.7,0.3])
    >>> len(train) == 1000
    True
    >>> len(val) == 300
    True

Split Filter
~~~~~~~~~~~~
Splits the dataset based on a predicate evaluated on each sample.
For example the MNIST dataset may be split into the samples corresponding to zero and all others.

.. doctest::

    >>> def func(s):
    >>>     return s.lbl == 0
    >>>
    >>> zeros, others = do.load_mnist().split_filter(func)
    >>> all([s.lbl == 0 for s in zeros])
    True

Changing Data
-------------

Transform
~~~~~~~~~
Applies an user defined transform to each sample of the dataset.

.. doctest::

    >>> def func(s):
    >>>     return someFunc(s)
    >>>
    >>> train, val = do.load_mnist().transform(TODO)
    >>> TODO
    True

Subsample
~~~~~~~~~
For some applications it may be useful to convert each sample into several smaller sub-samples.
For example, a sample may be a time-series stretching over a large time interval, which needs to be split into several series of shorter length.

To subsample a dataset the :func:`subsample <datasetops.dataset.subsample>` method is called with a function that describes
how each sample should be divided. This function must return an iterable consisting of the new samples as seen below:

.. doctest::

    >>> def func1(s):
    >>>     img, lbl = s
    >>>     return [(img,lbl),(img,lbl)]
    >>>    
    >>> def funct2(img):
    >>>     return [img,img]
    >>>
    >>> ss1 = ds_mnist.subsample(func1)
    >>> ss2 = ds_mnist.subsample("img", func2)
    >>> ss3 = ds_mnist.subsample(func1, n=4)
    >>> ss4 = ds_mnist.subsample("img", func2, n=4)
    True

The function can be called in several ways as shown in the example.
In the first case, the entire sample is passed to the supplied function.
In the second case, the *img* item is specified which 


The difference between the :meth:`transform <datasetops.dataset.Dataset.transform>` and :func:`subsample <datasetops.dataset.subsample>` methods, 
is that the former modifies the sample itself, but not the number of samples, whereas the latter is allowed to do both.


Images Manipulation
-------------------

Convolves the images in the dataset with the specified filter.

.. doctest::

    >>> kernel = np.ones((5,5))/(5*5)
    >>> do.load_mnist().image_filter(kernel)
    TODO

Resize
~~~~~~
Resize the images of the dataset to a specified size.

    >>> do.load_mnist().resize((10,10))
    >>> s = next(do)
    >>> assert np.shape(s.image) == (10,10)


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

