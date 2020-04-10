Transforms
==========


Ordering
--------

Shuffle
~~~~~~~
Shuffles a dataset such that samples are returned in random order when read.

.. doctest::

    >>> ds_shuffled = ds_mnist.shuffle(seed=0)
    >>> ds_mnist[0] == ds_mnist[0]
    True
    >>> ds_mnist[0] == ds_shuffled[0]
    False


Split
~~~~~
Divides a dataset into several smaller sets, each containing a specific fraction of all samples.
This may be used for creating a training and validation split.

.. doctest::

    >>> train, val = ds_mnist.split([0.5,0.5])
    >>> len(train) == len(ds_mnist)/2
    True
    >>> len(val) == len(ds_mnist)/2
    True

Split Filter
~~~~~~~~~~~~
Splits the dataset based on a predicate evaluated on each sample.
For example the MNIST dataset may be split into the samples corresponding to zero and all others.

.. doctest::

    >>> def func(s):
    >>>     return s.lbl == 0
    >>>
    >>> zeros, others = ds_mnist.split_filter(func)
    >>> all([s.lbl == 0 for s in zeros])
    True

Changing Data
-------------

Transform
~~~~~~~~~
Applies an user defined transform to each sample of a dataset.
The function must take an sample as argument and in turn return a new sample.

.. doctest::

    >>> def func1(s):
    >>>     img, lbl = s
    >>>     img += np.Random.randn(img.shape)
    >>>     return (img,lbl)
    >>> 
    >>> def func2(img):
    >>>     return img + np.Random.randn(img.shape)
    >>>
    >>> ds1 = ds_mnist.transform(func1)
    >>> ds2 = ds_mnist.transform("img",func2)
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
    >>> def func2(img):
    >>>     return [img,img]
    >>>
    >>> ss1 = ds_mnist.subsample(func1)
    >>> ss2 = ds_mnist.subsample("img", func2)
    >>> ss3 = ds_mnist.subsample(func1, n="eager")
    >>> ss4 = ds_mnist.subsample(func1, n="sample")
    >>> ss4 = ds_mnist.subsample(func1, n=4)
    True

The function can be called in several ways as shown in the example.
In the first case, the entire sample is passed to the supplied function.
In the second case, the first argument specifies that only the *img*-item is to be subsampled.
This results in only the image being passed as an argument to the function. 
The items which are not specified remain untouched, e.g. the first and second case are equivalent.

To define the number of samples in the new dataset, the number of subsamples per sample must be specified.
This can be done in one of three ways, by doing the subsampling eagerly on all samples, 
by performing subsampling on a single sample or by specifying the number of subsamples per sample.
In case the number of subsamples per sample may vary based on the concrete sample the first option should be used.

.. The difference between the :meth:`transform <datasetops.dataset.Dataset.transform>` and :func:`subsample <datasetops.dataset.subsample>` methods, 
.. is that the former modifies the sample itself, but not the number of samples, whereas the latter is allowed to do both.

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

