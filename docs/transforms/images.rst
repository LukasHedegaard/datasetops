Image Transforms
================

Image Filter
------------

Convolves the images in the dataset with the specified filter.

.. doctest::

    >>> kernel = np.ones((5,5))/(5*5)
    >>> do.load_mnist().image_filter(kernel)
    TODO

Resize
------
Resize the images of the dataset to a specified size.

    >>> do.load_mnist().resize((10,10))
    >>> s = next(do)
    >>> assert np.shape(s.image) == (10,10)


Normalize
---------


Rotate
------



