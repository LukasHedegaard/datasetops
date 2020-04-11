"""
Dataset Ops is a library that enables the loading and processing of datasets stored in various formats.
It does so by providing::
1. Loaders for various storage formats
2. Transformations which may chained to transform the data into the desired form.



Finding The Documentation
-------------------------
Documentation is available online at:

https://datasetops.readthedocs.io/en/latest/


"""


from .dataset import (
    Dataset,
    allow_unique,
    reshape,
    categorical,
    one_hot,
    categorical_template,
    numpy,
    image,
    image_resize,
    zipped,
    cartesian_product,
    concat,
    to_tensorflow,
    to_pytorch,
)

from .loaders import (
    Loader,
    from_pytorch,
    from_folder_data,
    from_folder_class_data,
    from_folder_group_data,
    from_folder_dataset_class_data,
    from_folder_dataset_group_data,
    from_mat_single_mult_data,
)
