"""
Module defining loaders for several formats which are commonly used to exchange datasets.
Additionally, the module provides adapters for the dataset types used by various ML frameworks.
"""

from pathlib import Path
import os
import re
import warnings
from collections import namedtuple


import numpy as np
from scipy.io import loadmat

from datasetops.dataset import zipped
from datasetops.abstract import ItemGetter
from datasetops.dataset import Dataset
from datasetops.types import AnyPath
from typing import Callable, Any, Optional, Sequence, List, Dict, Tuple, Iterable


class Loader(Dataset):
    def __init__(
        self,
        getdata: Callable[[Any], Any],
        ids: Sequence[Any],
        identifier: Optional[str] = None,
        name: str = None,
    ):
        if not callable(getdata):
            raise TypeError("get_data should be callable")

        self.identifier = identifier

        class Getter(ItemGetter):
            def __getitem__(self, i: int):
                return getdata(i)

            def __len__(self):
                return len(ids)

        super().__init__(
            parent=Getter(),
            name=name,
            ids=ids,
            operation_name="load",
            operation_parameters={"identifier": identifier},
        )

        self._cacheable = self.identifier is not None


def from_iterable(iterable: Iterable, identifier: str = None) -> Dataset:
    """Creates a new dataset from the elements of the sequence.

    An iterable must must implement implement at least one of the following
    functions:
    __next__ or __getitem__

    Arguments:
        iterable {Iterable} -- an iterable containing the samples
        identifier {Optional[str]} -- unique identifier

    Returns:
        AbstractDataset -- a new dataset containing the elements of the iterable.
    """

    """
    https://nelsonslog.wordpress.com/2016/04/06/python3-no-len-for-iterators/
    https://gist.github.com/NelsonMinar/90212fbbfc6465c8e263341b86aa01a8
    It appears the most effective way of getting length of a iterable is to
    convert it to a tuple or list"""

    itr = [i for i in iterable]

    def getter(idx):
        nonlocal itr
        return itr[idx]

    ldr = Loader(getdata=getter, ids=range(len(itr)), identifier=identifier)
    return ldr


def from_pytorch(pytorch_dataset, identifier: Optional[str] = None):
    """Create dataset from a Pytorch dataset

    Arguments:
        tf_dataset {torch.utils.data.Dataset} -- A Pytorch dataset to load from
        identifier {Optional[str]} -- unique identifier

    Returns:
        [Dataset] -- A datasetops.Dataset

    """

    def get_data(i) -> Tuple:
        nonlocal pytorch_dataset
        item = pytorch_dataset[i]
        return tuple([x.numpy() if hasattr(x, "numpy") else x for x in item])

    ds = Loader(get_data, ids=range(len(pytorch_dataset)), identifier=identifier)
    return ds


def from_tensorflow(tf_dataset, identifier: Optional[str] = None):
    """Create dataset from a Tensorflow dataset

    Arguments:
        tf_dataset {tf.data.Dataset} -- A Tensorflow dataset to load from
        identifier {Optional[str]} -- unique identifier

    Raises:
        AssertionError: Raises error if Tensorflow is not executing eagerly

    Returns:
        [Dataset] -- A datasetops.Dataset

    """
    import tensorflow as tf

    if not tf.executing_eagerly():
        raise AssertionError(
            """Tensorflow must be executing eagerly
            for `from_tensorflow` to work"""
        )

    # We could create a mem which is filled up gradually, when samples are needed.
    #  However, then we would the to get the number of samples as a parameter
    # The latency using this solution seems to be acceptable
    tf_ds = list(tf_dataset)

    if type(tf_dataset.element_spec) == dict:
        keys = list(tf_dataset.element_spec.keys())
    elif hasattr(tf_dataset.element_spec, "__len__"):
        keys = list(range(len(tf_dataset.element_spec)))
    else:
        keys = [0]

    def get_data(i) -> Tuple:
        nonlocal tf_ds, keys
        tf_item = tf_ds[i]
        if not type(tf_item) in [list, tuple, dict]:
            tf_item = [tf_item]
        item = tuple(
            [
                tf_item[k].numpy() if hasattr(tf_item[k], "numpy") else tf_item[k]
                for k in keys
            ]
        )
        return item

    ds = Loader(get_data, ids=range(len(tf_ds)), identifier=identifier)
    if type(keys[0]) == str:
        ds = ds.named([str(k) for k in keys])
    return ds


def from_folder_data(path: AnyPath) -> Dataset:
    """Load data from a folder with the data structure:

    .. code-block::

        folder
        ├ sample1.jpg
        ├ sample2.jpg

    Arguments:
        path {AnyPath} -- path to folder

    Returns:
        Dataset -- A dataset of data paths,
                   e.g. ('nested_folder/class1/sample1.jpg')

    """
    p = Path(path)
    full_p = str(p.absolute())
    ids = [str(x.relative_to(p)) for x in p.glob("[!._]*")]

    def getdata(i) -> Tuple:
        nonlocal p
        return (str(p / i),)

    ds = Loader(
        getdata,
        ids,
        full_p,
        "Data Getter for folder with structure 'root/data' and path '{}'".format(
            full_p
        ),
    )

    return ds


def from_folder_class_data(path: AnyPath) -> Dataset:
    """Load data from a folder with the data structure:

    .. code-block::

        data
        ├── class1
        │   ├── sample1.jpg
        │   └── sample2.jpg
        └── class2
            └── sample3.jpg

    Arguments:
        path {AnyPath} -- path to nested folder

    Returns:
        Dataset -- A labelled dataset of data paths and corresponding class labels,
                   e.g. ('nested_folder/class1/sample1.jpg', 'class1')
    """
    p = Path(path)
    full_p = str(p.absolute())
    classes = [x for x in p.glob("[!._]*")]

    def get_data(i) -> Tuple:
        nonlocal p
        return (str(p / i), re.split(r"/|\\", i)[0])

    ids = []
    for c in classes:
        ids.extend([str(x.relative_to(p)) for x in c.glob("[!._]*")])

    ds = Loader(
        get_data,
        ids,
        full_p,
        "Data Getter for folder with structure 'root/classes/data' and path '{}'".format(
            full_p
        ),
    )

    return ds


def from_folder_group_data(path: AnyPath) -> Dataset:
    """Load data from a folder with the data structure:

    .. code-block::

        data
        ├── group1
        │   ├── sample1.jpg
        │   └── sample2.jpg
        └── group2
            ├── sample1.jpg
            └── sample2.jpg

    Arguments:
        path {AnyPath} -- path to nested folder

    Returns:
        Dataset -- A dataset of paths to objects of each groups zipped together with corresponding names,
                   e.g. ('nested_folder/group1/sample1.jpg', 'nested_folder/group2/sample1.txt')

    """
    p = Path(path)
    groups = [x for x in p.glob("[!._]*")]

    datasets = []

    for group in groups:
        ds = from_folder_data(group).named(group.name)

        datasets.append(ds)

    datasets.sort(key=lambda val: val.names[0])

    return zipped(*datasets)


def from_folder_dataset_class_data(path: AnyPath) -> List[Dataset]:
    """Load data from a folder with the data structure:

    .. code-block::

        data
        ├── dataset1
        │   ├── class1
        │   │   ├── sample1.jpg
        │   │   └── sample2.jpg
        │   └── class2
        │       └── sample3.jpg
        └── dataset2
            └── sample3.jpg

    Arguments:
        path {AnyPath} -- path to nested folder

    Returns:
        List[Dataset] -- A list of labelled datasets, each with data paths and corresponding class labels,
                         e.g. ('nested_folder/class1/sample1.jpg', 'class1')

    """
    p = Path(path)
    dataset_paths = sorted([x for x in p.glob("[!._]*")])
    return [from_folder_class_data(dsp) for dsp in dataset_paths]


def from_folder_dataset_group_data(path: AnyPath) -> List[Dataset]:
    """Load data from a folder with the data structure:

        nested_folder
        |- dataset1
            |- group1
                |- sample1.jpg
                |- sample2.jpg
            |- group2
                |- sample1.txt
                |- sample2.txt
        |- dataset2
            |- ...

    Arguments:
        path {AnyPath} -- path to nested folder

    Returns:
        List[Dataset] -- A list of datasets, each with data composed from different types,
                         e.g. ('nested_folder/group1/sample1.jpg', 'nested_folder/group2/sample1.txt')
    """
    p = Path(path)
    dataset_paths = sorted([x for x in p.glob("[!._]*")])
    return [from_folder_group_data(dsp) for dsp in dataset_paths]


def _dataset_from_np_dict(
    data: Dict[str, np.ndarray],
    data_keys: List[str],
    label_key: str = None,
    name: str = None,
    identifier: str = None,
) -> Dataset:
    all_keys = [*data_keys, label_key]
    shapes_list = [data[k].shape for k in data_keys]
    if label_key:
        shapes_list.append(data[label_key].shape)

    # search for common dimension
    all_shapes = list(set([i for l in shapes_list for i in l]))
    common_shapes = [s for s in all_shapes if all([s in l for l in shapes_list])]

    if len(common_shapes) > 1:
        warnings.warn(
            "Warning: More than one common shape found for mat dataset. Using the largest dimension as index"
        )

    common_shape = max(common_shapes)

    # reshape data to have the instance as first dimensions
    reshaped_data = {
        k: np.moveaxis(  # type:ignore
            data[k], source=s.index(common_shape), destination=0
        )
        for k, s in zip(all_keys, shapes_list)
    }

    # prep data getter
    def get_unlabelled_data(idx: int):
        nonlocal reshaped_data
        if len(data_keys) == 1:
            return reshaped_data[data_keys[0]][idx]
        else:
            return tuple([reshaped_data[k][idx] for k in data_keys])

    def get_labelled_data(idx: int):
        nonlocal reshaped_data
        return (get_unlabelled_data(idx), reshaped_data[label_key][idx])

    get_data = get_labelled_data if label_key else get_unlabelled_data

    # populate data getter
    ids = []
    if label_key:
        unique_labels = np.unique(reshaped_data[label_key])

        for lbl in unique_labels:
            lbl_inds = np.extract(  # type:ignore
                condition=reshaped_data[label_key].squeeze() == lbl,
                arr=reshaped_data[label_key].squeeze(),
            )
            ids.extend(lbl_inds)
    else:
        ids.extend(list(range(common_shape)))

    ds = Loader(get_data, ids, identifier, name=name)

    return ds


def from_mat_single_mult_data(path: AnyPath) -> List[Dataset]:
    """Load data from .mat file consisting of multiple data.

    E.g. a .mat file with keys ['X_src', 'Y_src', 'X_tgt', 'Y_tgt']

    Arguments:
        path {AnyPath} -- path to .mat file

    Returns:
        List[Dataset] -- A list of datasets, where a dataset was created for each suffix
                         e.g. a dataset with data from the keys ('X_src', 'Y_src') and from ('X_tgt', 'Y_tgt')

    """
    p = Path(path)
    if p.is_dir():
        file_paths = list(p.glob("[!._]*"))
        assert len(file_paths) == 1
        p = file_paths[0]

    assert p.suffix == ".mat"

    mat = loadmat(p)

    keys = [k for k in mat.keys() if not str(k).startswith("_")]

    # group keys according to common suffixes (assuming '_' was used to divide name parts)
    suffixes = [k.split("_")[-1] for k in keys]
    keys_by_suffix = {
        suf: [str(sk[1]) for sk in zip(suffixes, keys) if sk[0] == suf]
        for suf in set(suffixes)
    }

    LABEL_INDICATORS = ["y", "lbl", "lbls", "label", "labels"]

    # create a dataset for each suffix

    datasets: List[Dataset] = []
    for suffix, keys in keys_by_suffix.items():
        label_keys = list(
            filter(
                lambda k: any(
                    [
                        label_indicator in k.lower()
                        for label_indicator in LABEL_INDICATORS
                    ]
                ),
                keys,
            )
        )
        label_key = label_keys[0] if len(label_keys) > 0 else None
        data_keys = [k for k in keys if k != label_key]

        datasets.append(
            _dataset_from_np_dict(
                data=mat,
                data_keys=data_keys,
                label_key=label_key,
                name=suffix,
                identifier=str(Path(path) / suffix),
            )
        )

    return sorted(datasets, key=lambda d: d.name)


def from_csv(
    path: AnyPath,
    load_func=lambda path, data: data,
    predicate_func=lambda path: True,
    data_format="tuple",
    **kwargs,
):
    """Load data stored as comma-separated values (CSV).
    The csv-data can be stored as either a single file or in several smaller
    files stored in a tree structure.

    Information from the path of the individual CSV files can be incorporated
    through a user-defined function.
    The function is invoked with the path to the CSV files and its contents,
    and must return a new sample.

    Additionally, specific files may be skipped by supplying a predicate function.
    This function is invoked with the path of each file.

    Arguments:
        path {AnyPath} -- path to either a single csv file or a directory containing CSV files.

    Keyword Arguments:
        load_func {Callable} -- optional user-defined function called with the path and contents of each CSV file. (default: {None})
        predicate_func {Callable} -- optional predicate function used to define files to be skipped. (default: {None})
        data_format {bool} -- defines how the data read from the csv is formatted. Possible options are {"tuple", "dataframe"}
        kwargs {Any} -- additional arguments passed to pandas read_csv function

    Examples:

    Consider the example below:

    .. code-block::

        cars
        ├── car_1
        │   ├── load_1000.csv
        │   └── load_2000.csv
        └── car_2
            ├── load_1000.csv
            └── load_2000.csv

    """
    p = Path(path)

    formats = {"tuple", "numpy", "dataframe"}

    if data_format not in formats:
        raise ValueError(
            f"Unable to load the dataset from CSV, the specified data fromat : {data_format} is not recognized. Options are: {formats}"
        )

    # read csv using pandas
    # if specified the dataframe is converted to a tuple of numpy arrays.
    def read_single_csv(path):
        nonlocal kwargs
        data = _read_single_csv(path, data_format, kwargs)
        return load_func(path, data)

    if p.is_file():
        ds = from_files_list([p], read_single_csv)

    elif p.is_dir():
        ds = from_recursive_files(p, read_single_csv, predicate_func)
    else:
        raise ValueError(
            f"Unable to load the dataset from CSV, the supplied path: {p} is neither a file or directory"
        )

    return ds


def _read_single_csv(path: Path, data_format, kwargs):
    """Read the contents of the specified csv file and format the results

    Arguments:
        path {Path} -- path to a csv file
        data_format {[str]} -- string defining how the data should be formatted.

    Returns:
        [Any] -- data from the csv file in the specified format.
    """
    import pandas as pd

    data = pd.read_csv(path, **kwargs)
    # convert dataframe to
    if data_format == "tuple":
        # try to create named tuple, otherwise create plain tuple
        try:
            Row = namedtuple("Row", data.columns)
            data = Row(*data.to_numpy().T.tolist())
            data = tuple(data)
        except Exception:
            data = tuple(data.to_numpy().T.tolist())
    elif data_format == "numpy":
        data = data.to_numpy()

    return data


def from_files_list(files: Sequence[AnyPath], load_func: Callable[[AnyPath], Any]):
    """Reads a list of files using by invoking a user-defined function on each file.
    The function is invoked with the path of each file and must return a new sample.

    Arguments:
        files {[Iterable]} -- an list of files to load
        load_func {[type]} -- function invoked with the path to each file to produce a sample.

    Returns:
        Dataset -- The resulting dataset.
    """

    ids_to_file = {idx: f for idx, f in enumerate(files)}

    def get_data(i):
        nonlocal ids_to_file
        nonlocal load_func
        sample = load_func(ids_to_file[i])
        return sample

    ds = Loader(
        get_data,
        ids=list(ids_to_file.keys()),
        name="files_list",
        identifier=str(hash(frozenset(ids_to_file.items()))),
    )
    return ds


def from_recursive_files(
    root: AnyPath, load_func: Callable[[AnyPath], Any], predicate_func=lambda x: True
) -> Dataset:
    """Provides functionality to load files stored in a tree structure in a recursively in a generic manner.
    A callback function must be specified which is invoked with the path of each file.
    When called this function should return a sample corresponding to the contents of the file.
    Specific files may be skipped by supplying a predicate function.

    Arguments:
        root {AnyPath} -- Path to the root directory
        load_func {[type]} -- Function invoked with the path of each file.
        predicate_func {[type]} -- Predicate function determining

    Returns:
        Dataset -- The resulting dataset.

    Examples:

    Consider the file structure shown below:

    .. code-block::

        patients
        ├── control
        │   ├── somefile.csv
        │   ├── subject_a.txt
        │   └── subject_b.txt
        └── experimental
            ├── subject_c.txt
            └── subject_d.txt
    """
    root_dir = Path(root)

    # find all files matching predicate function
    matches = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            p = Path(root) / f
            if predicate_func(p):
                matches.append(p)

    return from_files_list(matches, load_func)
