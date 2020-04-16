from pathlib import Path
from typing import NamedTuple
from datasetops.loaders import Loader
import numpy as np

RESOURCES_PATH = Path(__file__).parent.parent / "resources"


class DatasetPaths(NamedTuple):
    FOLDER_DATA: str = "folder_dataset_class_data/amazon/back_pack"
    FOLDER_CLASS_DATA: str = "folder_dataset_class_data/amazon"
    FOLDER_DATASET_CLASS_DATA: str = "folder_dataset_class_data"
    MAT_SINGLE_WITH_MULTI_DATA: str = "mat_single_with_multi_data"
    KITTI_DATASET: str = "kitti_dataset"
    CACHE_ROOT_PATH: str = "caching/cache_root"
    FOLDER_GROUP_DATA: str = KITTI_DATASET + "/training"
    FOLDER_DATASET_GROUP_DATA: str = KITTI_DATASET
    PATIENTS: str = "patients"
    CSV: str = "csv"


DATASET_PATHS = DatasetPaths()


def get_test_dataset_path(dataset_path: str) -> Path:
    """Returns the path to the dataset relative to the test-resources folder.

    Arguments:
        dataset_path {str} -- path to the dataset defined relative to the test-resource folder

    Returns:
        Path -- path to the dataset
    """
    return Path(__file__).parent.parent / "resources" / dataset_path


def from_dummy_data(num_total=11, with_label=False) -> Loader:
    a_ids = list(range(5))
    b_ids = list(range(5, num_total))
    ids = a_ids + b_ids

    def get_data(i):
        return (i,)

    def get_labelled_data(i):
        nonlocal a_ids
        return i, "a" if i < len(a_ids) else "b"

    ds = Loader(get_labelled_data if with_label else get_data, ids=ids)
    return ds


DUMMY_NUMPY_DATA_SHAPE_1D = (18,)
DUMMY_NUMPY_DATA_SHAPE_2D = (6, 3)
DUMMY_NUMPY_DATA_SHAPE_3D = (2, 3, 3)


def from_dummy_numpy_data() -> Loader:
    a_ids = list(range(5))
    b_ids = list(range(5, 11))
    ids = a_ids + b_ids
    labels = [*[1 for _ in a_ids], *[2 for _ in b_ids]]

    num_samples = len(a_ids) + len(b_ids)
    data = np.arange(num_samples * DUMMY_NUMPY_DATA_SHAPE_1D[0]).reshape(
        (num_samples, DUMMY_NUMPY_DATA_SHAPE_1D[0])
    )
    # data = data / data.max()

    def get_data(idx):
        return data[idx], labels[idx]

    ds = Loader(get_data, ids)
    return ds


def read_text(path):
    with open(path, "r") as file:
        return file.read()


def read_lines(path):
    with open(path, "r") as file:
        return file.readlines()


def read_bin(path):
    return np.fromfile(path, dtype=np.float32, count=-1)  # type:ignore


def multi_shape_dataset(SHAPE_1D=(2,), SHAPE_3D=(5, 4, 3)):
    data = [
        # (string, 1D, 3D, scalar)
        ("0", 0 * np.ones(SHAPE_1D), 0 * np.ones(SHAPE_3D), 0),
        ("1", 1 * np.ones(SHAPE_1D), 1 * np.ones(SHAPE_3D), 1),
        ("2", 2 * np.ones(SHAPE_1D), 2 * np.ones(SHAPE_3D), 2),
    ]

    def get_data(idx):
        return data[idx]

    ds = Loader(get_data, range(len(data)))
    return ds
