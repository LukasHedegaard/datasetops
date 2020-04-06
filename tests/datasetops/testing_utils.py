from pathlib import Path
from typing import NamedTuple
from datasetops.loaders import Loader
import numpy as np


class DatasetPaths(NamedTuple):
    FOLDER_DATA: str = "folder_dataset_class_data/amazon/back_pack"
    FOLDER_CLASS_DATA: str = "folder_dataset_class_data/amazon"
    FOLDER_DATASET_CLASS_DATA: str = "folder_dataset_class_data"
    MAT_SINGLE_WITH_MULTI_DATA: str = "mat_single_with_multi_data"
    KITTI_DATASET: str = "caching/kitti_dataset"  # TODO: remove caching (parent) folder?
    CACHE_ROOT_PATH: str = "caching/cache_root"  # TODO: move?
    FOLDER_GROUP_DATA: str = KITTI_DATASET + "/training"
    FOLDER_DATASET_GROUP_DATA: str = KITTI_DATASET


DATASET_PATHS = DatasetPaths()


def get_test_dataset_path(dataset_path: str) -> str:
    return str((Path(__file__).parent.parent / "resourses" / dataset_path).absolute())


def from_dummy_data(num_total=11, with_label=False) -> Loader:
    a_ids = list(range(5))
    b_ids = list(range(5, num_total))

    def get_data(i):
        return (i,)

    def get_labelled_data(i):
        nonlocal a_ids
        return i, "a" if i < len(a_ids) else "b"

    ds = Loader(get_labelled_data if with_label else get_data)
    ds.extend(a_ids)
    ds.extend(b_ids)
    return ds


DUMMY_NUMPY_DATA_SHAPE_1D = (18,)
DUMMY_NUMPY_DATA_SHAPE_2D = (6, 3)
DUMMY_NUMPY_DATA_SHAPE_3D = (2, 3, 3)


def from_dummy_numpy_data() -> Loader:
    a_ids = list(range(5))
    b_ids = list(range(5, 11))
    labels = [*[1 for _ in a_ids], *[2 for _ in b_ids]]

    num_samples = len(a_ids) + len(b_ids)
    data = np.arange(num_samples * DUMMY_NUMPY_DATA_SHAPE_1D[0]).reshape(
        (num_samples, DUMMY_NUMPY_DATA_SHAPE_1D[0])
    )
    # data = data / data.max()

    def get_data(idx):
        return data[idx], labels[idx]

    ds = Loader(get_data)
    ds.extend(a_ids)
    ds.extend(b_ids)
    return ds


def read_text(path):
    with open(path, "r") as file:
        return file.read()


def read_bin(path):
    return np.fromfile(path, dtype=np.float32, count=-1)  # type:ignore
