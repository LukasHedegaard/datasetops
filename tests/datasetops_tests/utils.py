from pathlib import Path
from typing import NamedTuple
from datasetops import Dataset
from datasetops.loaders import from_iterable
from datasetops.interfaces import IDataset
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
    return Path(__file__).parent.parent / "resources" / dataset_path


def from_dummy_data(num_total=11, with_label=False) -> IDataset:
    a = list(range(5))
    b = list(range(5, num_total))

    if not with_label:
        data = a + b
    else:
        data = [(i, str("a")) for i in a] + [(i, str("b")) for i in b]

    return from_iterable(data, "from_dummy_data")


DUMMY_NUMPY_DATA_SHAPE_1D = (18,)
DUMMY_NUMPY_DATA_SHAPE_2D = (6, 3)
DUMMY_NUMPY_DATA_SHAPE_3D = (2, 3, 3)


def from_dummy_numpy_data() -> IDataset:
    a_ids = list(range(5))
    b_ids = list(range(5, 11))
    ids = a_ids + b_ids
    labels = [*[1 for _ in a_ids], *[2 for _ in b_ids]]

    num_samples = len(a_ids) + len(b_ids)
    np_data = np.arange(num_samples * DUMMY_NUMPY_DATA_SHAPE_1D[0]).reshape(
        (num_samples, DUMMY_NUMPY_DATA_SHAPE_1D[0])
    )

    def get_data(idx):
        return np_data[idx], labels[idx]

    data = [get_data(i) for i in ids]

    return Dataset.from_iterable(data, "from_dummy_numpy_data")
