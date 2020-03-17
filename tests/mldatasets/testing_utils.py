from pathlib import Path
from typing import NamedTuple
from mldatasets.function_dataset import FunctionDataset
import numpy as np

class DatasetPaths(NamedTuple):
    FOLDER_DATA: str = 'folder_dataset_class_data/amazon/back_pack'
    FOLDER_CLASS_DATA: str = 'folder_dataset_class_data/amazon'
    FOLDER_DATASET_CLASS_DATA: str = 'folder_dataset_class_data'
    MAT_SINGLE_WITH_MULTI_DATA: str = 'mat_single_with_multi_data'

DATASET_PATHS = DatasetPaths()


def get_test_dataset_path(dataset_path: str) -> str:
    return str((Path(__file__).parent.parent / 'recourses' / dataset_path).absolute())


def load_dummy_data(num_total=11, with_label=False) -> FunctionDataset:
    a_ids = list(range(5))
    b_ids = list(range(5,num_total))

    def get_data(i):
        return (i,)

    def get_labelled_data(i):
        nonlocal a_ids
        return i, 'a' if i < len(a_ids) else 'b'

    ds = FunctionDataset(get_labelled_data if with_label else get_data)
    ds._extend(a_ids)
    ds._extend(b_ids)
    return ds


DUMMY_NUMPY_DATA_SHAPE_1D = (18,)
DUMMY_NUMPY_DATA_SHAPE_2D = (6,3)
DUMMY_NUMPY_DATA_SHAPE_3D = (2,3,3)

def load_dummy_numpy_data() -> FunctionDataset:
    a_ids = list(range(5))
    b_ids = list(range(5,11))
    labels = [
        *[1 for _ in a_ids],
        *[2 for _ in b_ids]
    ]

    num_samples = len(a_ids)+len(b_ids)
    data = np.arange(num_samples*DUMMY_NUMPY_DATA_SHAPE_1D[0]).reshape((num_samples, DUMMY_NUMPY_DATA_SHAPE_1D[0]))
    # data = data / data.max()

    def get_data(idx):
        return data[idx], labels[idx]

    ds = FunctionDataset(get_data)
    ds._extend(a_ids)
    ds._extend(b_ids)
    return ds