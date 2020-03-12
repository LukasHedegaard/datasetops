from pathlib import Path
from typing import NamedTuple

class DatasetPaths(NamedTuple):
    FOLDER_DATA: str = 'folder_dataset_class_data/amazon/back_pack'
    FOLDER_CLASS_DATA: str = 'folder_dataset_class_data/amazon'
    FOLDER_DATASET_CLASS_DATA: str = 'folder_dataset_class_data'
    MAT_SINGLE_WITH_MULTI_DATA: str = 'mat_single_with_multi_data'

DATASET_PATHS = DatasetPaths()

def get_test_dataset_path(dataset_path: str) -> str:
    return str((Path(__file__).parent.parent / 'recourses' / dataset_path).absolute())