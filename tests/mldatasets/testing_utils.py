from pathlib import Path
from enum import Enum

class TestDatasets(Enum):
    FOLDER_DATA = 'folder_dataset_class_data/amazon/back_pack'
    FOLDER_CLASS_DATA = 'folder_dataset_class_data/amazon'
    FOLDER_DATASET_CLASS_DATA = 'folder_dataset_class_data'
    MAT_SINGLE_WITH_MULTI_DATA = 'mat_single_with_multi_data'


def get_test_dataset_path(dataset_enum: TestDatasets) -> str:
    return str((Path(__file__).parent.parent / 'recourses' / dataset_enum.value).absolute())