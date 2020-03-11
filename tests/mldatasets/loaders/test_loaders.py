from pathlib import Path
import mldatasets.loaders as loaders
import random
from pathlib import Path
from enum import Enum

# utils ##########################

class TestDatasets(Enum):
    FOLDER_DATA = 'folder_dataset_class_data/amazon/back_pack'
    FOLDER_CLASS_DATA = 'folder_dataset_class_data/amazon'
    FOLDER_DATASET_CLASS_DATA = 'folder_dataset_class_data'
    MAT_SINGLE_WITH_MULTI_DATA = 'mat_single_with_multi_data'


def get_test_dataset_path(dataset_enum: TestDatasets) -> str:
    return str((Path(__file__).parent.parent / 'recourses' / dataset_enum.value).absolute())


# tests ##########################

def test_folder_data():
    path = get_test_dataset_path(TestDatasets.FOLDER_DATA)

    expected_items = [str(Path(path)/'frame_000{}.jpg'.format(i)) for i in range(1,7)]

    ds = loaders.load_folder_data(path)
    found_items = [i for i in ds]

    assert(set(expected_items) == set(found_items))


def test_folder_class_data():
    path = get_test_dataset_path(TestDatasets.FOLDER_CLASS_DATA)

    expected_items = [str(p) for p in Path(path).glob('*/*.jpg')]

    ds = loaders.load_folder_class_data(path)
    found_items = [i for i in ds]

    assert(set(expected_items) == set(found_items))



def test_folder_dataset_class_data():
    path = get_test_dataset_path(TestDatasets.FOLDER_DATASET_CLASS_DATA)
    sets = Path(path).glob('[!._]*')

    sets_of_expected_items = [
        set([str(p) for p in Path(s).glob('*/*.jpg')])
        for s in sets
    ]

    datasets = loaders.load_folder_dataset_class_data(path)
    sets_of_found_items = [
        set([i for i in ds])
        for ds in datasets
    ]

    for expected_items_set in sets_of_expected_items:
        assert(any([expected_items_set == found_items for found_items in sets_of_found_items]))


def test_mat_single_with_multi_data():
    path = get_test_dataset_path(TestDatasets.MAT_SINGLE_WITH_MULTI_DATA)

    datasets = loaders.load_mat_single_mult_data(path)

    for ds in datasets:
        # check dataset sizes and names
        if ds.name == 'src':
            assert(len(ds) == 2000)
        elif ds.name == 'tar':
            assert(len(ds) == 1800)
        else:
            assert(False)

        # randomly check some samples for their dimension
        ids = random.sample(range(len(ds)), 42)
        class_names = ds.class_names()
        for i in ids:
            data, label = ds[i]

            assert(data.shape == (256,))
            assert(str(int(label)) in class_names)

