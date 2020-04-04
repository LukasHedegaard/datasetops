from datasetops.dataset import zipped
from datasetops import loaders
import numpy as np
from testing_utils import get_test_dataset_path, \
    from_dummy_numpy_data, DATASET_PATHS  # type:ignore


def test_cachable():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, _ = loaders.from_folder_dataset_group_data(path)

    def read_text(path):
        with open(path, "r") as file:
            return file.read()

    def read_bin(path):
        with open(path, "rb") as file:
            return np.array(file.read())

    assert(test.cachable)

    test = test.image(False, True, False)
    assert(test.cachable)

    test = test.transform((read_text, None, None))
    assert(test.cachable)

    test = test.transform((None, None, read_bin))
    assert(test.cachable)

    test = test.image_resize(None, (10, 10), None)
    assert(test.cachable)

    test1, test2 = test.split([0.3, -1])
    assert(not test1.cachable)
    assert(not test2.cachable)

    test3, test4 = test.split([0.3, -1], 2605)
    assert(test3.cachable)
    assert(test4.cachable)

    unidentified = from_dummy_numpy_data()
    assert(not unidentified.cachable)
