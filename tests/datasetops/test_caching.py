from datasetops.dataset import zipped
from datasetops import loaders
import numpy as np
from testing_utils import get_test_dataset_path, DATASET_PATHS  # type:ignore


def test_cachable():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, _ = loaders.from_folder_dataset_group_data(path)

    def read_text(path):
        with open(path, "r") as file:
            return file.read()

    def read_bin(path):
        with open(path, "rb") as file:
            return np.array(file.read())

    assert(test.cachable == True)

    test = test.image(False, True, False)
    assert(test.cachable == True)

    test = test.transform((read_text, None, None))
    assert(test.cachable == True)

    test = test.transform((None, None, read_bin))
    assert(test.cachable == True)

    test = test.image_resize(None, (10, 10), None)
    assert(test.cachable == True)

    test1, test2 = test.split([0.3, -1])
    assert(test1.cachable == False)
    assert(test2.cachable == False)
    
    test3, test4 = test.split([0.3, -1], 2605)
    assert(test3.cachable == True)
    assert(test4.cachable == True)
