from datasetops import loaders
import numpy as np
from testing_utils import get_test_dataset_path, DATASET_PATHS  # type:ignore


def test_image_to_tensorflow():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, train = loaders.from_folder_dataset_group_data(path)

    def read_text(path):
        with open(path, "r") as file:
            return file.read()

    def read_bin(path):
        with open(path, "rb") as file:
            return np.array(file.read())

    test1, test2 = test \
        .image(False, True, False) \
        .transform((read_text, None, None)) \
        .transform((None, None, read_bin)) \
        .split([0.3, -1], 2605)

    tfds = test1.to_tensorflow()
    tfds2 = test1.image_resize(None, (10, 10), None).to_tensorflow()

    for data in tfds:
        pass

    for data in tfds2:
        pass

    assert(True)
