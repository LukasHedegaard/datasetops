from datasetops.dataset import StreamDataset
from typing import List
import dill
from datasetops.cache import Cache
from datasetops import loaders
from testing_utils import (  # type:ignore
    get_test_dataset_path,
    from_dummy_numpy_data,
    read_text,
    read_bin,
    DATASET_PATHS,
)
from pathlib import Path
import random


def are_same(a, b):
    return dill.dumps(a) == dill.dumps(b)


def test_read_from_file():
    path = get_test_dataset_path(DATASET_PATHS.KITTI_DATASET)
    test, train = loaders.from_folder_dataset_group_data(path)

    original = (
        train.image(False, True, False)
        .transform(calib=read_text)
        .transform(velodyne=read_bin)
    )

    cache_path = get_test_dataset_path(DATASET_PATHS.CACHE_ROOT_PATH)
    pkl_path = str(Path(cache_path) / "cache_1.pkl")
    Cache.clear(cache_path)

    original.cached(cache_path)

    file1 = open(pkl_path, "rb")
    ds1 = StreamDataset(file1, pkl_path, False)

    file2 = open(pkl_path, "rb")
    ds2 = StreamDataset(file2, pkl_path, True)

    def compare_dataset(original, dataset, repeats=4, randomize=False):
        for i in range(repeats):
            assert len(dataset) == len(original)

            ids = list(range(len(original)))

            if randomize:
                random.shuffle(ids)

            for i in ids:
                original_data = original[i]
                data = dataset[i]
                assert are_same(original_data, data)

    compare_dataset(original, ds1)
    compare_dataset(original, ds2)
    compare_dataset(original, ds2, randomize=True)

    try:
        compare_dataset(original, ds1, repeats=1000, randomize=True)
        assert False
    except Exception:
        assert True

    ds1.close()
    ds2.close()
