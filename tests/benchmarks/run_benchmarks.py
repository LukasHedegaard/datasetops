from typing import List, Optional, Callable
from datasetops.cache import Cache
from datasetops import loaders
from ..datasetops_tests.testing_utils import (
    read_lines,
    read_bin,
)
import numpy as np
from timeit import timeit
from .benchmark_kitti_common import (
    reduce_point_cloud,
    get_label_anno,
    parse_calib,
)
import random


class Benchmark:
    @staticmethod
    def run(tests: List[Callable], repeats=2):

        resuts = []

        for test in tests:
            seconds = timeit(test, number=repeats)
            resuts.append(seconds)

        return resuts

    @staticmethod
    def is_better(tests: List[Callable], repeats=2) -> bool:
        """Returns if first test is better than other tests

        Arguments:
            tests {List[Callable]} -- Array of tests

        Keyword Arguments:
            repeats {int} -- How much times to repeat each test (default: {2})

        Returns:
            [bool] -- Is first test is better than other tests
        """

        results = Benchmark.run(tests, repeats)

        return any(map(lambda val: results[0] < val, results[1:]))


def benchmark_kitti(kitti_full_path: str):

    path = kitti_full_path
    test, train = loaders.from_folder_dataset_group_data(path)

    def train_reduce_point_cloud(data):

        velodyne = data[3]

        w, h = data[1].size
        image_shape = (h, w)

        calib = data[0]
        velodyne = reduce_point_cloud(
            velodyne,
            calib["R0_rect"],
            calib["P"][2],
            calib["Tr_velo_to_cam"],
            image_shape,
        )

        return (data[0], data[1], data[2], velodyne)

    def create_train_dataset(
        train,
        is_cached: bool,
        take: Optional[int] = None,
        keep_loaded_items: bool = False,
    ):
        train = (
            train.image(False, True, False)
            .transform(calib=read_lines)
            .transform(calib=parse_calib)
            .transform(velodyne=read_bin)
            .transform(velodyne=lambda d: np.reshape(d, (-1, 4)))
            .transform(label_2=read_lines)
            .transform(label_2=get_label_anno)
            .transform(train_reduce_point_cloud)
        )

        if take is not None:
            train = train.take(take)

        if is_cached:
            train = train.cached(
                keep_loaded_items=keep_loaded_items, display_progress=True
            )

        return train

    def create_traverse_dataset(dataset, randomize=False):
        def traverse():

            ids = list(range(len(dataset)))

            if randomize:
                random.shuffle(ids)

            for i, id in enumerate(ids):
                dataset[id]
                print("[" + str(i + 1) + "/" + str(len(dataset)) + "]", end="\r")

            print()

        return traverse

    def run_test(
        train,
        take: Optional[int],
        traverse_repeats: int,
        keep_loaded_items: bool,
        randomize: bool,
    ):

        Cache.clear()

        (caching_time,) = Benchmark.run(
            [lambda: create_train_dataset(train, True, take)]
        )

        cached_dataset = create_train_dataset(train, True, take, keep_loaded_items)

        results = Benchmark.run(
            [
                create_traverse_dataset(
                    create_train_dataset(train, False, take, keep_loaded_items),
                    randomize,
                ),
                create_traverse_dataset(cached_dataset, randomize),
            ],
            traverse_repeats,
        )

        cached_dataset.close()

        return {
            "amount": str(take) if take is not None else "all",
            "traverse_repeats": traverse_repeats,
            "results": results,
            "caching_time": caching_time,
            "storage": "memory" if keep_loaded_items else "file",
            "access": "random" if randomize else "forward",
        }

    results = [
        run_test(train, 20, 50, False, False),
        run_test(train, 20, 50, True, False),
        run_test(train, 20, 50, True, True),
        run_test(train, 60, 100, False, False),
        run_test(train, 60, 100, True, False),
        run_test(train, 60, 100, True, True),
        run_test(train, 60, 200, False, False),
        run_test(train, 60, 200, True, False),
        run_test(train, 60, 200, True, True),
        run_test(train, 60, 600, False, False),
        run_test(train, 60, 600, True, False),
        run_test(train, 60, 600, True, True),
        run_test(train, None, 2, False, False),
    ]

    row_format = "{:>18}" * 8

    print()
    print(
        row_format.format(
            "Dataset Size",
            "Repeats",
            "Storage",
            "Access",
            "Time Raw (s)",
            "Time Cached (s)",
            "Caching Time (s)",
            "Difference",
        )
    )

    for result in results:
        print(
            row_format.format(
                result["amount"],
                result["traverse_repeats"],
                result["storage"],
                result["access"],
                int(result["results"][0] * 100) / 100,
                int(result["results"][1] * 100) / 100,
                int(result["caching_time"] * 100) / 100,
                "x" + str(int(result["results"][0] / result["results"][1] * 100) / 100),
            )
        )


if __name__ == "__main__":
    full_kitti_dataset_path = "Add/your/path/to/the/full/KITTI/dataset/here"
    benchmark_kitti(full_kitti_dataset_path)
