from datasetops.dataset import Dataset
from typing import List
from datasetops.cache import Cache
from datasetops import loaders
from testing_utils import (  # type:ignore
    get_test_dataset_path,
    from_dummy_numpy_data,
    read_text,
    read_bin,
    DATASET_PATHS,
)


def test_cachable():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, _ = loaders.from_folder_dataset_group_data(path)

    assert test.cachable

    test = test.image(False, True, False)
    assert test.cachable

    test = test.transform((read_text, None, None))
    assert test.cachable

    test = test.transform((None, None, read_bin))
    assert test.cachable

    test = test.image_resize(None, (10, 10), None)
    assert test.cachable

    test1, test2 = test.split([0.3, -1])
    assert not test1.cachable
    assert not test2.cachable

    test3, test4 = test.split([0.3, -1], 2605)
    assert test3.cachable
    assert test4.cachable

    unidentified = from_dummy_numpy_data()
    assert not unidentified.cachable


def test_cache():

    import dill

    test_cache_path = get_test_dataset_path(DATASET_PATHS.CACHE_ROOT_PATH)

    def assert_cache(
        datasets: List[Dataset], cache_path: str = None, clear: bool = True
    ):
        def are_same(a, b):
            return dill.dumps(a) == dill.dumps(b)

        if clear:
            Cache.clear(cache_path)

        for dataset in datasets:

            all_data = []

            for data in dataset:
                all_data.append(data)

            cached = dataset.cached(cache_path)
            cached_all_data = []

            assert cached.names == dataset.names

            for data in cached:
                cached_all_data.append(data)

            for a, c in zip(all_data, cached_all_data):
                assert are_same(a, c)

            cached2 = dataset.cached(cache_path)
            cached2_all_data = []

            assert cached2.names == dataset.names

            for data in cached2:
                cached2_all_data.append(data)

            for a, c2 in zip(all_data, cached2_all_data):
                assert are_same(a, c2)

            cached.close()
            cached2.close()

    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, train = loaders.from_folder_dataset_group_data(path)

    test_f = (
        test.image(False, True, False)
        .transform((read_text, None, None))
        .transform((None, None, read_bin))
        .image_resize(None, (10, 10), None)
    )

    for cache_path in [None, test_cache_path]:
        assert_cache([test_f], cache_path)
        assert_cache([train], cache_path)

        test1 = test.image(False, True, False)
        assert_cache([test1], cache_path)

        test2 = test1.image(False, True, False)
        assert_cache([test2], cache_path)

        test3 = test2.transform((read_text, None, None))
        assert_cache([test3], cache_path)

        test4 = test3.transform((None, None, read_bin))
        assert_cache([test4], cache_path)

        test5 = test4.image_resize(None, (10, 10), None)
        assert_cache([test5], cache_path)

        test5_1, test5_2 = test5.split([0.3, -1], 2605)
        assert_cache(
            [test5_1, test5_2], cache_path
        )
        assert_cache(
            [test5_2, test5_1], cache_path
        )

        test5_3, test5_4 = test5_1.split_filter(lambda x: False)
        assert_cache(
            [test5_3],
            cache_path,
        )
        assert_cache(
            [test5_4],
            cache_path,
        )

        test6 = test5_2.filter(lambda x: False)
        assert_cache(
            [test6],
            cache_path,
        )

        test7 = test5_2.filter(lambda x: True)
        assert_cache(
            [
                test7
            ],
            cache_path,
        )

        test8 = test7.sample(2, 2605)
        assert_cache(
            [
                test8
            ],
            cache_path,
        )

        test9 = test8.shuffle(2605)
        assert_cache(
            [
                test9
            ],
            cache_path,
        )

        test10 = test9.take(1)
        assert_cache(
            [
                test10
            ],
            cache_path,
        )

        test11 = test10.repeat(3)
        assert_cache(
            [
                test11
            ],
            cache_path,
        )

        test12 = test11.reorder("image_2", "calib", "velodyne")
        assert_cache(
            [
                test12
            ],
            cache_path,
        )

        test13 = test12.cartesian_product(test12)
        assert_cache(
            [
                test13
            ],
            cache_path,
        )

        test14 = test13.concat(test13)
        assert_cache(
            [
                test14,
                test13,
                test12,
                test11,
                test10,
                test9,
                test8,
                test7,
                test6,
                test5_4,
                test5_1,
                test5_2,
                test5,
                test4,
                test3,
                test2,
                test1,
                test,
            ],
            cache_path,
        )

        Cache.clear(cache_path)

        test1 = test.image(False, True, False).cached(cache_path)
        assert_cache([test1], cache_path, False)

        test2 = test1.image(False, True, False).cached(cache_path)
        assert_cache([test2], cache_path, False)

        test3 = test2.transform((read_text, None, None)).cached(cache_path)
        assert_cache([test3], cache_path, False)

        test4 = test3.transform((None, None, read_bin)).cached(cache_path)
        assert_cache([test4], cache_path, False)

        test5 = test4.image_resize(None, (10, 10), None).cached(cache_path)
        assert_cache([test5], cache_path, False)

        test5_1, test5_2 = test5.split([0.3, -1], 2605)

        test5_1 = test5_1.cached(cache_path)
        test5_2 = test5_2.cached(cache_path)

        assert_cache(
            [test5_1, test5_2],
            cache_path,
            False,
        )

        test5_3, test5_4 = test5_1.split_filter(lambda x: False)

        test5_3 = test5_1.cached(cache_path)
        test5_4 = test5_2.cached(cache_path)

        assert_cache(
            [test5_3],
            cache_path,
            False,
        )
        assert_cache(
            [test5_4],
            cache_path,
            False,
        )

        test6 = test5_2.filter(lambda x: False).cached(cache_path)
        assert_cache(
            [test6],
            cache_path,
            False,
        )

        test7 = test5_2.filter(lambda x: True).cached(cache_path)
        assert_cache(
            [
                test7
            ],
            cache_path,
            False,
        )

        test8 = test7.sample(2, 2605).cached(cache_path)
        assert_cache(
            [
                test8
            ],
            cache_path,
            False,
        )

        test9 = test8.shuffle(2605).cached(cache_path)
        assert_cache(
            [
                test9
            ],
            cache_path,
            False,
        )

        test10 = test9.take(1).cached(cache_path)
        assert_cache(
            [
                test10
            ],
            cache_path,
            False,
        )

        test11 = test10.repeat(3).cached(cache_path)
        assert_cache(
            [
                test11
            ],
            cache_path,
            False,
        )

        test12 = test11.reorder("image_2", "calib", "velodyne").cached(
            cache_path
        )
        assert_cache(
            [
                test12
            ],
            cache_path,
            False,
        )

        test13 = test12.cartesian_product(test12).cached(cache_path)
        assert_cache(
            [
                test13
            ],
            cache_path,
            False,
        )

        test14 = test13.concat(test13).cached(cache_path)
        assert_cache(
            [
                test14,
                test13,
                test12,
                test11,
                test10,
                test9,
                test8,
                test7,
                test6,
                test5_4,
                test5_1,
                test5_2,
                test5,
                test4,
                test3,
                test2,
                test1,
                test,
            ],
            cache_path,
            False,
        )

        for dataset in [
            test14,
            test13,
            test12,
            test11,
            test10,
            test9,
            test8,
            test7,
            test6,
            test5_4,
            test5_1,
            test5_2,
            test5,
            test4,
            test3,
            test2,
            test1,
            test,
        ]:
            dataset.close()

        Cache.clear(cache_path)
