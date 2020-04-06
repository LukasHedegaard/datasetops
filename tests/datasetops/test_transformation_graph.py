from datasetops.dataset import Dataset
from datasetops import loaders
import numpy as np
from testing_utils import (  # type:ignore
    get_test_dataset_path,
    DATASET_PATHS,
    read_text,
    read_bin,
)
import pytest
from datasetops.dataset import zipped


def test_roots_kitti():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, train = loaders.from_folder_dataset_group_data(path)

    test1, test2 = (
        test.image(False, True, False)
        .transform((read_text, None, None))
        .transform((None, None, read_bin))
        .image_resize(None, (10, 10), None)
        .split([0.3, -1], 2605)
    )

    assert len(test.get_transformation_graph().roots) == 3
    assert len(train.get_transformation_graph().roots) == 4


@pytest.mark.slow
def test_roots_tfds():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, _ = loaders.from_folder_dataset_group_data(path)

    tfds = test.to_tensorflow()

    ds = loaders.from_tensorflow(tfds, "tfds")

    assert ds.get_transformation_graph().roots == ["tfds"]

    def same(x):
        return x

    ds = ds.transform((same, same, same))

    assert ds.get_transformation_graph().roots == ["tfds"]


def test_common_nodes_equality():
    """Tests if datasets that were made from one dataset share same origin nodes
    (across one transformation graph) and not their duplicates
    (applicable for split/split_filter operations)

    """
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, train = loaders.from_folder_dataset_group_data(path)

    test1, test2 = (
        test.image(False, True, False)
        .transform((read_text, None, None))
        .transform((None, None, read_bin))
        .image_resize(None, (10, 10), None)
        .split([0.3, -1], 2605)
    )

    test = zipped(test1, test2)

    test.get_transformation_graph().display()

    graph = test.get_transformation_graph().graph
    test1_node = graph["edge"]["parent"]["edge"][0]["parent"]  # split[0]
    test2_node = graph["edge"]["parent"]["edge"][1]["parent"]  # split[1]

    test1_node_parent = test1_node["edge"]["parent"]
    test2_node_parent = test2_node["edge"]["parent"]

    assert test1_node_parent is test2_node_parent


def test_operation_origins():
    """Tests if all operations create relevant origin
    """
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, train = loaders.from_folder_dataset_group_data(path)

    test = test.image(False, True, False)
    assert test._get_origin()["operation"]["name"] == "transform"
    assert "function" in test._get_origin()["operation"]["parameters"]

    test = test.transform((read_text, None, None))
    assert test._get_origin()["operation"]["name"] == "transform"
    assert "function" in test._get_origin()["operation"]["parameters"]

    test = test.transform((None, None, read_bin))
    assert test._get_origin()["operation"]["name"] == "transform"
    assert "function" in test._get_origin()["operation"]["parameters"]

    test = test.image_resize(None, (10, 10), None)
    assert test._get_origin()["operation"]["name"] == "transform"
    assert "function" in test._get_origin()["operation"]["parameters"]

    test1, test2 = test.split([0.3, -1], 2605)
    assert test1._get_origin()["operation"]["name"] == "split"
    assert "fractions" in test1._get_origin()["operation"]["parameters"]
    assert "seed" in test1._get_origin()["operation"]["parameters"]
    assert "index" in test1._get_origin()["operation"]["parameters"]
    assert test2._get_origin()["operation"]["name"] == "split"
    assert "fractions" in test1._get_origin()["operation"]["parameters"]
    assert "seed" in test1._get_origin()["operation"]["parameters"]
    assert "index" in test1._get_origin()["operation"]["parameters"]

    test3, test4 = test1.split_filter(lambda x: False)
    assert len(test4) == len(test1)
    assert len(test3) == 0
    assert test3._get_origin()["operation"]["name"] == "split_filter"
    assert "predicates" in test3._get_origin()["operation"]["parameters"]
    assert "kwpredicates" in test3._get_origin()["operation"]["parameters"]
    assert "index" in test3._get_origin()["operation"]["parameters"]
    assert test4._get_origin()["operation"]["name"] == "split_filter"
    assert "predicates" in test4._get_origin()["operation"]["parameters"]
    assert "kwpredicates" in test4._get_origin()["operation"]["parameters"]
    assert "index" in test4._get_origin()["operation"]["parameters"]

    test5 = test2.filter(lambda x: False)
    assert len(test5) == 0
    assert test5._get_origin()["operation"]["name"] == "filter"
    assert "predicates" in test5._get_origin()["operation"]["parameters"]
    assert "kwpredicates" in test5._get_origin()["operation"]["parameters"]

    test6 = test2.filter(lambda x: True)
    assert len(test6) == len(test2)
    assert test6._get_origin()["operation"]["name"] == "filter"
    assert "predicates" in test6._get_origin()["operation"]["parameters"]
    assert "kwpredicates" in test6._get_origin()["operation"]["parameters"]

    test6 = test6.sample(2, 2605)
    assert len(test6) == 2
    assert test6._get_origin()["operation"]["name"] == "sample"
    assert "num" in test6._get_origin()["operation"]["parameters"]
    assert "seed" in test6._get_origin()["operation"]["parameters"]

    test6 = test6.shuffle(2605)
    assert len(test6) == 2
    assert test6._get_origin()["operation"]["name"] == "shuffle"
    assert "seed" in test6._get_origin()["operation"]["parameters"]

    test6 = test6.take(1)
    assert len(test6) == 1
    assert test6._get_origin()["operation"]["name"] == "take"
    assert "num" in test6._get_origin()["operation"]["parameters"]

    test6 = test6.repeat(3)
    assert len(test6) == 3
    assert test6._get_origin()["operation"]["name"] == "repeat"
    assert "times" in test6._get_origin()["operation"]["parameters"]
    assert "mode" in test6._get_origin()["operation"]["parameters"]

    test7 = test6.reorder("image_2", "calib", "velodyne_reduced")
    assert len(test7) == 3
    assert test7[0][0] == test6[0][1]
    assert test7._get_origin()["operation"]["name"] == "reorder"
    assert "keys" in test7._get_origin()["operation"]["parameters"]

    test7 = test7.cartesian_product(test7)
    assert len(test7[0]) == 6
    assert len(test7) == 9
    assert test7._get_origin()["operation"]["name"] == "copy"

    test7 = test7.concat(test7)
    assert len(test7[0]) == 6
    assert len(test7) == 18
    assert test7._get_origin()["operation"]["name"] == "copy"


def test_serialization_same():
    def assert_serialized_is_same(dataset: Dataset):
        graph = dataset.get_transformation_graph()
        assert graph.is_same_as_serialized(graph.serialize())

    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, train = loaders.from_folder_dataset_group_data(path)
    assert_serialized_is_same(test)
    assert_serialized_is_same(train)

    test = test.image(False, True, False)
    assert_serialized_is_same(test)

    test = test.image(False, True, False)
    assert_serialized_is_same(test)

    test = test.transform((read_text, None, None))
    assert_serialized_is_same(test)

    test = test.transform((None, None, read_bin))
    assert_serialized_is_same(test)

    test = test.image_resize(None, (10, 10), None)
    assert_serialized_is_same(test)

    test1, test2 = test.split([0.3, -1], 2605)
    assert_serialized_is_same(test1)
    assert_serialized_is_same(test2)

    test3, test4 = test1.split_filter(lambda x: False)
    assert_serialized_is_same(test3)
    assert_serialized_is_same(test4)

    test5 = test2.filter(lambda x: False)
    assert_serialized_is_same(test5)

    test6 = test2.filter(lambda x: True)
    assert_serialized_is_same(test6)

    test6 = test6.sample(2, 2605)
    assert_serialized_is_same(test6)

    test6 = test6.shuffle(2605)
    assert_serialized_is_same(test6)

    test6 = test6.take(1)
    assert_serialized_is_same(test6)

    test6 = test6.repeat(3)
    assert_serialized_is_same(test6)

    test7 = test6.reorder("image_2", "calib", "velodyne_reduced")
    assert_serialized_is_same(test7)

    test7 = test7.cartesian_product(test7)
    assert_serialized_is_same(test7)

    test7 = test7.concat(test7)
    assert_serialized_is_same(test7)


def test_serialization_not_same():
    def assert_serialized_is_not_same(target, *datasets: Dataset):
        graph = target.get_transformation_graph()

        for dataset in datasets:
            assert not graph.is_same_as_serialized(
                dataset.get_transformation_graph().serialize()
            )

    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, train = loaders.from_folder_dataset_group_data(path)
    assert_serialized_is_not_same(test, train)
    assert_serialized_is_not_same(train, test)

    test1 = test.image(False, True, False)
    assert_serialized_is_not_same(test1, test)

    test2 = test1.image(False, True, False)
    assert_serialized_is_not_same(test2, test1, test)

    test3 = test2.transform((read_text, None, None))
    assert_serialized_is_not_same(test3, test2, test1, test)

    test4 = test3.transform((None, None, read_bin))
    assert_serialized_is_not_same(test4, test3, test2, test1, test)

    test5 = test4.image_resize(None, (10, 10), None)
    assert_serialized_is_not_same(test5, test4, test3, test2, test1, test)

    test5_1, test5_2 = test5.split([0.3, -1], 2605)
    assert_serialized_is_not_same(
        test5_1, test5_2, test5, test4, test3, test2, test1, test
    )
    assert_serialized_is_not_same(
        test5_2, test5_1, test5, test4, test3, test2, test1, test
    )

    test5_3, test5_4 = test5_1.split_filter(lambda x: False)
    assert_serialized_is_not_same(
        test5_3, test5_1, test5_2, test5, test4, test3, test2, test1, test
    )
    assert_serialized_is_not_same(
        test5_4, test5_1, test5_2, test5, test4, test3, test2, test1, test
    )

    test6 = test5_2.filter(lambda x: False)
    assert_serialized_is_not_same(
        test6, test5_4, test5_1, test5_2, test5, test4, test3, test2, test1, test
    )

    test7 = test5_2.filter(lambda x: True)
    assert_serialized_is_not_same(
        test7, test6, test5_4, test5_1, test5_2, test5, test4, test3, test2, test1, test
    )

    test8 = test7.sample(2, 2605)
    assert_serialized_is_not_same(
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
    )

    test9 = test8.shuffle(2605)
    assert_serialized_is_not_same(
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
    )

    test10 = test9.take(1)
    assert_serialized_is_not_same(
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
    )

    test11 = test10.repeat(3)
    assert_serialized_is_not_same(
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
    )

    test12 = test11.reorder("image_2", "calib", "velodyne_reduced")
    assert_serialized_is_not_same(
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
    )

    test13 = test12.cartesian_product(test12)
    assert_serialized_is_not_same(
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
    )

    test14 = test13.concat(test13)
    assert_serialized_is_not_same(
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
    )

