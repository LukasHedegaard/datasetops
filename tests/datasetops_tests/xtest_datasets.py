import pytest
import numpy as np
from PIL import Image
from typing import Sequence, List, Tuple

from datasetops.xloaders import from_csv, from_iterable
from datasetops.xdataset import (
    image_resize,
    reshape,
    allow_unique,
    one_hot,
    categorical,
    categorical_template,
    _DEFAULT_SHAPE,
)
import datasetops.xloaders as loaders
from .xtesting_utils import (
    get_test_dataset_path,
    from_dummy_data,
    from_dummy_numpy_data,
    DATASET_PATHS,
    DUMMY_NUMPY_DATA_SHAPE_1D,
    DUMMY_NUMPY_DATA_SHAPE_2D,
    DUMMY_NUMPY_DATA_SHAPE_3D,
)


def test_generator():
    ds = from_dummy_data()
    gen = ds.generator()

    for d in list(ds):
        assert d == next(gen)


def test_shuffle():

    # empty set
    ds = loaders.Loader(lambda i: i, [])
    ds.shuffle()
    assert len(ds) == 0

    ds = from_dummy_data()

    # no seed
    ds_shuffled = ds.shuffle()
    assert set(ds) == set(ds_shuffled)  # same data
    assert list(ds) != list(ds_shuffled)  # different sequence

    ds_shuffled = ds.shuffle(seed=42)
    assert set(ds) == set(ds_shuffled)  # same data
    assert list(ds) != list(ds_shuffled)  # different sequence


def test_sample():
    seed = 42
    ds = from_dummy_data()
    ds_sampled = ds.sample(5, seed)
    found_items = [i for i in ds_sampled]

    # check list uniqueness
    assert len(found_items) == len(set(found_items))

    # check items
    expected_items = [(i,) for i in [10, 1, 0, 4, 9]]
    assert set(expected_items) == set(found_items)

    # check that different seeds yield different results
    ds_sampled2 = ds.sample(5, seed + 1)
    found_items2 = [i for i in ds_sampled2]
    assert set(found_items2) != set(found_items)


def test_filter():
    num_total = 10
    ds = from_dummy_data(num_total=num_total, with_label=True).named("data", "label")

    # expected items
    a: List[Tuple[int, str]] = [(x, "a") for x in list(range(5))]
    b: List[Tuple[int, str]] = [(x, "b") for x in list(range(5, num_total))]
    even_a = [x for x in a if x[0] % 2 == 0]
    even_b = [x for x in b if x[0] % 2 == 0]
    odd_b = [x for x in b if x[0] % 2 == 1]

    # itemwise
    ds_even = ds.filter([lambda x: x % 2 == 0])
    assert list(ds_even) == even_a + even_b

    ds_even_a = ds.filter([lambda x: x % 2 == 0, lambda x: x == "a"])
    assert list(ds_even_a) == even_a

    # by key
    ds_b = ds.filter(label=lambda x: x == "b")
    assert list(ds_b) == b

    # bulk
    ds_odd_b = ds.filter(lambda x: x[0] % 2 == 1 and x[1] == "b")
    assert list(ds_odd_b) == odd_b

    # mix
    ds_even_b = ds.filter([lambda x: x % 2 == 0], label=lambda x: x == "b")
    assert list(ds_even_b) == even_b

    # sample_classwise
    ds_classwise = ds.filter(label=allow_unique(2))
    assert list(ds_classwise) == list(a[:2] + b[:2])

    # error scenarios
    with pytest.warns(UserWarning):
        ds_same = ds.filter()  # no args
        assert list(ds) == list(ds_same)

    with pytest.raises(ValueError):
        ds.filter([None, None, None])  # too many args

    with pytest.raises(KeyError):
        ds.filter(badkey=lambda x: True)  # key doesn't exist


def test_split_filter():
    num_total = 10
    ds = from_dummy_data(num_total=num_total, with_label=True).named("data", "label")

    # expected items
    a: List[Tuple[int, str]] = [(x, "a") for x in list(range(5))]
    b: List[Tuple[int, str]] = [(x, "b") for x in list(range(5, num_total))]
    even_a = [x for x in a if x[0] % 2 == 0]
    odd_a = [x for x in a if x[0] % 2 == 1]
    even_b = [x for x in b if x[0] % 2 == 0]
    odd_b = [x for x in b if x[0] % 2 == 1]

    # itemwise
    ds_even, ds_odd = ds.split_filter([lambda x: x % 2 == 0])
    assert list(ds_even) == even_a + even_b
    assert list(ds_odd) == odd_a + odd_b

    ds_even_a, ds_not_even_a = ds.split_filter(
        [lambda x: x % 2 == 0, lambda x: x == "a"]
    )
    assert list(ds_even_a) == even_a
    assert list(ds_not_even_a) == odd_a + b

    # by key
    ds_b, ds_a = ds.split_filter(label=lambda x: x == "b")
    assert list(ds_b) == b
    assert list(ds_a) == a

    # bulk
    ds_odd_b, ds_even_b = ds.split_filter(lambda x: x[0] % 2 == 1 and x[1] == "b")
    assert list(ds_odd_b) == odd_b
    assert list(ds_even_b) == a + even_b

    # mix
    ds_even_b, ds_not_even_b = ds.split_filter(
        [lambda x: x % 2 == 0], label=lambda x: x == "b"
    )
    assert list(ds_even_b) == even_b
    assert list(ds_not_even_b) == [x for x in list(ds) if x not in even_b]

    # sample_classwise
    ds_classwise_2, ds_classwise_rest = ds.split_filter(label=allow_unique(2))
    assert list(ds_classwise_2) == list(a[:2] + b[:2])
    assert list(ds_classwise_rest) == list(a[2:] + b[2:])

    # error scenarios
    with pytest.raises(ValueError):
        _ = ds.split_filter()  # no args

    with pytest.raises(ValueError):
        ds.split_filter([None, None, None])  # too many args

    with pytest.raises(KeyError):
        ds.split_filter(badkey=lambda x: True)  # key doesn't exist


def test_split():
    seed = 42
    ds = from_dummy_data()
    ds1, ds2, ds3 = ds.split([0.6, 0.3, 0.1], seed=seed)

    # new sets are distinct
    assert set(ds1) != set(ds2)
    assert set(ds1) != set(ds3)
    assert set(ds2) != set(ds3)

    # no values are lost
    assert set(ds) == set(ds1).union(set(ds2), set(ds3))

    # repeat for wildcard
    ds1w, ds2w, ds3w = ds.split([0.6, -1, 0.1], seed=seed)

    # using wildcard produces same results
    assert set(ds1) == set(ds1w)
    assert set(ds2) == set(ds2w)
    assert set(ds3) == set(ds3w)


def test_repeat():
    ds = from_dummy_data()

    # itemwise
    ds_item = ds.repeat(3, mode="itemwise")

    assert set(ds) == set(ds_item)

    # whole
    ds_whole = ds.repeat(2, mode="whole")
    ds_whole_alt = ds.repeat(2)
    assert list(ds_whole_alt) == list(ds_whole)
    assert set(ds) == set(ds_whole)
    assert list(ds) == list(ds_whole)[: len(ds)] == list(ds_whole)[len(ds) :]


def test_take():
    ds = from_dummy_data().transform(lambda x: 10 * x)

    ds_5 = ds.take(5)
    assert list(ds)[:5] == list(ds_5)

    with pytest.raises(ValueError):
        ds.take(10000000)


def test_reorder():
    ds = from_dummy_numpy_data().named("mydata", "mylabel")

    # error scenarios
    with pytest.raises(ValueError):
        # indexes out of range
        ds_re = ds.reorder(3, 4)

    with pytest.raises(KeyError):
        # a keys doesn't exist
        ds_re = ds.reorder("badkey", "mydata")

    # working scenarios

    # using indexes
    ds_re = ds.reorder(1, 0)
    ds_re_alt = ds.reorder([1, 0])  # wrapped in list
    ds_re_alt2 = ds.reorder((1, 0))  # wrapped in tuple
    for (ldata, llbl), (rlbl, rdata), (rlbl1, rdata1), (rlbl2, rdata2) in zip(
        list(ds), list(ds_re), list(ds_re_alt), list(ds_re_alt2)
    ):
        assert np.array_equal(ldata, rdata)
        assert llbl == rlbl

        assert np.array_equal(rdata, rdata1) and np.array_equal(rdata, rdata2)
        assert rlbl == rlbl1 == rlbl2

    # same results using keys
    ds_re_key = ds.reorder("mylabel", "mydata")
    for (llbl, ldata), (rlbl, rdata) in zip(list(ds_re_key), list(ds_re)):
        assert np.array_equal(ldata, rdata)
        assert llbl == rlbl

    # same result using a mix
    ds_re_mix = ds.reorder(1, "mydata")
    for (llbl, ldata), (rlbl, rdata) in zip(list(ds_re_mix), list(ds_re)):
        assert np.array_equal(ldata, rdata)
        assert llbl == rlbl

    # we can even place the same element multiple times
    with pytest.warns(UserWarning):
        # but discard item-names (gives a warning)
        ds_re_creative = ds.reorder(0, 1, 1, 0)

    for (ldata, llbl), (rdata1, rlbl1, rlbl2, rdata2) in zip(
        list(ds), list(ds_re_creative)
    ):
        assert np.array_equal(ldata, rdata1)
        assert np.array_equal(ldata, rdata2)
        assert llbl == rlbl1 == rlbl2

    # shape updates accordingly
    assert ds_re_creative.shape == (
        DUMMY_NUMPY_DATA_SHAPE_1D,
        _DEFAULT_SHAPE,
        _DEFAULT_SHAPE,
        DUMMY_NUMPY_DATA_SHAPE_1D,
    )

    # error scenarios
    with pytest.warns(UserWarning):
        ds.named("one", "two").reorder(
            0, 1, 1
        )  # key needs to be unique, but wouldn't be


def test_getitem():
    n = 5
    itr_int = list(range(n))
    itr_tuple = [(i,) for i in itr_int]
    itr_str = [str(i) for i in itr_int]

    def do_test(itr):
        ds = from_iterable(itr)

        # index access
        for i in range(n):
            assert ds[i] == itr[i]

        with pytest.raises(IndexError):
            ds[n + 1]

        # slice access
        for i in range(n):
            for j in range(n):
                for s in range(n):
                    assert ds[i:j:n] == itr[i:j:n]

        assert ds[:] == itr[:]

    do_test(itr_int)
    do_test(itr_tuple)
    do_test(itr_str)


class TestSubsample:
    cars = from_csv(get_test_dataset_path(DATASET_PATHS.CSV + "/cars"))

    def test_subsample(self):
        def func(s):
            return (s, s)

        assert len(self.cars) == 4
        ds = TestSubsample.cars.subsample(func, 2)
        assert len(ds) == 8

        s = ds[0]
        assert len(s[0]) == 3
        assert len(s[1]) == 3

    def test_invalid_subsample_func(self):

        # incorrect number of subsamples returned
        def func1(s):
            return (s,)

        with pytest.raises(RuntimeError):
            ds = TestSubsample.cars.subsample(func1, 3)
            _ = ds[0]

        # invalid subsample returned
        def func2(s):
            return None

        with pytest.raises(RuntimeError):
            ds = TestSubsample.cars.subsample(func2, 2)
            _ = ds[0]

    def test_invalid_nsamples(self):
        def func(s):
            return s

        with pytest.raises(ValueError):
            TestSubsample.cars.subsample(func, 0)

        with pytest.raises(ValueError):
            TestSubsample.cars.subsample(func, -1)

    def test_caching(self):

        cnt = 0

        def func(s):
            nonlocal cnt
            cnt += 1
            return (s, s)

        """no caching, every time a subsample is read
        the parent sample is read as well"""
        ds = TestSubsample.cars.subsample(func, 2, cache_method=None)
        ds[0]
        ds[1]
        assert cnt == 2

        """block caching, store the subsamples of produced
        from the last read of the parent sample. In this case
        each sample produces 2 subsamples. As such reading idx 0 and 1
        should result in one read. Reading beyond this will case another read.
        Going back again the sample 0 and 1 should now be cleared
        """
        cnt = 0
        ds = TestSubsample.cars.subsample(func, 2, cache_method="block")
        ds[0]
        ds[1]
        assert cnt == 1
        ds[2]
        assert cnt == 2
        ds[0]
        assert cnt == 3
        ds[1]
        assert cnt == 3

    def test_getitem(self):
        def func(s):
            return s, s

        ds = from_iterable([1, 2]).subsample(func, 2)

        # index
        assert ds[0] == 1
        assert ds[1] == 1
        assert ds[2] == 2
        assert ds[3] == 2

        with pytest.raises(IndexError):
            ds[4]

        # slicing
        assert ds[:] == [1, 1, 2, 2]
        assert ds[0:1] == [1]
        assert ds[1:2] == [1]
        assert ds[0:3] == [1, 1, 2]
        assert ds[4:] == []


class TestSupersample:
    def test_supersample(self):

        ds = from_iterable([1, 2, 3, 4])
        assert len(ds) == 4

        ds = ds.supersample(sum, 2)
        assert len(ds) == 2
        assert ds[0] == 3
        assert ds[1] == 7

    def test_getitem(self):

        ds = from_iterable([1, 2, 3, 4]).supersample(sum, 2)

        # index
        assert ds[0] == 3
        assert ds[1] == 7
        with pytest.raises(IndexError):
            ds[2]

        assert ds[0:1] == [3]
        assert ds[0:2] == [3, 7]
        assert ds[:] == [3, 7]
        assert ds[3:] == []


# ========= Tests relating to stats =========


def test_counts():
    num_total = 11
    ds = from_dummy_data(num_total=num_total, with_label=True).named("data", "label")

    counts = ds.counts("label")  # name based
    counts_alt = ds.counts(1)  # index based
    counts_alt1 = ds.counts([1])  # index based
    counts_alt2 = ds.counts((1,))  # index based

    expected_counts = [("a", 5), ("b", num_total - 5)]
    assert counts == counts_alt == counts_alt1 == counts_alt2 == expected_counts

    counts_all = ds.counts()
    # count all if no args are given
    assert set(counts_all) == set([(x, 1) for x in ds])


def test_unique():
    ds = from_dummy_data(with_label=True).named("data", "label")

    unique_labels = ds.unique("label")
    unique_labels_alt = ds.unique(["label"])
    assert unique_labels == unique_labels_alt == ["a", "b"]

    unique_items = ds.unique()
    assert unique_items == list(ds)


def test_shape():
    def get_data(i):
        return i, i

    # no shape yet
    ds = loaders.Loader(get_data, ids=[])
    assert ds.shape == _DEFAULT_SHAPE

    # shape given
    ds = loaders.Loader(get_data, ids=[1])
    assert ds.shape == (_DEFAULT_SHAPE, _DEFAULT_SHAPE)

    # numpy data
    ds_np = from_dummy_numpy_data().reshape(DUMMY_NUMPY_DATA_SHAPE_2D)
    assert ds_np.shape == (DUMMY_NUMPY_DATA_SHAPE_2D, _DEFAULT_SHAPE)

    # changed to new size
    IMG_SIZE = (6, 6)
    ds_img = ds_np.image_resize(IMG_SIZE)
    assert ds_img.shape == (IMG_SIZE, _DEFAULT_SHAPE)

    # image with three channels
    DUMMY_NUMPY_DATA_SHAPE_3D
    ds_np3 = ds_np.reshape(DUMMY_NUMPY_DATA_SHAPE_3D)
    assert ds_np3.shape == (DUMMY_NUMPY_DATA_SHAPE_3D, _DEFAULT_SHAPE)

    ds_img3 = ds_np3.image_resize(IMG_SIZE)
    assert ds_img3.shape == ((*IMG_SIZE, 3), _DEFAULT_SHAPE)


def test_item_naming():
    ds = from_dummy_numpy_data()
    items = [x for x in ds]
    assert ds.names == []

    item_names = ["mydata", "mylabel"]

    # named transform syntax doesn't work without item_names
    with pytest.raises(Exception):
        ds.transform(moddata=reshape(DUMMY_NUMPY_DATA_SHAPE_2D))

    # passed one by one as arguments
    ds_named = ds.named(*item_names)
    assert ds.names == []  # old dataset is unchanged
    assert ds_named.names == item_names

    # passed in a list, overide previous
    item_names2 = ["moddata", "modlabel"]
    ds_named = ds.named(item_names2)  # type: ignore
    assert ds_named.names == item_names2

    # test named transform syntax
    ds_trans = ds_named.transform(moddata=reshape(DUMMY_NUMPY_DATA_SHAPE_2D))
    items_trans = [x for x in ds_trans]
    for (old_data, _), (new_data, _) in zip(items, items_trans):
        assert set(old_data) == set(new_data.flatten())
        assert old_data.shape != new_data.shape

    # invalid name doesn't work
    with pytest.raises(Exception):
        ds_named.transform(badname=reshape(DUMMY_NUMPY_DATA_SHAPE_2D))


def test_categorical():
    ds = (
        from_dummy_data(with_label=True)
        .reorder(0, 1, 1)
        .named("data", "label", "label_duplicate")
    )

    assert ds.unique("label") == ["a", "b"]

    ds_label = ds.categorical(1)

    # alternative syntaxes
    ds_label = ds.categorical(1)
    ds_label_alt1 = ds.categorical("label")
    ds_label_alt2 = ds.transform(label=categorical())
    ds_label_alt3 = ds.transform([None, categorical()])

    expected = [0, 1]

    for l, l1, l2, l3, e in zip(
        ds_label.unique("label"),
        ds_label_alt1.unique("label"),
        ds_label_alt2.unique("label"),
        ds_label_alt3.unique("label"),
        expected,
    ):
        assert np.array_equal(l, l1)
        assert np.array_equal(l, l2)
        assert np.array_equal(l, l3)
        assert np.array_equal(l, e)  # type:ignore

    assert list(ds_label) == [(d, 0 if l1 == "a" else 1, l2) for d, l1, l2 in ds]

    ds_label_userdef = ds.categorical("label", lambda x: 1 if x == "a" else 0)

    assert ds_label_userdef.unique("label") == [1, 0]
    assert list(ds_label_userdef) == [
        (d, 1 if l1 == "a" else 0, l2) for d, l1, l2 in ds
    ]

    # error scenarios
    with pytest.raises(TypeError):
        # we need to know what to label
        ds.categorical()  # type: ignore

    with pytest.raises(IndexError):
        ds.categorical(42)  # wrong key

    with pytest.raises(KeyError):
        ds.categorical("wrong")  # wrong key


def test_one_hot():
    ds = (
        from_dummy_data(with_label=True)
        .reorder(0, 1, 1)
        .named("data", "label", "label_duplicate")
    )
    assert ds.unique("label") == ["a", "b"]

    # alternative syntaxes
    ds_oh = ds.one_hot(1, encoding_size=2)
    ds_oh_alt1 = ds.one_hot("label", encoding_size=2)
    ds_oh_alt2 = ds.transform(label=one_hot(encoding_size=2))
    ds_oh_alt3 = ds.transform([None, one_hot(encoding_size=2)])

    ds_oh_auto = ds.one_hot("label")  # automatically compute encoding size

    expected = [np.array([True, False]), np.array([False, True])]

    for l, l1, l2, l3, la, e in zip(
        ds_oh.unique("label"),
        ds_oh_alt1.unique("label"),
        ds_oh_alt2.unique("label"),
        ds_oh_alt3.unique("label"),
        ds_oh_auto.unique("label"),
        expected,
    ):
        assert np.array_equal(l, l1)
        assert np.array_equal(l, l2)
        assert np.array_equal(l, l3)
        assert np.array_equal(l, la)
        assert np.array_equal(l, e)  # type:ignore

    for x, l, l2 in ds_oh:
        ind = 0 if l2 == "a" else 1
        assert np.array_equal(l, expected[ind])  # type:ignore

    # spiced up
    ds_oh_userdef = ds.one_hot(
        "label", encoding_size=3, mapping_fn=lambda x: 1 if x == "a" else 0, dtype="int"
    )

    for l, e in zip(
        ds_oh_userdef.unique("label"), [np.array([0, 1, 0]), np.array([1, 0, 0])]
    ):
        assert np.array_equal(l, e)  # type:ignore

    # error scenarios
    with pytest.raises(TypeError):
        # we need some arguments
        ds.one_hot()  # type:ignore

    with pytest.raises(IndexError):
        ds.one_hot(42, encoding_size=2)  # wrong key

    with pytest.raises(IndexError):
        list(
            ds.one_hot("label", encoding_size=1)
        )  # encoding size too small -- found at runtime

    with pytest.raises(KeyError):
        ds.one_hot("wrong", encoding_size=2)  # wrong key


def test_categorical_template():
    ds1 = from_dummy_data(with_label=True).named("data", "label")
    ds2 = ds1.shuffle(42)

    # when using categorical encoding on multiple datasets that are used together,
    # the encoding may turn out different. this is because the indexes are built
    # up and mapped as they are loaded (the order matters)
    assert set(ds1.transform(label=categorical())) != set(
        ds2.transform(label=categorical())
    )

    # we can use the categorical template to make matching encodings
    mapping_fn = categorical_template(ds1, "label")
    assert set(ds1.transform(label=categorical(mapping_fn))) == set(
        ds2.transform(label=categorical(mapping_fn))
    )

    # this is done implicitely when using the class-member functions
    assert set(ds1.categorical("label")) == set(ds2.categorical("label"))


def test_transform():
    ds = from_dummy_numpy_data().named("data", "label")

    # simple
    ds_itemwise = ds.transform([lambda x: x / 255.0])
    ds_keywise = ds.transform(data=lambda x: x / 255.0)
    ds_build = ds.transform(lambda x: (x[0] / 255.0, x[1]))

    for (d, l1), (di, li), (dk, lk), (db, lb) in zip(
        list(ds), list(ds_itemwise), list(ds_keywise), list(ds_build)
    ):
        assert np.array_equal(d / 255.0, di)
        assert np.array_equal(di, dk)
        assert np.array_equal(di, db)
        assert l1 == li == lk == lb

    # complex
    ds_complex = ds.transform(
        data=[reshape(DUMMY_NUMPY_DATA_SHAPE_2D), image_resize((10, 10))],
        label=one_hot(encoding_size=2),
    )

    assert ds_complex.shape == ((10, 10), (2,))

    # error scenarios
    with pytest.warns(UserWarning):
        # no args
        ds.transform()

    with pytest.raises(ValueError):
        # too many transforms given
        ds.transform([reshape(DUMMY_NUMPY_DATA_SHAPE_2D), None, None])


# ========= Tests relating to numpy data =========


def test_reshape():
    ds = from_dummy_numpy_data().named("data", "label")
    items = list(ds)

    assert ds.shape == (DUMMY_NUMPY_DATA_SHAPE_1D, _DEFAULT_SHAPE)
    assert ds[0][0].shape == DUMMY_NUMPY_DATA_SHAPE_1D

    # reshape adding extra dim
    ds_r = ds.reshape(DUMMY_NUMPY_DATA_SHAPE_2D)
    ds_r_alt = ds.reshape(data=DUMMY_NUMPY_DATA_SHAPE_2D)
    items_r = list(ds_r)
    items_r_alt = list(ds_r_alt)

    assert ds_r.shape == (DUMMY_NUMPY_DATA_SHAPE_2D, _DEFAULT_SHAPE)
    assert ds_r[0][0].shape == DUMMY_NUMPY_DATA_SHAPE_2D

    for (old_data, l1), (new_data, ln), (new_data_alt, lna) in zip(
        items, items_r, items_r_alt
    ):
        assert set(old_data) == set(new_data.flatten()) == set(new_data_alt.flatten())
        assert old_data.shape != new_data.shape == new_data_alt.shape
        assert l1 == ln == lna

    # use wildcard
    ds_wild = ds.reshape((-1, DUMMY_NUMPY_DATA_SHAPE_2D[1]))
    items_wild = list(ds_wild)
    for (old_data, _), (new_data, _) in zip(items_r, items_wild):
        assert np.array_equal(old_data, new_data)

    # reshape back, alternative syntax
    ds_back = ds_r.reshape(DUMMY_NUMPY_DATA_SHAPE_1D, None)
    items_back = [x for x in ds_back]

    for (old_data, _), (new_data, _) in zip(items, items_back):
        assert np.array_equal(old_data, new_data)

    # yet another syntax
    ds_trans = ds.transform([reshape(DUMMY_NUMPY_DATA_SHAPE_2D)])
    items_trans = [x for x in ds_trans]
    for (old_data, _), (new_data, _) in zip(items_r, items_trans):
        assert np.array_equal(old_data, new_data)

    # doing nothing also works
    with pytest.warns(UserWarning):
        ds_dummy = ds.reshape(None, None)
    items_dummy = [x for x in ds_dummy]
    for (old_data, _), (new_data, _) in zip(items, items_dummy):
        assert np.array_equal(old_data, new_data)

    # TODO test reshape on string data
    ds_str = loaders.from_folder_data(get_test_dataset_path(DATASET_PATHS.FOLDER_DATA))

    with pytest.raises(ValueError):
        # string has no shape
        ds_str.reshape((1, 2))

    with pytest.warns(UserWarning):
        # No input
        ds.reshape()

    with pytest.raises(TypeError):
        # bad input
        ds.reshape("whazzagh")  # type:ignore

    with pytest.warns(UserWarning):
        # Too many inputs
        ds.reshape(None, None, None)

    with pytest.raises(ValueError):
        # Dimensions don't match
        ds.reshape((13, 13))


# ========= Tests relating to image data =========


def test_numpy_image_numpy_conversion():
    ds_1d = from_dummy_numpy_data()
    items_1d = [x for x in ds_1d]

    # Warns because no elements where converted
    with pytest.warns(None) as record:
        ds2 = ds_1d.image()  # skipped all because they could't be converted
        ds3 = ds_1d.image(False, False)
    assert len(record) == 2  # warns on both

    # The two previous statements didn't create any changes
    items2 = [x for x in ds2]
    items3 = [x for x in ds3]
    for (one, _), (two, _), (three, _) in zip(items_1d, items2, items3):
        assert np.array_equal(one, two)
        assert np.array_equal(two, three)

    # Force conversion of first arg - doesn't work due to shape incompatibility
    with pytest.raises(Exception):
        # Tries to convert first argument
        ds_1d.image(True)

    ds_2d = ds_1d.reshape(DUMMY_NUMPY_DATA_SHAPE_2D)
    items_2d = [x for x in ds_2d]

    # Succesful conversion should happen here
    with pytest.warns(None) as record:
        ds_img = ds_2d.image()
    assert len(record) == 0

    items_img = [x for x in ds_img]
    for (one, lbl1), (two, lbl2) in zip(items_2d, items_img):
        assert type(one) == np.ndarray
        assert type(two) == Image.Image
        assert lbl1 == lbl2

    # test the backward-conversion
    ds_np = ds_img.numpy()
    items_np = [x for x in ds_np]
    for (one, lbl1), (two, lbl2) in zip(items_2d, items_np):
        assert type(one) == type(two)
        assert np.array_equal(one, two)
        assert lbl1 == lbl2

    # well get a warning if it doens't convert any
    with pytest.warns(UserWarning):
        ds_img.numpy(False)


def test_string_image_conversion():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATA)
    ds_str = loaders.from_folder_data(path)

    ds_img = ds_str.image()
    items_img = [x for x in ds_img]

    for data in items_img:
        data = data[0]
        assert issubclass(type(data), Image.Image)


def test_image_resize():
    ds = from_dummy_numpy_data().reshape(DUMMY_NUMPY_DATA_SHAPE_2D)
    for tpl in ds:
        data = tpl[0]
        assert data.shape == DUMMY_NUMPY_DATA_SHAPE_2D

    NEW_SIZE = (5, 5)

    # works directly on numpy arrays (ints)
    ds_resized = ds.image_resize(NEW_SIZE)
    for tpl in ds_resized:
        data = tpl[0]
        assert data.size == NEW_SIZE
        assert data.mode == "L"  # grayscale int

    # also if they are floats
    ds_resized_float = ds.transform([lambda x: np.float32(x)]).image_resize(NEW_SIZE)
    for tpl in ds_resized_float:
        data = tpl[0]
        assert data.size == NEW_SIZE
        assert data.mode == "F"  # grayscale float

    # works directly on strings
    ds_str = loaders.from_folder_data(get_test_dataset_path(DATASET_PATHS.FOLDER_DATA))
    ds_resized_from_str = ds_str.image_resize(NEW_SIZE)
    for tpl in ds_resized_from_str:
        data = tpl[0]
        assert data.size == NEW_SIZE

    # works on other images (scaling down)
    ds_resized_again = ds_resized.image_resize(DUMMY_NUMPY_DATA_SHAPE_2D)
    for tpl in ds_resized_again:
        data = tpl[0]
        assert data.size == DUMMY_NUMPY_DATA_SHAPE_2D

    # Test error scenarios
    with pytest.warns(UserWarning):
        ds.image_resize()  # No args

    with pytest.raises(ValueError):
        ds.image_resize(NEW_SIZE, NEW_SIZE, NEW_SIZE)  # Too many args

    with pytest.raises(AssertionError):
        ds.image_resize((4, 4, 4))  # Invalid size


# ========= Framework converters =========


@pytest.mark.slow
def test_to_tensorflow():
    # prep data
    ds = from_dummy_numpy_data().named("data", "label").one_hot("label").shuffle(42)
    tf_ds = ds.to_tensorflow().batch(2)

    # prep model
    import tensorflow as tf  # type:ignore

    tf.random.set_seed(42)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(ds.shape[0]),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # model should be able to fit the data
    model.fit(tf_ds, epochs=10)
    preds = model.predict(tf_ds)
    pred_labels = np.argmax(preds, axis=1)

    expected_labels = np.array([v[0] for v in ds.reorder("label").categorical("label")])
    assert sum(pred_labels == expected_labels) > len(ds) // 2  # type:ignore


@pytest.mark.slow
def test_image_to_tensorflow():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_GROUP_DATA)
    test, train = loaders.from_folder_dataset_group_data(path)

    def read_text(path):
        with open(path, "r") as file:
            return file.read()

    def read_bin(path):
        return np.fromfile(path, dtype=np.float32, count=-1)  # type:ignore

    test1, test2 = (
        test.image(False, True, False)
        .transform((read_text, None, None))
        .transform((None, None, read_bin))
        .split([0.3, -1], 2605)
    )

    tfds = test1.to_tensorflow()
    tfds2 = test1.image_resize(None, (10, 10), None).to_tensorflow()

    for data in tfds:
        pass

    for data in tfds2:
        pass

    assert True


@pytest.mark.slow
def test_to_pytorch():
    # prep data
    ds = from_dummy_numpy_data().named("data", "label").one_hot("label")
    pt_ds = ds.to_pytorch()

    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(pt_ds, batch_size=2, shuffle=False)

    elem = next(iter(loader))

    # data equals
    assert torch.all(torch.eq(elem[0][0], torch.Tensor(ds[0][0])))  # type:ignore
    assert torch.all(torch.eq(elem[0][1], torch.Tensor(ds[1][0])))  # type:ignore

    # labels equal
    assert torch.all(torch.eq(elem[1][0], torch.Tensor(ds[0][1])))  # type:ignore
    assert torch.all(torch.eq(elem[1][1], torch.Tensor(ds[1][1])))  # type:ignore


def test_stats():
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # numpy data
    ds_np = from_dummy_numpy_data().reshape(DUMMY_NUMPY_DATA_SHAPE_3D)

    std_scaler = StandardScaler()
    mm_scaler = MinMaxScaler()

    axis = 2

    def _make_scaler_reshapes(data_shape: Sequence[int], axis: int = 0):

        ishape = list(data_shape)
        ishape[0], ishape[axis] = ishape[axis], ishape[0]

        def reshape_to_scale(d):
            return np.swapaxes(  # type:ignore
                np.swapaxes(d, 0, axis).reshape((data_shape[axis], -1)),  # type:ignore
                0,
                1,
            )

        def reshape_from_scale(d):
            return np.swapaxes(  # type:ignore
                np.swapaxes(d, 0, 1).reshape(ishape), 0, axis  # type:ignore
            )

        return reshape_to_scale, reshape_from_scale

    reshape_to_scale, reshape_from_scale = _make_scaler_reshapes(ds_np[0][0].shape, 2)

    for d, l in ds_np:
        old_shape = d.shape
        # stats are accumulated along 2nd axis in sklearn preprocessing:
        # reshape accordingly
        dlist = np.swapaxes(  # type:ignore
            np.swapaxes(d, 0, axis).reshape((old_shape[axis], -1)), 0, 1  # type:ignore
        )
        std_scaler.partial_fit(dlist)
        mm_scaler.partial_fit(dlist)
