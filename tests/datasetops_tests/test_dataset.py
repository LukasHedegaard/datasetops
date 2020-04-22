import pytest

from datasetops import Dataset
from datasetops.dataset import _DEFAULT_SHAPE

from .utils import (
    from_dummy_data,
    from_dummy_numpy_data,
    DUMMY_NUMPY_DATA_SHAPE_1D,
)


def test_len():
    ds = from_dummy_data(num_total=11)
    assert len(ds) == 11


def test_getitem():

    ds = from_dummy_data(num_total=11)

    # getting by index
    assert ds[0] == (0,)  # lower boundary
    assert ds[10] == (10,)  # upper boundary

    with pytest.raises(IndexError):
        ds[11]  # beyond boundary

    # slicing
    assert ds[:1] == [(0,)]  # start, single
    assert ds[2:4] == [(2,), (3,)]  # multiple
    assert ds[9:] == [(9,), (10,)]  # end
    assert ds[:] == list(ds)  # all
    assert ds[10:12] == [(10,)]  # beyond boundary doesn't trigger error


def test_iterator():
    ds = from_dummy_data(num_total=11)
    expected = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]
    assert list(ds) == [s for s in ds] == expected


def test_generator():
    ds = from_dummy_data()
    gen = ds.generator()

    for d in list(ds):
        assert d == next(gen)


def test_shape():
    assert Dataset.from_iterable([1]).shape == (_DEFAULT_SHAPE,)
    assert Dataset.from_iterable([(1, 1)]).shape == (_DEFAULT_SHAPE, _DEFAULT_SHAPE)

    # numpy data
    assert from_dummy_numpy_data().shape == (DUMMY_NUMPY_DATA_SHAPE_1D, _DEFAULT_SHAPE)


def test_item_naming():
    ds = from_dummy_data(with_label=True)

    assert ds.names == []

    item_names = ["mydata", "mylabel"]

    assert (
        ds.named(item_names).names
        == ds.named(*item_names).names
        == ds.named(tuple(item_names)).names
        == item_names
    )

    # samples don't change
    assert list(ds) == list(ds.named(item_names))


def test_reorder():
    ds = from_dummy_data(with_label=True).named("mydata", "mylabel")

    # error scenarios
    with pytest.raises(IndexError):
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
        assert ldata == rdata == rdata1 == rdata2
        assert llbl == rlbl == rlbl1 == rlbl2

    # same results using keys
    ds_re_key = ds.reorder("mylabel", "mydata")
    for (llbl, ldata), (rlbl, rdata) in zip(list(ds_re_key), list(ds_re)):
        assert ldata == rdata
        assert llbl == rlbl

    # same result using a mix
    ds_re_mix = ds.reorder(1, "mydata")
    for (llbl, ldata), (rlbl, rdata) in zip(list(ds_re_mix), list(ds_re)):
        assert ldata == rdata
        assert llbl == rlbl

    # we can even place the same element multiple times
    with pytest.warns(UserWarning):
        # but discard item-names (gives a warning)
        ds_re_creative = ds.reorder(0, 1, 1, 0)

    for (ldata, llbl), (rdata1, rlbl1, rlbl2, rdata2) in zip(
        list(ds), list(ds_re_creative)
    ):
        assert ldata == rdata1 == rdata2
        assert llbl == rlbl1 == rlbl2

    # shape updates accordingly
    assert ds_re_creative.shape == (
        _DEFAULT_SHAPE,
        _DEFAULT_SHAPE,
        _DEFAULT_SHAPE,
        _DEFAULT_SHAPE,
    )


def test_trace():
    ds = from_dummy_data(with_label=True).named("data", "label").reorder(1, 0)
    assert ds.trace() == {
        "operation_name": "reorder",
        "operation_parameters": {"keys": [1, 0]},
        "parent": {
            "operation_name": "named",
            "operation_parameters": {"names": ["data", "label"]},
            "parent": {
                "operation_name": "from_iterable",
                "operation_parameters": {},
                "parent": "from_dummy_data",
            },
        },
    }


def test_transform():
    ds = from_dummy_data(with_label=True).named("data", "label")

    expected = [
        (1, "a"),
        (2, "a"),
        (3, "a"),
        (4, "a"),
        (5, "a"),
        (6, "b"),
        (7, "b"),
        (8, "b"),
        (9, "b"),
        (10, "b"),
        (11, "b"),
    ]

    # elemement wise
    assert (
        list(ds.transform(0, lambda x: x + 1))
        == list(ds.transform("data", lambda x: x + 1))
        == expected
    )

    # whole
    assert list(ds.transform(lambda x: tuple((x[0] + 1, x[1])))) == expected
