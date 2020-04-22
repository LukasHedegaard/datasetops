from .utils import from_dummy_data
from datasetops import filters
import pytest


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


def test_split_filter():
    num_total = 10
    ds = from_dummy_data(num_total=num_total, with_label=True).named("data", "label")

    # expected items
    a = [(x, str("a")) for x in list(range(5))]
    b = [(x, str("b")) for x in list(range(5, num_total))]
    even_a = [x for x in a if x[0] % 2 == 0]
    odd_a = [x for x in a if x[0] % 2 == 1]
    even_b = [x for x in b if x[0] % 2 == 0]
    odd_b = [x for x in b if x[0] % 2 == 1]

    # element wise
    ds_even, ds_odd = ds.split_filter("data", lambda x: x % 2 == 0)
    assert list(ds_even) == even_a + even_b
    assert list(ds_odd) == odd_a + odd_b

    # sample wise
    ds_even_a, ds_not_even_a = ds.split_filter(lambda s: s[0] % 2 == 0 and s[1] == "a")
    assert list(ds_even_a) == even_a
    assert list(ds_not_even_a) == odd_a + b

    # sample_classwise
    ds_classwise_2, ds_classwise_rest = ds.split_filter(
        "label", filters.allow_unique(2)
    )
    assert list(ds_classwise_2) == list(a[:2] + b[:2])
    assert list(ds_classwise_rest) == list(a[2:] + b[2:])

    # error scenarios
    with pytest.raises(KeyError):
        ds.split_filter("badkey", lambda x: True)  # key doesn't exist


def test_split_train_test():

    seed = 42
    ds = from_dummy_data()

    # default splits
    ds1, ds2 = ds.split_train_test(seed=seed)

    assert round(len(ds1) / len(ds), 1) == 0.8
    assert round(len(ds2) / len(ds), 1) == 0.2

    assert set(ds1) != set(ds2)
    assert set.union(set(ds1), set(ds2)) == set(ds)

    # custom splits
    ds1, ds2 = ds.split_train_test([0.5, 0.5], seed=seed)
    assert round(len(ds1) / len(ds), 1) == 0.5
    assert round(len(ds2) / len(ds), 1) == 0.5

    assert set(ds1) != set(ds2)
    assert set.union(set(ds1), set(ds2)) == set(ds)

    with pytest.raises(AssertionError):
        ds.split_train_test([1, 2, 3], seed)

    with pytest.raises(AssertionError):
        ds.split_train_test([], seed)


def test_split_train_val_test():
    seed = 42
    ds = from_dummy_data()

    # default splits
    ds1, ds2, ds3 = ds.split_train_val_test([0.2, 0.2, -1], seed=seed)

    assert round(len(ds1) / len(ds), 1) == 0.2
    assert round(len(ds2) / len(ds), 1) == 0.2
    assert round(len(ds3) / len(ds), 1) == 0.6

    assert set(ds1) != set(ds2) != set(ds3) != set(ds1)
    assert set.union(set(ds1), set(ds2), set(ds3)) == set(ds)
