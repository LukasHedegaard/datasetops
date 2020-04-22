from .utils import from_dummy_data
import pytest


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


def test_shuffle():
    ds = from_dummy_data()

    # no seed
    ds_shuffled = ds.shuffle()
    assert set(ds) == set(ds_shuffled)  # same data
    assert list(ds) != list(ds_shuffled)  # different sequence

    ds_shuffled = ds.shuffle(seed=42)
    assert set(ds) == set(ds_shuffled)  # same data
    assert list(ds) != list(ds_shuffled)  # different sequence


def test_take():
    ds = from_dummy_data().transform(0, lambda x: 10 * x)

    ds_5 = ds.take(5)
    assert list(ds)[:5] == list(ds_5)

    with pytest.raises(IndexError):
        ds.take(10000000)


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
