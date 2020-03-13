import pytest

from mldatasets.loaders import FunctionDataset


def load_dummy_data() -> FunctionDataset:

    a_ids = list(range(5))
    b_ids = list(range(5, 11))

    def get_data(i):
        return i

    ds = FunctionDataset(get_data)
    ds._extend(a_ids, 'a')
    ds._extend(b_ids, 'b')
    return ds


def test_noSeed_valid():
    ds = load_dummy_data()

    ds.shuffle()


def test_emptyDataset_valid():

    def get_data(i):
        return i

    ds = FunctionDataset(get_data)

    ds.shuffle()


def test_shuffleStringIds_valid():

    def get_data(i):
        return i

    ds = FunctionDataset(get_data)
    ds._extend(['1', '2'])

    ds_shuffled = ds.shuffle()

    diff = {'1', '2'}.difference(ds_shuffled._ids)
    assert(diff == {})


def test_containsSameElements():

    ds = load_dummy_data()

    expected_items = [i for i in ds]
    ds_shuffled = ds.shuffle()
    found_items = [i for i in ds_shuffled]

    assert(set(expected_items) == set(found_items))


def test_elementsShuffled():

    seed = 42

    ds = load_dummy_data()

    expected_items = [i for i in ds]
    ds_shuffled = ds.shuffle(seed)
    found_items = [i for i in ds_shuffled]

    assert(expected_items != found_items)
