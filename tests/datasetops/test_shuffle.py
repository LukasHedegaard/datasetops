import pytest
from datasetops.loaders import Loader


def _get_data(i):
    return i


def from_dummy_data() -> Loader:

    a_ids = list(range(5))
    b_ids = list(range(5, 11))

    ds = Loader(_get_data)
    ds.extend(a_ids)
    ds.extend(b_ids)
    return ds


def test_noSeed_valid():
    ds = from_dummy_data()

    ds.shuffle()


def test_emptyDataset_valid():

    ds = Loader(_get_data)
    ds.shuffle()
    assert len(ds) == 0


def test_shuffleStringIds_valid():
    def _get_data(i):
        return i

    ds = Loader(_get_data)
    ds.extend(["1", "2"])

    ds_shuffled = ds.shuffle()

    assert "1" in ds_shuffled
    assert "2" in ds_shuffled


def test_containsSameElements():

    ds = from_dummy_data()

    expected_items = [i for i in ds]
    ds_shuffled = ds.shuffle()
    found_items = [i for i in ds_shuffled]

    assert set(expected_items) == set(found_items)


def test_elementsShuffled():

    seed = 42

    ds = from_dummy_data()

    expected_items = [i for i in ds]
    ds_shuffled = ds.shuffle(seed)
    found_items = [i for i in ds_shuffled]

    assert expected_items != found_items
