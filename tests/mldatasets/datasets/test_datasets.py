
from mldatasets.loaders import DataGetter
from mldatasets.datasets.dataset import Dataset, _DEFAULT_SHAPE
import pytest
import numpy as np

def load_dummy_data() -> Dataset:

    a_ids = list(range(5))
    b_ids = list(range(5,11))

    def get_data(i):
        return i

    datagetter = DataGetter(get_data)

    ds = Dataset(downstream_getter=datagetter)
    ds._extend(a_ids, 'a')
    ds._extend(b_ids, 'b')
    return ds


def test_class_names():
    ds = load_dummy_data()
    assert(set(ds.class_names()) == set(['a','b']))


def test_class_counts():
    ds = load_dummy_data()
    assert(set(ds.class_counts().items()) == set([('a', 5),('b',6)]))


def test_shuffle():
    seed = 42
    ds = load_dummy_data()
    expected_items = [i for i in ds]
    ds_shuffled = ds.shuffle(seed)
    found_items = [i for i in ds_shuffled]

    # same data
    assert(set(expected_items) == set(found_items))

    # different sequence
    assert(expected_items != found_items)


def test_sample():
    seed = 42
    ds = load_dummy_data()
    ds_sampled = ds.sample(5, seed)
    found_items = [i for i in ds_sampled]

    # check list uniqueness
    assert(len(found_items) == len(set(found_items)))

    # check items
    expected_items = [10,1,0,4,9]
    assert(set(expected_items) == set(found_items))

    # check that different seeds yield different results
    ds_sampled2 = ds.sample(5, seed+1)
    found_items2 = [i for i in ds_sampled2]
    assert(set(found_items2) != set(found_items))


def test_sample_classwise():
    seed = 42
    num_per_class = 2
    ds = load_dummy_data()
    ds_sampled = ds.sample_classwise(num_per_class, seed)
    found_items = [i for i in ds_sampled]

    # check list uniqueness
    assert(len(found_items) == len(set(found_items)))

    # check equal count
    assert(set(ds_sampled.class_counts().items()) == set([('a', num_per_class),('b', num_per_class)]))

    # check items
    expected_items = [0,4,10,7]
    assert(set(expected_items) == set(found_items))

    # check that different seeds yield different results
    ds_sampled2 = ds.sample_classwise(num_per_class, seed+1)
    found_items2 = [i for i in ds_sampled2]
    assert(set(found_items2) != set(found_items))


def test_split():
    seed = 42
    ds = load_dummy_data()
    ds1, ds2, ds3 = ds.split([0.6, 0.3, 0.1], seed=seed)

    # new sets are distinct
    assert(set(ds1) != set(ds2))
    assert(set(ds1) != set(ds3))
    assert(set(ds2) != set(ds3))

    # no values are lost
    assert(set(ds) == set(ds1).union(set(ds2),set(ds3)))

    # repeat for wildcard
    ds1w, ds2w, ds3w = ds.split([0.6, -1, 0.1], seed=seed)

    # using wildcard produces same results
    assert(set(ds1) == set(ds1w))
    assert(set(ds2) == set(ds2w))
    assert(set(ds3) == set(ds3w))



########## Tests relating to numpy data #########################

DUMMY_NUMPY_DATA_SHAPE_2D = (4,4)
DUMMY_NUMPY_DATA_SHAPE_1D = 4*4

def load_dummy_numpy_data() -> Dataset:

    a_ids = list(range(5))
    b_ids = list(range(5,11))
    # labels = [
    #     *[np.array([1]) for _ in a_ids],
    #     *[np.array([2]) for _ in b_ids]
    # ]
    labels = [
        *[1 for _ in a_ids],
        *[2 for _ in b_ids]
    ]

    num_samples = len(a_ids)+len(b_ids)
    data = np.arange(num_samples*DUMMY_NUMPY_DATA_SHAPE_1D).reshape((num_samples, DUMMY_NUMPY_DATA_SHAPE_1D))

    def get_data(idx):
        return data[idx], labels[idx]

    datagetter = DataGetter(get_data)

    ds = Dataset(downstream_getter=datagetter)
    ds._extend(a_ids, '1')
    ds._extend(b_ids, '2')
    return ds


def test_reshape():
    ds = load_dummy_numpy_data()
    items = [x for x in ds]

    s = ds.shape
    assert(ds.shape == ((DUMMY_NUMPY_DATA_SHAPE_1D,), _DEFAULT_SHAPE) )
    assert(ds[0][0].shape == (DUMMY_NUMPY_DATA_SHAPE_1D,))

    # reshape adding extra dim
    ds_r = ds.reshape(DUMMY_NUMPY_DATA_SHAPE_2D)
    items_r = [x for x in ds_r]

    assert(ds_r.shape == ( DUMMY_NUMPY_DATA_SHAPE_2D, _DEFAULT_SHAPE) )
    assert(ds_r[0][0].shape == DUMMY_NUMPY_DATA_SHAPE_2D)

    for (old_data, old_label), (new_data, new_label) in zip(items, items_r):
        assert(set(old_data) == set(new_data.flatten()))
        assert(old_data.shape != new_data.shape)

    # use wildcard
    ds_wild = ds.reshape((-1,DUMMY_NUMPY_DATA_SHAPE_2D[1]))
    items_wild = [x for x in ds_wild]
    for (old_data, old_label), (new_data, new_label) in zip(items_r, items_wild):
        assert(np.array_equal(old_data, new_data))

    # reshape back, alternative syntax
    ds_back = ds_r.reshape((DUMMY_NUMPY_DATA_SHAPE_1D,), None)
    items_back = [x for x in ds_back]

    for (old_data, old_label), (new_data, new_label) in zip(items, items_back):
        assert(np.array_equal(old_data, new_data))

    # doing nothing also works
    ds_dummy = ds.reshape(None, None)
    items_dummy = [x for x in ds_dummy]
    for (old_data, old_label), (new_data, new_label) in zip(items, items_dummy):
        assert(np.array_equal(old_data, new_data))

    with pytest.raises(ValueError):
        ds.reshape() # No input
    
    with pytest.raises(ValueError):
        ds.reshape(None, None, None) # Too many inputs

    with pytest.raises(ValueError):
        ds.reshape((13,13)) # Dimensions don't match