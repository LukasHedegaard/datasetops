
from mldatasets.loaders import DataGetter
from mldatasets.datasets.dataset import Dataset

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
    ds_shuffled = ds.shuffle()
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

