
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


# def test_shuffle():
#     seed = 42
#     ds = load_dummy_data().shuffle()
#     assert(...)


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