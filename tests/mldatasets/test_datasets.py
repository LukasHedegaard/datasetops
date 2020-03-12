
from mldatasets.dataset import Dataset, reshape, custom, _DEFAULT_SHAPE
from mldatasets.loaders import FunctionDataset
import mldatasets.loaders as loaders
import pytest
import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(__file__))
from testing_utils import get_test_dataset_path, DATASET_PATHS # type:ignore

def load_dummy_data() -> FunctionDataset:

    a_ids = list(range(5))
    b_ids = list(range(5,11))

    def get_data(i):
        return i

    ds = FunctionDataset(get_data)
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

def load_dummy_numpy_data() -> FunctionDataset:

    a_ids = list(range(5))
    b_ids = list(range(5,11))
    labels = [
        *[1 for _ in a_ids],
        *[2 for _ in b_ids]
    ]

    num_samples = len(a_ids)+len(b_ids)
    data = np.arange(num_samples*DUMMY_NUMPY_DATA_SHAPE_1D).reshape((num_samples, DUMMY_NUMPY_DATA_SHAPE_1D))
    # data = data / data.max()

    def get_data(idx):
        return data[idx], labels[idx]

    ds = FunctionDataset(get_data)
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

    for (old_data, _), (new_data, _) in zip(items, items_r):
        assert(set(old_data) == set(new_data.flatten()))
        assert(old_data.shape != new_data.shape)

    # use wildcard
    ds_wild = ds.reshape((-1,DUMMY_NUMPY_DATA_SHAPE_2D[1]))
    items_wild = [x for x in ds_wild]
    for (old_data, _), (new_data, _) in zip(items_r, items_wild):
        assert(np.array_equal(old_data, new_data))

    # reshape back, alternative syntax
    ds_back = ds_r.reshape((DUMMY_NUMPY_DATA_SHAPE_1D,), None)
    items_back = [x for x in ds_back]

    for (old_data, _), (new_data, _) in zip(items, items_back):
        assert(np.array_equal(old_data, new_data))

    # yet another syntax
    ds_trans = ds.transform(reshape(DUMMY_NUMPY_DATA_SHAPE_2D))
    items_trans = [x for x in ds_trans]
    for (old_data, _), (new_data, _) in zip(items_r, items_trans):
        assert(np.array_equal(old_data, new_data))

    # doing nothing also works
    ds_dummy = ds.reshape(None, None)
    items_dummy = [x for x in ds_dummy]
    for (old_data, _), (new_data, _) in zip(items, items_dummy):
        assert(np.array_equal(old_data, new_data))

    with pytest.warns(UserWarning):
        ds.reshape() # No input
    
    with pytest.raises(ValueError):
        ds.reshape(None, None, None) # Too many inputs

    with pytest.raises(ValueError):
        ds.reshape((13,13)) # Dimensions don't match


def test_item_naming():
    ds = load_dummy_numpy_data()
    items = [x for x in ds]
    assert(ds.item_names == [])

    item_names = ['mydata', 'mylabel']

    # named transform syntax doesn't work without item_names
    with pytest.raises(Exception):
        ds.transform(moddata=reshape(DUMMY_NUMPY_DATA_SHAPE_2D))

    # passed one by one as arguments
    ds.set_item_names(*item_names)
    assert(ds.item_names == item_names)

    # passed in a list, overide previous
    item_names2 = ['moddata', 'modlabel']
    ds.set_item_names(item_names2) #type: ignore
    assert(ds.item_names == item_names2)

    # test named transform syntax
    ds_trans = ds.transform(moddata=reshape(DUMMY_NUMPY_DATA_SHAPE_2D))
    items_trans = [x for x in ds_trans]
    for (old_data, _), (new_data, _) in zip(items, items_trans):
        assert(set(old_data) == set(new_data.flatten()))
        assert(old_data.shape != new_data.shape)

    # invalid name doesn't work
    with pytest.raises(Exception):
        ds.transform(badname=reshape(DUMMY_NUMPY_DATA_SHAPE_2D))


def test_custom_transform():
    ds = load_dummy_numpy_data()
    items = [x for x in ds]

    ds_tf = ds.transform(custom(lambda x: x/255.0))
    items_tf = [x for x in ds_tf]
    
    for (ldata,llbl), (rdata, rlbl) in zip(items, items_tf):
        assert(np.array_equal(ldata/255.0, rdata))
        assert(llbl == rlbl)

    # passing the function directly also works
    ds_tf_alt = ds.transform(lambda x: x/255.0) # type:ignore
    items_tf_alt = [x for x in ds_tf]

    for (ldata,llbl), (rdata, rlbl) in zip(items_tf_alt, items_tf):
        assert(np.array_equal(ldata, rdata))
        assert(llbl == rlbl)


########## Tests relating to image data #########################

def test_numpy_image_numpy_conversion():
    ds_1d = load_dummy_numpy_data()
    items_1d = [x for x in ds_1d]

    # Warns because no elements where converted
    with pytest.warns(None) as record:
        ds2 = ds_1d.as_img() # skipped all because they could't be converted
        ds3 = ds_1d.as_img(False, False)
    assert(len(record) == 2) # warns on both

    # The two previous statements didn't create any changes
    items2 = [x for x in ds2]
    items3 = [x for x in ds3]
    for (one, _), (two, _), (three, _) in zip(items_1d, items2, items3):
        assert(np.array_equal(one, two))
        assert(np.array_equal(two, three))

    # Force conversion of first arg - doesn't work due to shape incompatibility
    with pytest.raises(Exception):
        # Tries to convert first argument
        ds_1d.as_img(True)

    ds_2d = ds_1d.reshape(DUMMY_NUMPY_DATA_SHAPE_2D)
    items_2d = [x for x in ds_2d]

    # Succesful conversion should happen here
    with pytest.warns(None) as record:
        ds_img = ds_2d.as_img()
    assert(len(record) == 0)

    items_img = [x for x in ds_img]
    for (one, lbl1), (two, lbl2) in zip(items_2d, items_img):
        assert(type(one) == np.ndarray)
        assert(type(two) == Image.Image)
        assert(lbl1 == lbl2)

    # test the backward-conversion
    ds_np = ds_img.as_numpy()
    items_np = [x for x in ds_np]
    for (one, lbl1), (two, lbl2) in zip(items_2d, items_np):
        assert(type(one) == type(two))
        assert(np.array_equal(one, two))
        assert(lbl1 == lbl2)


def test_string_image_conversion():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATA)
    ds_str = loaders.load_folder_data(path)

    ds_img = ds_str.as_img()
    items_img = [x for x in ds_img]

    for data in items_img:
        data = data[0]
        assert(issubclass(type(data), Image.Image))


def test_resize():
    ds = load_dummy_numpy_data().reshape(DUMMY_NUMPY_DATA_SHAPE_2D)
    for tpl in ds:
        data = tpl[0]
        assert(data.shape == DUMMY_NUMPY_DATA_SHAPE_2D)

    NEW_SIZE = (5,5)

    # works directly on numpy arrays (ints)
    ds_resized = ds.img_resize(NEW_SIZE)
    for tpl in ds_resized:
        data = tpl[0]
        assert(data.size == NEW_SIZE)
        assert(data.mode == 'L') # grayscale int

    # also if they are floats
    ds_resized_float = ds.transform(custom(np.float32)).img_resize(NEW_SIZE)
    for tpl in ds_resized_float:
        data = tpl[0]
        assert(data.size == NEW_SIZE)
        assert(data.mode == 'F') # grayscale float

    # works directly on strings
    ds_str = loaders.load_folder_data(get_test_dataset_path(DATASET_PATHS.FOLDER_DATA))
    ds_resized_from_str = ds_str.img_resize(NEW_SIZE)
    for tpl in ds_resized_from_str:
        data = tpl[0]
        assert(data.size == NEW_SIZE)

    # works on other images (scaling down)
    ds_resized_again = ds_resized.img_resize(DUMMY_NUMPY_DATA_SHAPE_2D)
    for tpl in ds_resized_again:
        data = tpl[0]
        assert(data.size == DUMMY_NUMPY_DATA_SHAPE_2D)

    # Test error scenarios
    with pytest.warns(UserWarning):
        ds.img_resize() # No args

    with pytest.raises(ValueError):
        ds.img_resize(NEW_SIZE, NEW_SIZE, NEW_SIZE) # Too many args

    with pytest.raises(AssertionError):
        ds.img_resize((4,4,4)) # Invalid size


    