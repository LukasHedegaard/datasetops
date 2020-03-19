
from datasetops.dataset import Dataset, reshape, custom, allow_unique, one_hot, label, _DEFAULT_SHAPE
from datasetops.function_dataset import FunctionDataset
import datasetops.loaders as loaders
import pytest
import numpy as np
from PIL import Image
import sys
import os
from typing import List
sys.path.append(os.path.dirname(__file__))
from testing_utils import ( # type:ignore
    get_test_dataset_path, load_dummy_data, load_dummy_numpy_data,
    DATASET_PATHS, DUMMY_NUMPY_DATA_SHAPE_1D, DUMMY_NUMPY_DATA_SHAPE_2D, DUMMY_NUMPY_DATA_SHAPE_3D
)


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
    expected_items = [ (i,) for i in [10,1,0,4,9]]
    assert(set(expected_items) == set(found_items))

    # check that different seeds yield different results
    ds_sampled2 = ds.sample(5, seed+1)
    found_items2 = [i for i in ds_sampled2]
    assert(set(found_items2) != set(found_items))


def test_filter():
    num_total=10
    ds = load_dummy_data(num_total=num_total, with_label=True).set_item_names('data', 'label')

    # expected items
    a = [ (x, 'a') for x in list(range(5))]
    b = [ (x, 'b') for x in list(range(5,num_total))]
    even_a = [x for x in a if x[0]%2==0]
    odd_a  = [x for x in a if x[0]%2==1]
    even_b = [x for x in b if x[0]%2==0]
    odd_b  = [x for x in b if x[0]%2==1]

    # itemwise
    ds_even = ds.filter(itemwise=[lambda x: x%2==0])
    assert(list(ds_even) == even_a + even_b) 

    ds_even_a = ds.filter(itemwise=[lambda x: x%2==0, lambda x: x=='a'])
    assert(list(ds_even_a) == even_a) 

    # by key
    ds_b = ds.filter(label=lambda x:x=='b')
    assert(list(ds_b) == b)

    # bulk
    ds_odd_b = ds.filter(lambda x: x[0]%2==1 and x[1]=='b')
    assert(list(ds_odd_b) == odd_b)

    # mix
    ds_even_b_no_4 = ds.filter(lambda x: x[0]!= 4, itemwise=[lambda x: x%2==0], label=lambda x: x=='b')
    assert(list(ds_even_b_no_4) == [x for x in even_b if x[0]!=4])

    # sample_classwise
    ds_classwise = ds.filter(label=allow_unique(2))
    assert(list(ds_classwise) == list(a[:2] + b[:2]))

    # error scenarios
    with pytest.warns(UserWarning):
        ds_same = ds.filter() # no args
        assert(list(ds) == list(ds_same))

    with pytest.raises(AssertionError):
        ds.filter(itemwise=[None, None, None]) # too many args

    with pytest.raises(AssertionError):
        ds.filter(badkey=lambda x:True) # key doesn't exist


def test_filter_split():
    num_total=10
    ds = load_dummy_data(num_total=num_total, with_label=True).set_item_names('data', 'label')

    # expected items
    a = [ (x, 'a') for x in list(range(5))]
    b = [ (x, 'b') for x in list(range(5,num_total))]
    even_a = [x for x in a if x[0]%2==0]
    odd_a  = [x for x in a if x[0]%2==1]
    even_b = [x for x in b if x[0]%2==0]
    odd_b  = [x for x in b if x[0]%2==1]

    # itemwise
    ds_even, ds_odd = ds.filter_split(itemwise=[lambda x: x%2==0])
    assert(list(ds_even) == even_a + even_b) 
    assert(list(ds_odd) == odd_a + odd_b) 

    ds_even_a, ds_not_even_a = ds.filter_split(itemwise=[lambda x: x%2==0, lambda x: x=='a'])
    assert(list(ds_even_a) == even_a) 
    assert(list(ds_not_even_a) == odd_a + b) 

    # by key
    ds_b, ds_a = ds.filter_split(label=lambda x:x=='b')
    assert(list(ds_b) == b)
    assert(list(ds_a) == a)

    # bulk
    ds_odd_b, ds_even_b = ds.filter_split(lambda x: x[0]%2==1 and x[1]=='b')
    assert(list(ds_odd_b) == odd_b)
    assert(list(ds_even_b) == a + even_b)

    # mix
    ds_even_b_no_4, ds_not_even_b_no_4 = ds.filter_split(lambda x: x[0]!= 4, itemwise=[lambda x: x%2==0], label=lambda x: x=='b')
    assert(list(ds_even_b_no_4) == [x for x in even_b if x[0]!=4])
    assert(list(ds_not_even_b_no_4) == [x for x in list(ds) if not x in [x for x in even_b if x[0]!=4]] )

    # sample_classwise
    ds_classwise_2, ds_classwise_rest = ds.filter_split(label=allow_unique(2))
    assert(list(ds_classwise_2) == list(a[:2] + b[:2]))
    assert(list(ds_classwise_rest) == list(a[2:] + b[2:]))

    # error scenarios
    with pytest.raises(ValueError):
        ds_same = ds.filter_split() # no args

    with pytest.raises(AssertionError):
        ds.filter_split(itemwise=[None, None, None]) # too many args

    with pytest.raises(AssertionError):
        ds.filter_split(badkey=lambda x:True) # key doesn't exist


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


def test_repeat():
    ds = load_dummy_data()

    # itemwise
    ds_item = ds.repeat(3)
    ds_item_alt = ds.repeat(3, mode='itemwise')

    assert(set(ds) == set(ds_item_alt))
    assert(list(ds_item) == list(ds_item_alt))

    # whole
    ds_whole = ds.repeat(2, mode='whole')
    assert(set(ds) == set(ds_whole))
    assert(list(ds) == list(ds_whole)[:len(ds)] == list(ds_whole)[len(ds):])


def test_take():
    ds = load_dummy_data().transform(lambda x: 10*x)

    ds_5 = ds.take(5)
    assert(list(ds)[:5] == list(ds_5))

    with pytest.raises(ValueError):
        ds.take(10000000)


def test_reorder():
    ds = load_dummy_numpy_data()
    ds.set_item_names("mydata", "mylabel")

    ## error scenarios
    with pytest.warns(UserWarning):
        # no order given
        ds_ignored = ds.reorder()
        assert(ds == ds_ignored)

    with pytest.raises(ValueError):
        # indexes out of range
        ds_re = ds.reorder(3,4)

    with pytest.raises(KeyError):
        # a keys doesn't exist
        ds_re = ds.reorder("badkey", "mydata")

    ## working scenarios

    # using indexes
    ds_re = ds.reorder(1,0)
    for (ldata, llbl), (rlbl, rdata) in zip(list(ds), list(ds_re)):
        assert(np.array_equal(ldata, rdata))
        assert(llbl == rlbl)

    # same results using keys
    ds_re_key = ds.reorder("mylabel","mydata")
    for (llbl, ldata), (rlbl, rdata) in zip(list(ds_re_key), list(ds_re)):
        assert(np.array_equal(ldata, rdata))
        assert(llbl == rlbl)

    # same result using a mix
    ds_re_mix = ds.reorder(1,"mydata")
    for (llbl, ldata), (rlbl, rdata) in zip(list(ds_re_mix), list(ds_re)):
        assert(np.array_equal(ldata, rdata))
        assert(llbl == rlbl)

    # we can even place the same element multiple times
    ds_re_creative = ds.reorder(0,1,1,0)
    for (ldata, llbl), (rdata1, rlbl1, rlbl2, rdata2 ) in zip(list(ds), list(ds_re_creative)):
        assert(np.array_equal(ldata, rdata1))
        assert(np.array_equal(ldata, rdata2))
        assert(llbl == rlbl1 == rlbl2)

    # shape updates accordingly
    assert(ds_re_creative.shape == (DUMMY_NUMPY_DATA_SHAPE_1D, _DEFAULT_SHAPE, _DEFAULT_SHAPE, DUMMY_NUMPY_DATA_SHAPE_1D))

    # error scenarios
    with pytest.warns(UserWarning):
        ds.set_item_names('one','two').reorder(0,1,1) # key needs to be unique, but wouldn't be


########## Tests relating to stats #########################

def test_counts():
    num_total=11
    ds = load_dummy_data(num_total=num_total, with_label=True).set_item_names('data', 'label')

    counts = ds.counts('label') # name based
    counts_alt = ds.counts(1) # index based

    expected_counts = [('a', 5), ('b', num_total-5)]
    assert(counts == counts_alt == expected_counts)

    with pytest.warns(UserWarning):
        counts_all = ds.counts()
        # count all if no args are given
        assert(set(counts_all) == set([(x, 1) for x in ds]))


def test_unique():
    ds = load_dummy_data(with_label=True).set_item_names('data', 'label')

    unique_labels = ds.unique('label')
    assert(unique_labels == ['a','b'])

    with pytest.warns(UserWarning):
        unique_items = ds.unique()
        assert(unique_items == list(ds))


def test_shape():
    def get_data(i):
        return i,i

    # no shape yet
    ds = FunctionDataset(get_data)
    assert(ds.shape == _DEFAULT_SHAPE)

    # shape given
    ds._append(1)
    assert(ds.shape == (_DEFAULT_SHAPE, _DEFAULT_SHAPE))

    # numpy data
    ds_np = load_dummy_numpy_data().reshape(DUMMY_NUMPY_DATA_SHAPE_2D)
    assert( ds_np.shape == (DUMMY_NUMPY_DATA_SHAPE_2D,_DEFAULT_SHAPE) )

    # changed to new size
    IMG_SIZE = (6,6)
    ds_img = ds_np.img_resize(IMG_SIZE)
    assert( ds_img.shape == (IMG_SIZE,_DEFAULT_SHAPE) )

    # image with three channels
    DUMMY_NUMPY_DATA_SHAPE_3D
    ds_np3 = ds_np.reshape(DUMMY_NUMPY_DATA_SHAPE_3D)
    assert( ds_np3.shape == (DUMMY_NUMPY_DATA_SHAPE_3D,_DEFAULT_SHAPE) )

    ds_img3 = ds_np3.img_resize(IMG_SIZE)
    assert( ds_img3.shape == ((*IMG_SIZE,3),_DEFAULT_SHAPE) )


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


def test_label():
    ds = load_dummy_data(with_label=True).reorder(0,1,1).set_item_names('data', 'label', 'label_duplicate')

    assert(ds.unique('label') == ['a','b'])

    ds_label = ds.label(1)
    ds_label_alt = ds.label('label')

    # alternative syntaxes
    ds_label = ds.label(1)
    ds_label_alt1 = ds.label("label")
    ds_label_alt2 = ds.transform(label=label())
    ds_label_alt3 = ds.transform(None, label())

    expected = [0, 1]

    for l, l1, l2, l3, e in zip(
        ds_label.unique('label'), 
        ds_label_alt1.unique('label'), 
        ds_label_alt2.unique('label'), 
        ds_label_alt2.unique('label'), 
        expected
    ):
        assert(np.array_equal(l,l1))
        assert(np.array_equal(l,l2))
        assert(np.array_equal(l,l3))
        assert(np.array_equal(l,e)) #type:ignore

    assert(list(ds_label) == [(d, 0 if l == 'a' else 1 ,l2) for d, l, l2 in ds])

    ds_label_userdef = ds.label('label', lambda x: 1 if x == 'a' else 0)

    assert(ds_label_userdef.unique('label') == [1, 0])
    assert(list(ds_label_userdef) == [(d, 1 if l == 'a' else 0 ,l2) for d, l, l2 in ds])

    # error scenarios
    with pytest.raises(TypeError):
        ds.label() # we need to know what to label

    with pytest.raises(ValueError):
        ds.label(42) # wrong key

    with pytest.raises(KeyError):
        ds.label("wrong") # wrong key


def test_one_hot():
    ds = load_dummy_data(with_label=True).reorder(0,1,1).set_item_names('data', 'label', 'label_duplicate')
    assert(ds.unique('label') == ['a','b'])

    # alternative syntaxes
    ds_oh = ds.one_hot(1, encoding_size=2)
    ds_oh_alt1 = ds.one_hot("label", encoding_size=2)
    ds_oh_alt2 = ds.transform(label=one_hot(encoding_size=2))
    ds_oh_alt3 = ds.transform(None, one_hot(encoding_size=2))

    ds_oh_auto = ds.one_hot("label") # automatically compute encoding size

    expected = [np.array([True, False]), np.array([False, True])]

    for l, l1, l2, l3, la, e in zip(
        ds_oh.unique('label'), 
        ds_oh_alt1.unique('label'), 
        ds_oh_alt2.unique('label'), 
        ds_oh_alt2.unique('label'), 
        ds_oh_auto.unique('label'),
        expected
    ):
        assert(np.array_equal(l,l1))
        assert(np.array_equal(l,l2))
        assert(np.array_equal(l,l3))
        assert(np.array_equal(l,la))
        assert(np.array_equal(l,e)) #type:ignore

    for x, l, l2 in ds_oh:
        ind = 0 if l2 == 'a' else 1
        assert(np.array_equal(l, expected[ind])) #type:ignore

    # spiced up
    ds_oh_userdef = ds.one_hot('label', encoding_size=3, mapping_fn=lambda x: 1 if x == 'a' else 0, dtype='int')

    for l, e in zip(ds_oh_userdef.unique('label'), [np.array([1,0,0]), np.array([0,1,0])]):
        assert(np.array_equal(l,e)) #type:ignore

    # error scenarios
    with pytest.raises(TypeError):
        ds.one_hot() # we need some arguments

    with pytest.raises(ValueError):
        ds.one_hot(42, encoding_size=2) # wrong key

    with pytest.raises(ValueError):
        list(ds.one_hot('label', encoding_size=1)) # encoding size too small -- found at runtime

    with pytest.raises(KeyError):
        ds.one_hot("wrong", encoding_size=2) # wrong key

def test_transform():
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

    # error scenarios
    with pytest.warns(UserWarning):
        # no args
        ds.transform()
    
    with pytest.raises(ValueError): 
        # too many transforms given
        ds.transform( reshape(DUMMY_NUMPY_DATA_SHAPE_2D), None, None )


########## Tests relating to numpy data #########################


def test_reshape():
    ds = load_dummy_numpy_data()
    items = [x for x in ds]

    s = ds.shape
    assert(ds.shape == (DUMMY_NUMPY_DATA_SHAPE_1D, _DEFAULT_SHAPE) )
    assert(ds[0][0].shape == DUMMY_NUMPY_DATA_SHAPE_1D)

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
    ds_back = ds_r.reshape(DUMMY_NUMPY_DATA_SHAPE_1D, None)
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

    # TODO test reshape on string data
    ds_str = loaders.load_folder_data(get_test_dataset_path(DATASET_PATHS.FOLDER_DATA))

    with pytest.raises(ValueError):
        # string has no shape 
        ds_str.reshape((1,2))

    with pytest.raises(ValueError):
        # No input
        ds.reshape() 

    with pytest.raises(TypeError):
        # bad input
        ds.reshape('whazzagh') 
    
    with pytest.raises(ValueError):
        # Too many inputs
        ds.reshape(None, None, None) 

    with pytest.raises(ValueError):
        # Dimensions don't match
        ds.reshape((13,13)) 



########## Tests relating to image data #########################

def test_numpy_image_numpy_conversion():
    ds_1d = load_dummy_numpy_data()
    items_1d = [x for x in ds_1d]

    # Warns because no elements where converted
    with pytest.warns(None) as record:
        ds2 = ds_1d.as_image() # skipped all because they could't be converted
        ds3 = ds_1d.as_image(False, False)
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
        ds_1d.as_image(True)

    ds_2d = ds_1d.reshape(DUMMY_NUMPY_DATA_SHAPE_2D)
    items_2d = [x for x in ds_2d]

    # Succesful conversion should happen here
    with pytest.warns(None) as record:
        ds_img = ds_2d.as_image()
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

    # well get a warning if it doens't convert any
    with pytest.warns(UserWarning):
        ds_img.as_numpy(False)


    


def test_string_image_conversion():
    path = get_test_dataset_path(DATASET_PATHS.FOLDER_DATA)
    ds_str = loaders.load_folder_data(path)

    ds_img = ds_str.as_image()
    items_img = [x for x in ds_img]

    for data in items_img:
        data = data[0]
        assert(issubclass(type(data), Image.Image))


def test_image_resize():
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
    with pytest.raises(ValueError):
        ds.img_resize() # No args

    with pytest.raises(ValueError):
        ds.img_resize(NEW_SIZE, NEW_SIZE, NEW_SIZE) # Too many args

    with pytest.raises(AssertionError):
        ds.img_resize((4,4,4)) # Invalid size


########## Framework converters #########################

@pytest.mark.slow
def test_to_tf_simple():
    # prep data
    ds = load_dummy_numpy_data().set_item_names("data", "label").one_hot("label")
    tf_ds = ds.to_tf().batch(2)

    # prep model
    import tensorflow as tf #type:ignore     

    model = tf.keras.Sequential([
        tf.keras.layers.Input(ds.shape[0]),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # model should be able to fit the data
    model.fit(tf_ds, epochs=50)
    preds = model.predict(tf_ds)
    pred_labels = np.argmax(preds, axis=1)

    expected_labels = np.array([v[0] for v in ds.reorder('label').label(0)])
    assert(sum(pred_labels == expected_labels) > len(ds)//2) #type:ignore

