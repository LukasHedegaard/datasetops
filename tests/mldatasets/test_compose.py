import pytest
from mldatasets.compose import ZipDataset, InterleaveDataset, CartesianProductDataset
from mldatasets.dataset import cartesian_product, zipped
import os
import sys
sys.path.append(os.path.dirname(__file__))
from testing_utils import ( # type:ignore
    get_test_dataset_path, load_dummy_data, load_dummy_numpy_data,
    DATASET_PATHS, DUMMY_NUMPY_DATA_SHAPE_1D
)

def test_zip():
    ds_pos = load_dummy_data(num_total=10)
    ds_neg = load_dummy_data(num_total=11).transform(lambda x: -x)
    ds_np = load_dummy_numpy_data()
    ds_labelled = load_dummy_data(num_total=10, with_label=True)

    # syntax 1
    zds = zipped(ds_pos, ds_neg)
    assert(len(zds) == min(len(ds_pos), len(ds_neg)))
    assert(zds.shape == (*ds_pos.shape, *ds_neg.shape))

    # syntax 2
    zds_alt = ds_pos.zip(ds_neg)
    assert(len(zds_alt) == len(zds))
    assert(zds_alt.shape == zds.shape)

    # with self
    zds_self = zipped(ds_pos, ds_pos)
    assert(len(zds_self) == len(ds_pos))
    assert(zds_self.shape == (*ds_pos.shape, *ds_pos.shape))

    # mix labelled and unlabelled data
    zds_mix_labelling = ds_neg.zip(ds_labelled)
    assert(len(zds_mix_labelling) == min(len(ds_neg), len(ds_labelled)))
    assert(zds_mix_labelling.shape == ( *ds_neg.shape, *ds_labelled.shape))

    # zip three
    zds_all = zipped(ds_pos, ds_neg, ds_np)
    assert(len(zds) == min(len(ds_pos), len(ds_neg), len(ds_np)))
    assert(zds_all.shape == (*ds_pos.shape, *ds_neg.shape, *ds_np.shape))

    # error scenarios
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            zipped()

    with pytest.warns(UserWarning):
        zipped(ds_pos)

    with pytest.warns(UserWarning):
        ds_pos.zip()


def test_cartesian_product():
    ds_pos = load_dummy_data().take(2).transform(lambda x: x+1)
    ds_10x = ds_pos.transform(lambda x: 10*x)
    ds_100x = ds_pos.transform(lambda x: 100*x)

    # two
    ds_prod2 = cartesian_product(ds_pos, ds_10x)

    ds_prod2_alt = ds_pos.cartesian_product(ds_10x)

    expected2 = [(1, 10), (2, 10), (1, 20), (2, 20)]
    assert(list(ds_prod2) == list(ds_prod2_alt) == expected2)
    assert(len(ds_prod2) == len(ds_prod2_alt) == len(set(expected2)))
    ds_prod2.shape

    # three
    expected3 = [
        (1, 10, 100), 
        (2, 10, 100), 
        (1, 20, 100), 
        (2, 20, 100), 
        (1, 10, 200), 
        (2, 10, 200), 
        (1, 20, 200), 
        (2, 20, 200)
    ]
    ds_prod3 = cartesian_product(ds_pos, ds_10x, ds_100x)
    ds_prod3_alt = ds_pos.cartesian_product(ds_10x, ds_100x)
    assert(list(ds_prod3) == list(ds_prod3_alt) == expected3)
    assert(len(ds_prod3) == len(ds_prod3_alt) == len(set(expected3)))

    # error scenarios
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            cartesian_product()

    with pytest.warns(UserWarning):
        cartesian_product(ds_pos)

    with pytest.warns(UserWarning):
        ds_pos.cartesian_product()


# def test_dataset_interleave():
#     assert(False)




