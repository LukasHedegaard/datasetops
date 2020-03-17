import pytest
from mldatasets.compose import ZipDataset, InterleaveDataset, CartesianProductDataset
from mldatasets.dataset import cartesian_product, zipped, concat
import numpy as np
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

    # classwise information is lost
    assert(set(ds_prod2._classwise_id_inds.keys())
        == set(ds_prod2_alt._classwise_id_inds.keys())
        == set(ds_prod3._classwise_id_inds.keys())
        == set(ds_prod3_alt._classwise_id_inds.keys())
        == set([None])
    )

    # error scenarios
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            cartesian_product()

    with pytest.warns(UserWarning):
        cartesian_product(ds_pos)

    with pytest.warns(UserWarning):
        ds_pos.cartesian_product()


def test_concat():
    ds_pos = load_dummy_data().sample_classwise(2).transform(lambda x: x+1)
    ds_neg = ds_pos.transform(lambda x: -x) 
    ds_100x = ds_pos.transform(lambda x: 100*x) 

    # two
    ds_concat = concat(ds_pos, ds_neg)
    ds_concat_alt = ds_pos.concat(ds_neg)
    assert(len(ds_concat) == len(ds_concat_alt) == len(ds_pos) + len(ds_neg))
    assert(list(ds_concat) == list(ds_concat_alt) == list(ds_pos) + list(ds_neg))
    for key in ds_concat._classwise_id_inds.keys():
        assert(ds_concat._classwise_id_inds[key] == ds_pos._classwise_id_inds[key] + list(np.array(ds_neg._classwise_id_inds[key])+len(ds_pos)))

    # three
    ds_concat3 = concat(ds_pos, ds_neg, ds_100x)
    ds_concat3_alt = ds_pos.concat(ds_neg, ds_100x)
    assert(len(ds_concat3) == len(ds_concat3_alt) == len(ds_pos) + len(ds_neg) + len(ds_100x))
    assert(list(ds_concat3) == list(ds_concat3_alt) == list(ds_pos) + list(ds_neg) + list(ds_100x))
    for key in ds_concat3._classwise_id_inds.keys():
        assert( 
            set([ds_concat3[i] for i in ds_concat3._classwise_id_inds[key]]) 
            == 
            set( [ds_pos[i] for i in ds_pos._classwise_id_inds[key]]
                +[ds_neg[i] for i in ds_neg._classwise_id_inds[key]]
                +[ds_100x[i] for i in ds_100x._classwise_id_inds[key]] )
        )

    # error scenarios
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            concat()

    with pytest.warns(UserWarning):
        concat(ds_pos)

    with pytest.warns(UserWarning):
        ds_pos.concat()

    with pytest.warns(UserWarning):
        ds_pos.concat(load_dummy_numpy_data()) # different shapes result in warning
