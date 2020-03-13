import pytest
from mldatasets.compose import ZipDataset, InterleaveDataset, CartesianProductDataset
from mldatasets.dataset import zipped
import os
import sys
sys.path.append(os.path.dirname(__file__))
from testing_utils import ( # type:ignore
    get_test_dataset_path, load_dummy_data, load_dummy_numpy_data,
    DATASET_PATHS, DUMMY_NUMPY_DATA_SHAPE_1D
)

def test_dataset_zip():
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


# def test_dataset_interleave():
#     assert(False)


# def test_dataset_cartesian_product():
#     assert(False)

