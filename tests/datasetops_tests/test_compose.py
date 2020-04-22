import pytest
from .utils import from_dummy_data, from_dummy_numpy_data


def test_zip():
    ds_pos = from_dummy_data(num_total=10).named("pos")
    ds_neg = from_dummy_data(num_total=11).transform(0, lambda x: -x).named("neg")
    ds_np = from_dummy_numpy_data()
    ds_labelled = from_dummy_data(num_total=10, with_label=True)

    zds = ds_pos.zip(ds_neg)
    assert len(zds) == min(len(ds_pos), len(ds_neg))
    assert zds.shape == (*ds_pos.shape, *ds_neg.shape)
    # item names survive because there were no clashes
    assert zds.names == {"pos": 0, "neg": 1}

    # with self
    zds_self = ds_pos.zip(ds_pos)
    assert len(zds_self) == len(ds_pos)
    assert zds_self.shape == (*ds_pos.shape, *ds_pos.shape)
    # item names are discarded because there are clashes
    assert zds_self.names == {}

    # mix labelled and unlabelled data
    zds_mix_labelling = ds_neg.zip(ds_labelled)
    assert len(zds_mix_labelling) == min(len(ds_neg), len(ds_labelled))
    assert zds_mix_labelling.shape == (*ds_neg.shape, *ds_labelled.shape)  # type: ignore

    # zip three
    zds_all = ds_pos.zip(ds_neg, ds_np)
    assert len(zds) == min(len(ds_pos), len(ds_neg), len(ds_np))
    assert zds_all.shape == (*ds_pos.shape, *ds_neg.shape, *ds_np.shape)  # type: ignore


def test_cartesian_product():
    ds_pos = from_dummy_data().take(2).transform(0, lambda x: x + 1)
    ds_10x = ds_pos.transform(0, lambda x: 10 * x)
    ds_100x = ds_pos.transform(0, lambda x: 100 * x)

    ds_prod2 = ds_pos.cartesian_product(ds_10x)

    expected2 = [(1, 10), (2, 10), (1, 20), (2, 20)]
    assert list(ds_prod2) == expected2
    assert len(ds_prod2) == len(set(expected2))

    expected3 = [
        (1, 10, 100),
        (2, 10, 100),
        (1, 20, 100),
        (2, 20, 100),
        (1, 10, 200),
        (2, 10, 200),
        (1, 20, 200),
        (2, 20, 200),
    ]
    ds_prod3 = ds_pos.cartesian_product(ds_10x, ds_100x)
    assert list(ds_prod3) == expected3
    assert len(ds_prod3) == len(set(expected3))


def test_concat():
    ds_pos = from_dummy_data().named("data").transform("data", lambda x: x + 1)
    ds_neg = ds_pos.transform("data", lambda x: -x)
    ds_100x = ds_pos.transform("data", lambda x: 100 * x)

    # two
    ds_concat = ds_pos.concat(ds_neg)
    assert len(ds_concat) == len(ds_pos) + len(ds_neg)
    assert list(ds_concat) == list(ds_pos) + list(ds_neg)

    # three
    ds_concat3 = ds_pos.concat(ds_neg, ds_100x)
    assert len(ds_concat3) == len(ds_pos) + len(ds_neg) + len(ds_100x)
    assert list(ds_concat3) == list(ds_pos) + list(ds_neg) + list(ds_100x)

    # error scenarios
    with pytest.warns(UserWarning):
        ds_pos.concat(from_dummy_numpy_data())  # different shapes result in warning
