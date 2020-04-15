from datasetops.scaler import (
    ElemStats,
    _make_scaler_reshapes,
    fit,
)
from datasetops.dataset import center, standardize, minmax, maxabs  # , normalize
import numpy as np
from .testing_utils import multi_shape_dataset
import pytest


def equal(list_a, list_b) -> bool:
    return all(
        [
            np.array_equal(elem_a, elem_b)
            for item_a, item_b in zip(list_a, list_b)
            for elem_a, elem_b in zip(item_a, item_b)
        ]
    )


def test_scaler_reshapes():
    shape = [3, 4, 5]
    d = np.arange(np.prod(shape)).reshape(shape)  # type:ignore

    for axis in range(len(shape)):
        forward, backward = _make_scaler_reshapes(shape, axis)
        assert len(forward(d).shape) == 2  # type:ignore
        assert np.array_equal(d, backward(forward(d)))

    d = 42
    forward, backward = _make_scaler_reshapes()
    assert len(forward(d).shape) == 2  # type:ignore
    assert d == backward(forward(d))


def test_fit():
    SHAPE_1D = (2,)
    SHAPE_3D = (5, 4, 3)
    data0 = [0, 1, 2]
    data1 = [0 * np.ones(SHAPE_1D), 1 * np.ones(SHAPE_1D), 2 * np.ones(SHAPE_1D)]
    data2 = [0 * np.ones(SHAPE_3D), 1 * np.ones(SHAPE_3D), 2 * np.ones(SHAPE_3D)]

    stat0 = fit(data0)
    stat1 = fit(data1, SHAPE_1D, axis=0)
    stat2 = fit(data2, SHAPE_3D, axis=-1)

    assert np.array_equal(stat0.mean, np.array([1]))  # type: ignore
    assert np.array_equal(stat0.std, np.array([0.816496580927726]))  # type: ignore
    assert np.array_equal(stat0.min, np.array([0]))  # type: ignore
    assert np.array_equal(stat0.max, np.array([2]))  # type: ignore
    assert stat0.axis == 0

    assert np.array_equal(stat1.mean, np.array([1, 1]))  # type: ignore
    assert np.array_equal(
        stat1.std, np.array([0.816496580927726, 0.816496580927726])  # type: ignore
    )
    assert np.array_equal(stat1.min, np.array([0, 0]))  # type: ignore
    assert np.array_equal(stat1.max, np.array([2, 2]))  # type: ignore
    assert stat1.axis == 0

    assert np.array_equal(stat2.mean, np.array([1, 1, 1]))  # type: ignore
    assert np.array_equal(
        stat2.std,  # type: ignore
        np.array([0.816496580927726, 0.816496580927726, 0.816496580927726]),
    )
    assert np.array_equal(stat2.min, np.array([0, 0, 0]))  # type: ignore
    assert np.array_equal(stat2.max, np.array([2, 2, 2]))  # type: ignore
    assert stat2.axis == -1


def test_item_stats():
    # (string, 1D, 3D, scalar)
    ds = multi_shape_dataset()

    stat1d_0 = ds.item_stats(1, axis=None)
    stat1d_1 = ds.item_stats(1, axis=0)
    assert (
        stat1d_0
        == stat1d_1
        == ElemStats(
            mean=np.array([1, 1]),
            std=np.array([0.816496580927726, 0.816496580927726]),
            min=np.array([0, 0]),
            max=np.array([2, 2]),
        )
    )

    stat3d_0 = ds.item_stats(2, axis=0)
    stat3d_1 = ds.item_stats(2, axis=-1)
    stat3d_2 = ds.item_stats(2, axis=2)
    assert stat3d_0 != stat3d_1
    assert stat3d_1 == ElemStats(
        mean=np.array([1, 1, 1]),
        std=np.array([0.816496580927726, 0.816496580927726, 0.816496580927726]),
        min=np.array([0, 0, 0]),
        max=np.array([2, 2, 2]),
        axis=-1,
    )
    assert stat3d_2 == ElemStats(
        mean=np.array([1, 1, 1]),
        std=np.array([0.816496580927726, 0.816496580927726, 0.816496580927726]),
        min=np.array([0, 0, 0]),
        max=np.array([2, 2, 2]),
        axis=2,
    )

    stats_scalar_1 = ds.item_stats(3)
    stats_scalar_2 = ds.item_stats(3, axis=0)
    assert (
        stats_scalar_1
        == stats_scalar_2
        == ElemStats(
            mean=np.array([1]),
            std=np.array([0.816496580927726]),
            min=np.array([0]),
            max=np.array([2]),
        )
    )

    with pytest.raises(TypeError):
        ds.item_stats(0)  # string not supported


def test_center():
    SHAPE_1D = (2,)
    SHAPE_3D = (5, 4, 3)
    ds = multi_shape_dataset(SHAPE_1D, SHAPE_3D).named("str", "d1", "d3", "scalar")

    ds1 = ds.transform(d1=center())
    assert equal(
        list(ds1),
        [
            # (string, 1D, 3D, scalar)
            ("0", -1 * np.ones(SHAPE_1D), 0 * np.ones(SHAPE_3D), 0),
            ("1", 0 * np.ones(SHAPE_1D), 1 * np.ones(SHAPE_3D), 1),
            ("2", 1 * np.ones(SHAPE_1D), 2 * np.ones(SHAPE_3D), 2),
        ],
    )
    ds1_1 = ds.center("d1")  # centering again does nothing
    assert equal(list(ds1), list(ds1_1))

    ds2 = ds1.center("d3", axis=-1)
    assert equal(
        list(ds2),
        [
            # (string, 1D, 3D, scalar)
            ("0", -1 * np.ones(SHAPE_1D), -1 * np.ones(SHAPE_3D), 0),
            ("1", 0 * np.ones(SHAPE_1D), 0 * np.ones(SHAPE_3D), 1),
            ("2", 1 * np.ones(SHAPE_1D), 1 * np.ones(SHAPE_3D), 2),
        ],
    )

    # multiple with same axis
    ds1s = ds.center(["d1", "scalar"], axis=0)
    ds1s_alt = ds.center([1, 3], axis=0)  # index syntax
    assert equal(list(ds1s), list(ds1s_alt))
    assert equal(
        list(ds1s),
        [
            # (string, 1D, 3D, scalar)
            ("0", -1 * np.ones(SHAPE_1D), 0 * np.ones(SHAPE_3D), -1),
            ("1", 0 * np.ones(SHAPE_1D), 1 * np.ones(SHAPE_3D), 0),
            ("2", 1 * np.ones(SHAPE_1D), 2 * np.ones(SHAPE_3D), 1),
        ],
    )

    # all at once (also testing scalar)
    ds3 = ds.transform([None, center(), center(axis=-1), center()])
    ds3_alt = ds.transform(d1=center(), d3=center(axis=-1), scalar=center())
    assert equal(list(ds3), list(ds3_alt))
    assert equal(
        list(ds3),
        [
            # (string, 1D, 3D, scalar)
            ("0", -1 * np.ones(SHAPE_1D), -1 * np.ones(SHAPE_3D), -1),
            ("1", 0 * np.ones(SHAPE_1D), 0 * np.ones(SHAPE_3D), 0),
            ("2", 1 * np.ones(SHAPE_1D), 1 * np.ones(SHAPE_3D), 1),
        ],
    )


def test_standardize():
    # we've tested the api versions in the center above - just check basic functionality
    SHAPE_1D = (2,)
    SHAPE_3D = (5, 4, 3)
    ds = multi_shape_dataset(SHAPE_1D, SHAPE_3D).named("str", "d1", "d3", "scalar")

    ds_std = ds.transform(d1=standardize(), d3=standardize(axis=-1)).standardize(
        "scalar"
    )
    s = 1.224744871391589
    assert equal(
        list(ds_std),
        [
            # (string, 1D, 3D, scalar)
            ("0", -s * np.ones(SHAPE_1D), -s * np.ones(SHAPE_3D), -s),
            ("1", 0 * np.ones(SHAPE_1D), 0 * np.ones(SHAPE_3D), 0),
            ("2", s * np.ones(SHAPE_1D), s * np.ones(SHAPE_3D), s),
        ],
    )


# def test_normalize():
#     # we've tested the api versions in the center above
#     # just check basic functionality
#     SHAPE_1D = (2,)
#     SHAPE_3D = (5, 4, 3)
#     ds = multi_shape_dataset(SHAPE_1D, SHAPE_3D).named("str", "d1", "d3", "scalar")

#     ds_norm = ds.transform(
#         d1=normalize(),
#         d3=normalize(axis=-1, norm="l1")
#     ).normalize(
#         "scalar"
#     )
#     v1, v2 = 0.7071067811865475, 0.5773502691896258
#     assert equal(
#         list(ds_norm),
#         [
#             # (string, 1D, 3D, scalar)
#             ("0", 0 * np.ones(SHAPE_1D), 0 * np.ones(SHAPE_3D), 0),
#             ("1", v1 * np.ones(SHAPE_1D), v2 * np.ones(SHAPE_3D), 1),
#             ("2", v1 * np.ones(SHAPE_1D), v2 * np.ones(SHAPE_3D), 1),
#         ],
#     )


def test_minmax():
    # we've tested the api versions in the center above - just check basic functionality
    SHAPE_1D = (2,)
    SHAPE_3D = (5, 4, 3)
    ds = multi_shape_dataset(SHAPE_1D, SHAPE_3D).named("str", "d1", "d3", "scalar")

    ds_mm = ds.transform(d1=minmax(), d3=minmax(axis=-1)).minmax(
        "scalar", feature_range=(-2, 2)
    )
    assert equal(
        list(ds_mm),
        [
            # (string, 1D, 3D, scalar)
            ("0", 0.0 * np.ones(SHAPE_1D), 0.0 * np.ones(SHAPE_3D), -2),
            ("1", 0.5 * np.ones(SHAPE_1D), 0.5 * np.ones(SHAPE_3D), 0),
            ("2", 1.0 * np.ones(SHAPE_1D), 1.0 * np.ones(SHAPE_3D), 2),
        ],
    )


def test_maxabs():
    # we've tested the api versions in the center above - just check basic functionality
    SHAPE_1D = (2,)
    SHAPE_3D = (5, 4, 3)
    ds = multi_shape_dataset(SHAPE_1D, SHAPE_3D).named("str", "d1", "d3", "scalar")

    ds_ma = ds.transform(d1=maxabs(), d3=maxabs(axis=-1)).maxabs("scalar")
    assert equal(
        list(ds_ma),
        [
            # (string, 1D, 3D, scalar)
            ("0", 0.0 * np.ones(SHAPE_1D), 0.0 * np.ones(SHAPE_3D), 0.0),
            ("1", 0.5 * np.ones(SHAPE_1D), 0.5 * np.ones(SHAPE_3D), 0.5),
            ("2", 1.0 * np.ones(SHAPE_1D), 1.0 * np.ones(SHAPE_3D), 1.0),
        ],
    )
