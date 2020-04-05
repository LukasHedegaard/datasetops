from datasetops.scaler import Scaler, ScalingInfo, _make_scaler_reshapes
import numpy as np


def test_scaler_reshapes():
    shape = [3, 4, 5]
    d = np.arange(np.prod(shape)).reshape(shape)  # type:ignore

    for axis in range(len(shape)):
        forward, backward = _make_scaler_reshapes(shape, axis)
        assert len(forward(d).shape) == 2  # type:ignore
        assert np.array_equal(d, backward(forward(d)))


def test_minmax():
    SHAPE_1D = (2,)
    SHAPE_3D = (5, 4, 3)
    orig_data = [
        # (string, 1D, 3D, scalar, scalar)
        ("1", np.ones(SHAPE_1D), np.ones(SHAPE_3D), 1, 10),
        ("2", 2 * np.ones(SHAPE_1D), 2 * np.ones(SHAPE_3D), 2, 20),
        ("3", 3 * np.ones(SHAPE_1D), 3 * np.ones(SHAPE_3D), 3, 30),
    ]

    scaler = Scaler(
        [None, ScalingInfo(SHAPE_1D), ScalingInfo(SHAPE_3D, 2), ScalingInfo()]
    )  # omit last items

    for d in orig_data:
        scaler.fit(d)

    # default range (0,1)
    scale_01 = scaler.minmax()  # default range
    result_01 = [scale_01(d) for d in orig_data]
    expected_01 = [
        ("1", 0 * np.ones(SHAPE_1D), 0 * np.ones(SHAPE_3D), 0, 10),
        ("2", 0.5 * np.ones(SHAPE_1D), 0.5 * np.ones(SHAPE_3D), 0.5, 20),
        ("3", 1 * np.ones(SHAPE_1D), 1 * np.ones(SHAPE_3D), 1, 30),
    ]
    for r, e, o in zip(result_01, expected_01, orig_data):
        assert r[0] == e[0] == o[0]
        assert np.array_equal(r[1], e[1])
        assert np.array_equal(r[2], e[2])
        assert not np.array_equal(r[1], o[1])
        assert not np.array_equal(r[2], o[2])
        assert r[3] == e[3] != o[3]
        assert r[4] == e[4] == o[4]

    # custom range (-2,-1)
    scale_m21 = scaler.minmax(feature_range=(-2, -1))
    result_m21 = [scale_m21(d) for d in orig_data]
    expected_m21 = [
        ("1", -2 * np.ones(SHAPE_1D), -2 * np.ones(SHAPE_3D), -2, 10),
        ("2", -1.5 * np.ones(SHAPE_1D), -1.5 * np.ones(SHAPE_3D), -1.5, 20),
        ("3", -1 * np.ones(SHAPE_1D), -1 * np.ones(SHAPE_3D), -1, 30),
    ]
    for r, e, o in zip(result_m21, expected_m21, orig_data):
        assert r[0] == e[0] == o[0]
        assert np.array_equal(r[1], e[1])
        assert np.array_equal(r[2], e[2])
        assert not np.array_equal(r[1], o[1])
        assert not np.array_equal(r[2], o[2])
        assert r[3] == e[3] != o[3]
        assert r[4] == e[4] == o[4]


def test_standardize():
    SHAPE_1D = (2,)
    SHAPE_3D = (5, 4, 3)
    orig_data = [
        # (string, 1D, 3D, scalar, scalar)
        ("1", np.ones(SHAPE_1D), np.ones(SHAPE_3D), 1, 10),
        ("2", 2 * np.ones(SHAPE_1D), 2 * np.ones(SHAPE_3D), 2, 20),
        ("3", 3 * np.ones(SHAPE_1D), 3 * np.ones(SHAPE_3D), 3, 30),
    ]

    scaler = Scaler(
        [None, ScalingInfo(SHAPE_1D), ScalingInfo(SHAPE_3D, 2), ScalingInfo()]
    )  # omit last items

    for d in orig_data:
        scaler.fit(d)

    # default range (0,1)
    scale_std = scaler.standardize()  # default range
    result = [scale_std(d) for d in orig_data]
    v = 1.224744871391589  # np.std([-1.224744871391589, 0, -1.224744871391589]) = 1
    expected = [
        ("1", -v * np.ones(SHAPE_1D), -v * np.ones(SHAPE_3D), -v, 10),
        ("2", 0 * np.ones(SHAPE_1D), 0 * np.ones(SHAPE_3D), 0, 20),
        ("3", v * np.ones(SHAPE_1D), v * np.ones(SHAPE_3D), v, 30),
    ]
    for r, e, o in zip(result, expected, orig_data):
        assert r[0] == e[0] == o[0]
        assert np.array_equal(r[1], e[1])
        assert np.array_equal(r[2], e[2])
        assert not np.array_equal(r[1], o[1])
        assert not np.array_equal(r[2], o[2])
        assert r[3] == e[3] != o[3]
        assert r[4] == e[4] == o[4]
