from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    # Normalizer
)
from sklearn.preprocessing._data import _handle_zeros_in_scale
import numpy as np
from datasetops.xtypes import Shape, ItemTransformFn
from typing import Iterable, Any, NamedTuple


class ElemStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    axis: int = 0

    def __eq__(self, other):
        return all(
            [
                np.array_equal(self.mean, other.mean),  # type:ignore
                np.array_equal(self.std, other.std),  # type:ignore
                np.array_equal(self.min, other.min),  # type:ignore
                np.array_equal(self.max, other.max),  # type:ignore
                self.axis == other.axis,
            ]
        )


def _make_scaler_reshapes(data_shape: Shape = None, axis: int = 0):
    if not axis:
        axis = 0
    if not data_shape:
        # assuming data is a scalar
        return lambda x: np.array([[x]]), lambda x: x[0][0]

    ishape = list(data_shape)
    ishape[0], ishape[axis] = ishape[axis], ishape[0]

    def reshape_to_scale(d):
        return np.swapaxes(  # type:ignore
            np.swapaxes(d, 0, axis).reshape((data_shape[axis], -1)),  # type:ignore
            0,
            1,
        )

    def reshape_from_scale(d):
        return np.swapaxes(np.swapaxes(d, 0, 1).reshape(ishape), 0, axis)  # type:ignore

    return reshape_to_scale, reshape_from_scale


def _make_scaler(
    transform_fn: ItemTransformFn, shape: Shape, axis=0
) -> ItemTransformFn:
    forward, backward = _make_scaler_reshapes(shape, axis)

    def fn(elem: Any):
        nonlocal transform_fn
        return backward(transform_fn(forward(elem)))

    return fn


def fit(iterable: Iterable, shape: Shape = None, axis=0) -> ElemStats:
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    forward, _ = _make_scaler_reshapes(shape, axis)

    for x in iterable:
        reshaped = forward(x)
        std_scaler.partial_fit(reshaped)
        minmax_scaler.partial_fit(reshaped)

    return ElemStats(
        mean=std_scaler.mean_,
        std=std_scaler.scale_,
        min=minmax_scaler.data_min_,
        max=minmax_scaler.data_max_,
        axis=axis,
    )


def center(shape: Shape, stats: ElemStats) -> ItemTransformFn:
    scaler = StandardScaler(with_std=False)
    scaler.mean_ = stats.mean
    return _make_scaler(scaler.transform, shape, stats.axis)


def standardize(shape: Shape, stats: ElemStats) -> ItemTransformFn:
    scaler = StandardScaler()
    scaler.mean_ = stats.mean
    scaler.scale_ = stats.std
    scaler.var_ = stats.std ** 2
    return _make_scaler(scaler.transform, shape, stats.axis)


def minmax(shape: Shape, stats: ElemStats, feature_range=(0, 1)) -> ItemTransformFn:
    scaler = MinMaxScaler(feature_range)
    # repeating the scikit-learn implementation here, using our known stats
    scaler.data_range_ = stats.max - stats.min
    scaler.scale_ = (
        scaler.feature_range[1] - scaler.feature_range[0]
    ) / _handle_zeros_in_scale(scaler.data_range_)
    scaler.min_ = feature_range[0] - stats.min * scaler.scale_
    return _make_scaler(scaler.transform, shape, stats.axis)


def maxabs(shape: Shape, stats: ElemStats) -> ItemTransformFn:
    scaler = MaxAbsScaler()
    scaler.scale_ = np.max(
        np.array([np.abs(stats.min), stats.max]), axis=0  # type:ignore
    )
    return _make_scaler(scaler.transform, shape, stats.axis)


# def normalize(shape: Shape, axis=0, norm="l2") -> ItemTransformFn:
#     scaler = Normalizer(norm="l2")
#     return _make_scaler(scaler.transform, shape, axis)
