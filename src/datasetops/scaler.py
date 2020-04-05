from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale
import numpy as np
from datasetops.types import *
from collections import namedtuple
from copy import deepcopy

ScalingInfo = namedtuple("ScalingInfo", "shape axis")
ScalingInfo.__new__.__defaults__ = (None,) * len(ScalingInfo._fields)
ScalerFn = Callable[[Sequence[Any]], Sequence[Any]]


def _make_scaler_reshapes(data_shape: Sequence[int], axis: int = 0):
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


class Scaler:
    def __init__(self, scaling_info: Sequence[Optional[ScalingInfo]]):
        self._scaling_info = scaling_info
        self._std_scalers = []
        self._minmax_scalers = []
        self._forward = []  # data reshapers to scaling
        self._backward = []  # data reshapers from scaling

        for si in scaling_info:
            if si:
                self._std_scalers.append(StandardScaler())
                self._minmax_scalers.append(MinMaxScaler())
                forward, backward = _make_scaler_reshapes(si.shape, si.axis)
                self._forward.append(forward)
                self._backward.append(backward)
            else:
                self._std_scalers.append(None)
                self._minmax_scalers.append(None)
                self._forward.append(None)
                self._backward.append(None)

    def fit(self, item: Sequence[Any]):
        for i, x in enumerate(item):
            if i < len(self._scaling_info) and self._scaling_info[i]:
                reshaped = self._forward[i](x)
                self._std_scalers[i].partial_fit(reshaped)
                self._minmax_scalers[i].partial_fit(reshaped)

    def center(self) -> ScalerFn:
        scalers = deepcopy(self._std_scalers)
        for s in scalers:
            if s:
                s.with_std = False
        return self._make_transform(scalers)

    def standardize(self) -> ScalerFn:
        scalers = deepcopy(self._std_scalers)
        return self._make_transform(scalers)

    def minmax(self, feature_range=(0, 1)) -> ScalerFn:
        scalers = deepcopy(self._minmax_scalers)
        for s in scalers:
            if s and s.feature_range != feature_range:
                # update feature scaling (hacking the scikit-preprocessing implementation)
                s.feature_range = feature_range
                s.scale_ = (
                    s.feature_range[1] - s.feature_range[0]
                ) / _handle_zeros_in_scale(s.data_range_)
                s.min_ = feature_range[0] - s.data_min_ * s.scale_
        return self._make_transform(scalers)

    def maxabs(self) -> ScalerFn:
        scalers = [MaxAbsScaler() if s else None for s in self._scaling_info]
        for s, mms in zip(scalers, self._minmax_scalers):
            if s:
                s.scale_ = np.max(
                    np.array([np.abs(mms.data_min_), mms.data_max_]), axis=0
                )
        return self._make_transform(scalers)

    def _make_transform(self, scalers) -> ScalerFn:
        def fn(item: Sequence[Any]) -> Sequence[Any]:
            nonlocal scalers
            return tuple(
                [
                    self._backward[i](scalers[i].transform(self._forward[i](elem)))
                    if i < len(self._scaling_info) and self._scaling_info[i]
                    else elem
                    for i, elem in enumerate(item)
                ]
            )

        return fn
