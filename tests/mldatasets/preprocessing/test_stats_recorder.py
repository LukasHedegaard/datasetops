import numpy as np
from numpy.testing import assert_allclose

from mldatasets.preprocessing.stats import StatsRecorder

n_dim = 5
n_samples = 10000
X = np.random.rand(n_samples, n_dim)
#X = np.ones((n_samples, n_dim))


def test_mean():

    recorder = StatsRecorder()

    for i in range(n_samples):
        x = X[i, :]
        recorder.update(x)

    assert_allclose(recorder.mean, x.mean(axis=0))


def test_stdDev():

    recorder = StatsRecorder()

    for i in range(n_samples):
        x = X[i, :]
        recorder.update(x)

    assert_allclose(recorder.std, x.std(axis=0))


def test_var():

    recorder = StatsRecorder()

    for i in range(n_samples):
        x = X[i, :]
        recorder.update(x)

    assert_allclose(recorder.mean, x.mean(axis=0))
