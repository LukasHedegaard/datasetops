import pytest

from mldatasets.loaders import FunctionDataset


def _get_data(i):
    return i


def test_extend_identicalIds_throwsValueError():

    ds = FunctionDataset(_get_data)

    with pytest.raises(ValueError):
        ds._extend([1])
        ds._extend([1])

    with pytest.raises(ValueError):
        ds._extend([1, 1])


def test_append_identicalIds_throwsValueError():

    ds = FunctionDataset(_get_data)

    with pytest.raises(ValueError):
        ds._append([1])
        ds._append([1])

    with pytest.raises(ValueError):
        ds._append([1, 1])


def test_extend_nonContingousIds_valid():

    ds = FunctionDataset(_get_data)

    ds._extend([1])
    ds._extend([3])


def test_extend_acceptsTuples():
    ds = FunctionDataset(_get_data)
    ds._extend((1, 2))


def test_extend_acceptsSets():

    ds = FunctionDataset(_get_data)
    ds._extend({1, 2})


def test_extend_mixingIdsTypes_valid():

    ds = FunctionDataset(_get_data)

    ds._extend([1])
    ds._extend(['1'])

    assert(len(ds) == 2)
    assert(1 in ds._ids)
    assert('1' in ds._ids)


def test_ctor_nonCallableGetter_throwsTypeError():

    with pytest.raises(TypeError):
        FunctionDataset(None)

    with pytest.raises(TypeError):
        FunctionDataset("")

    with pytest.raises(TypeError):
        FunctionDataset(1)


def test_len_extendCalled_lenMatchesNumberOfElements():

    ds = FunctionDataset(_get_data)
    ds._extend([1])
    ds._extend([2])

    assert(len(ds) == 2)


def test_len_appendCalled_lenMatchesNumberOfElements():

    ds = FunctionDataset(_get_data)
    ds._append([1])
    ds._append([1])

    assert(len(ds) == 2)
