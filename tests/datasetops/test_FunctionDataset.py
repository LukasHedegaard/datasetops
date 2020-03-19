import pytest
from datasetops.function_dataset import FunctionDataset


def _get_data(i):
    return i


def test_extend_identicalValues_valid():

    ds = FunctionDataset(_get_data)

    ds._extend(['a'])
    ds._extend(['a'])

    ds._extend(['a', 'a'])

    assert(ds._ids[0] == 'a')
    assert(ds._ids[1] == 'a')
    assert(ds._ids[2] == 'a')
    assert(ds._ids[3] == 'a')


def test_append_identicalIds_throwsValueError():

    ds = FunctionDataset(_get_data)

    ds._append('a')
    ds._append('a')

    assert(ds._ids[0] == 'a')
    assert(ds._ids[1] == 'a')


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
        FunctionDataset(None) #type:ignore

    with pytest.raises(TypeError):
        FunctionDataset("") #type:ignore

    with pytest.raises(TypeError):
        FunctionDataset(1) #type:ignore


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
