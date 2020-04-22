from typing import Callable, Collection, Dict, Any, Sequence, Union, Set, List
from functools import reduce

from datasetops.interfaces import (
    ElemKey,
    ElemNameToIndex,
    ElemIndex,
    ElemPredicate,
    ElemTransform,
    SamplePredicate,
    SampleTransform,
    Sample,
)


# def documents(other_fn):
#     assert callable(other_fn)

#     def inner_decorator(fn):
#         other_fn.__doc__ = fn.__doc__

#         return fn

#     return inner_decorator


def documents(cls):
    def inner_decorator(fn):
        other_fn = getattr(cls, fn.__name__)
        assert callable(other_fn)
        other_fn.__doc__ = fn.__doc__
        return fn

    return inner_decorator


def documented(cls):
    def inner_decorator(fn):
        other_fn = getattr(cls, fn.__name__)
        assert callable(other_fn)
        fn.__doc__ = other_fn.__doc__
        return fn

    return inner_decorator


def monkeypatch(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func

    return decorator


def parameters(func: Callable) -> Dict[str, Any]:
    code = func.__code__
    args_names = code.co_varnames[: code.co_argcount]
    defaults = func.__defaults__ or tuple()
    consts = code.co_consts or tuple()
    arg_vals = consts[: len(args_names) - len(defaults)] + defaults
    return {key: val for key, val in zip(args_names, arg_vals)}


def signature(func: Callable):
    name = func.__name__
    params = ["{}={}".format(param, value) for param, value in parameters(func).items()]
    return "{}({})".format(name, ", ".join(params))


def is_key(x) -> bool:
    return type(x) in {int, str}


def is_collection(x) -> bool:
    return type(x) in {tuple, list, set}


def funcs_from(func_or_funcs: Union[Callable, Sequence[Callable]]) -> List[Callable]:
    if is_collection(func_or_funcs):
        ret: List[Callable] = list(func_or_funcs)  # type:ignore
    else:
        ret: List[Callable] = [func_or_funcs]  # type:ignore

    return ret


def keys_from(key_or_keys: Union[ElemKey, Collection[ElemKey]]) -> List[ElemKey]:
    if is_collection(key_or_keys):
        ret: List[ElemKey] = list(key_or_keys)  # type:ignore
    else:
        ret: List[ElemKey] = [key_or_keys]  # type:ignore

    return ret


def index_from(name2ind: ElemNameToIndex, key: ElemKey) -> ElemIndex:
    if type(key) == int:
        return int(key)

    if str(key) in name2ind:
        return name2ind[str(key)]

    raise KeyError(
        "Unknown key {}. Should be one of {}".format(
            key, list(name2ind.values()) + list(name2ind.keys())  # type:ignore
        )
    )


def inds_from(
    name2ind: ElemNameToIndex, key_or_keys: Union[ElemKey, Collection[ElemKey]]
):
    keys = keys_from(key_or_keys)
    return [(name2ind[str(k)] if type(k) == str else int(k)) for k in keys]


def sample_transform(idx: ElemIndex, elem_transform: ElemTransform) -> SampleTransform:
    def fn(sample: Sample) -> Sample:
        return tuple(elem_transform(elem) if idx == i else elem for i, elem in sample)

    return fn


def collect_keys(
    key_or_list: Union[ElemKey, Sequence[ElemKey]] = None,
    rest_keys: Sequence[ElemKey] = tuple(),
) -> List[ElemKey]:
    """Retreive element indices from a sequence of keys"""
    if key_or_list is None:
        return []

    key_list: List[ElemKey] = (
        list(key_or_list)  # type:ignore
        if type(key_or_list) in {list, tuple}
        else [key_or_list]  # type:ignore
    ) + list(rest_keys)

    return key_list


def sample_predicate(idx: ElemIndex, elem_predicate: ElemPredicate) -> SamplePredicate:
    def fn(sample: Sample) -> bool:
        return elem_predicate(sample[idx])

    return fn


def compose(*functions: Callable):
    def compose2(f, g):
        return lambda x: f(g(x))

    return reduce(compose2, functions, lambda x: x)
