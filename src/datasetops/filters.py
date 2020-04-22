from typing import Union
from .helpers import documents, monkeypatch
from .interfaces import IDataset, ElemPredicate, SamplePredicate, ElemKey
from .dataset import Dataset
from .helpers import index_from, sample_predicate, signature


@documents(IDataset)
@monkeypatch(Dataset)
def filter(
    self: Dataset,
    key_or_samplepredicate: Union[ElemKey, SamplePredicate],
    elem_predicate: ElemPredicate = lambda e: True,
) -> "Dataset":
    """Filter the dataset samples according to a user-defined condition

    Examples:
    .. code-block::
        >>> ds = Dataset.from_iterable([(1,10),(2,20),(3,30)]).named("ones", "tens")
        >>> ds.filter(0, lambda elem: elem%2 == 0) == [(200,20)]
        >>> ds.filter("ones", elem x: elem%2 == 0) == [(200,20)]
        >>> ds.filter(lambda sample: sample[0]%2 == 0) == [(200,20)]

    Arguments:
        key_or_samplepredicate {Union[ElemKey, SamplePredicate]} -- either a key (string or index) or a function checking the whole sample
        elem_predicate {Optional[ElemPredicate]} -- function that checks the element selected by the key

    Returns:
        {Dataset} -- dataset
    """
    if callable(key_or_samplepredicate):
        predicate_func: SamplePredicate = key_or_samplepredicate  # type:ignore
        operation_parameters = {"predicate_func": signature(predicate_func)}
    else:
        elem_idx = index_from(self._elem_name2index, key_or_samplepredicate)
        predicate_func = sample_predicate(elem_idx, elem_predicate)
        operation_parameters = {
            "key": elem_idx,
            "predicate_func": signature(predicate_func),
        }

    new_ids = [i for i, sample in enumerate(self) if predicate_func(sample)]

    return Dataset(
        parent=self,
        sample_ids=new_ids,
        operation_name="filter",
        operation_parameters=operation_parameters,
        stats={},  # reset - can we infer that it should be passed on?
    )


def allow_unique(max_num_duplicates=1) -> ElemPredicate:
    """Predicate used for filtering/sampling a dataset classwise.

    Keyword Arguments:
        max_num_duplicates {int} --
            max number of samples to take that share the same value (default: {1})

    Returns:
        Callable[[Any], bool] -- Predicate function
    """
    mem_counts = {}

    def fn(x):
        nonlocal mem_counts
        h = hash(str(x))
        if h not in mem_counts.keys():
            mem_counts[h] = 1
            return True
        if mem_counts[h] < max_num_duplicates:
            mem_counts[h] += 1
            return True
        return False

    return fn
