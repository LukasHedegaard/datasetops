from datasetops.dataset import Dataset
from datasetops.interfaces import IDataset, ElemKey, SamplePredicate, ElemPredicate
from datasetops.helpers import (
    monkeypatch,
    documents,
    signature,
    sample_predicate,
    index_from,
)
from typing import Union
import random


@documents(IDataset)
@monkeypatch(Dataset)
def sample(self: Dataset, num: int, seed: int = None) -> "Dataset":
    """Samplse data randomly from the dataset.

    Arguments:
        num {int} -- Number of samples

    Keyword Arguments:
        seed {int} -- Random seed (default: {None})

    Returns:
        [Dataset] -- Sampled dataset
    """
    if seed:
        random.seed(seed)

    available = len(self)

    if available < num:
        raise ValueError(
            "More samples requested ({}) than are available ({})".format(num, available)
        )

    new_ids = random.sample(range(available), num)

    return Dataset(
        parent=self,
        sample_ids=new_ids,
        operation_name="sample",
        operation_parameters={"num": num, "seed": seed},
        stats={},  # reset - can we infer that it should be passed on?
    )


@documents(IDataset)
@monkeypatch(Dataset)
def shuffle(self: Dataset, seed: int = None) -> "Dataset":
    """Shuffle the items in a dataset.

    Keyword Arguments:
        seed {[int]} -- Random seed (default: {None})

    Returns:
        [Dataset] -- Dataset with shuffled items
    """
    random.seed(seed)
    new_ids = list(range(len(self)))
    random.shuffle(new_ids)

    return Dataset(
        parent=self,
        sample_ids=new_ids,
        operation_name="shuffle",
        operation_parameters={"seed": seed},
    )


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


@documents(IDataset)
@monkeypatch(Dataset)
def take(self: Dataset, num: int) -> "Dataset":
    """Take the first elements of a dataset.

    Arguments:
        num {int} -- number of elements to take

    Returns:
        Dataset -- A dataset with only the first `num` elements
    """
    if num > len(self):
        raise ValueError("Can't take more elements than are available in dataset")

    new_ids = range(num)
    return Dataset(
        parent=self,
        sample_ids=new_ids,
        operation_name="take",
        operation_parameters={"num": num},
    )


@documents(IDataset)
@monkeypatch(Dataset)
def repeat(self: Dataset, copies=1, mode="whole") -> "Dataset":
    """Repeat the dataset elements.

    Keyword Arguments:
        copies {int} -- Number of copies an element is repeated (default: {1})
        mode {str} -- Repeat 'itemwise' (i.e. [1,1,2,2,3,3]) or as a 'whole'
                        (i.e. [1,2,3,1,2,3]) (default: {'whole'})

    Returns:
        [type] -- [description]
    """
    new_ids = {
        "whole": lambda: [i for _ in range(copies) for i in range(len(self))],
        "itemwise": lambda: [i for i in range(len(self)) for _ in range(copies)],
    }[mode]()

    return Dataset(
        parent=self,
        sample_ids=new_ids,
        operation_name="repeat",
        operation_parameters={"times": copies, "mode": mode},
    )
