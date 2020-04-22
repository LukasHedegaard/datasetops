from typing import Tuple, Sequence, Union
from .interfaces import IDataset, ElemPredicate, SamplePredicate, ElemKey
from .dataset import Dataset
from .helpers import monkeypatch, documents, signature, sample_predicate, index_from
import random


@documents(IDataset)
@monkeypatch(Dataset)
def split(self, fractions: Sequence[float], seed: int = None) -> Tuple[IDataset, ...]:
    """Split dataset into multiple datasets, determined by the fractions.

        A wildcard (-1) may be given at a single position, to fill in the rest.
        If fractions don't add up, the last fraction in the list receives the remainding data.

        Arguments:
            fractions {List[float]} -- a list or tuple of floats i the interval ]0,1[ One of the items may be a -1 wildcard.

        Keyword Arguments:
            seed {int} -- Random seed (default: {None})

        Returns:
            List[Dataset] -- Datasets with the number of samples corresponding to the fractions given
        """
    if seed:
        random.seed(seed)

    # max one entry is '-1'
    assert len(list(filter(lambda x: x == -1, fractions))) <= 1

    # replace wildcard with proper fraction
    if -1 in fractions:
        wildcard = -sum(fractions)
        fractions = [x if x != -1 else wildcard for x in fractions]

    assert sum(fractions) <= 1

    # create shuffled list
    new_ids = list(list(range(len(self))))
    random.shuffle(new_ids)

    # split according to fractions
    split_ids = [[] for _ in fractions]
    last_ind = 0
    for i, f in enumerate(fractions):
        next_last_ind = last_ind + round(f * len(new_ids))
        if i != len(fractions) - 1:
            split_ids[i].extend(new_ids[last_ind:next_last_ind])
            last_ind = next_last_ind
        else:
            split_ids[i].extend(new_ids[last_ind:])

    # create datasets corresponding to each split
    return tuple(
        Dataset(
            parent=self,
            sample_ids=new_ids,
            operation_name="split",
            operation_parameters={"fractions": fractions, "seed": seed, "index": idx,},
        )
        for idx, new_ids in enumerate(split_ids)
    )


@documents(IDataset)
@monkeypatch(Dataset)
def split_filter(
    self,
    key_or_samplepredicate: Union[ElemKey, SamplePredicate],
    elem_predicate: ElemPredicate = lambda e: True,
) -> Tuple[IDataset, IDataset]:
    """Split a dataset using a predicate function.

        Arguments:
            key_or_samplepredicate {Union[ElemKey, SamplePredicate]} -- either a key (string or index) or a function checking the whole sample
            elem_predicate {Optional[ElemPredicate]} -- function that checks the element selected by the key

        Returns:
            Tuple[IDataset, IDataset] -- Two datasets: one that passed the predicate and one that didn't
        """
    if callable(key_or_samplepredicate):
        predicate_func: SamplePredicate = key_or_samplepredicate  # type:ignore
        operation_parameters = {"predicate": signature(predicate_func)}
    else:
        elem_idx = index_from(self._elem_name2index, key_or_samplepredicate)
        predicate_func = sample_predicate(elem_idx, elem_predicate)
        operation_parameters = {
            "key": elem_idx,
            "predicate": signature(predicate_func),
        }

    ack, nack = [], []
    for i in range(len(self)):
        if predicate_func(self[i]):
            ack.append(i)
        else:
            nack.append(i)

    return (
        Dataset(
            parent=self,
            sample_ids=ack,
            operation_name="split_filter",
            operation_parameters={**operation_parameters, "predicate_results": "True"},
        ),
        Dataset(
            parent=self,
            sample_ids=nack,
            operation_name="split_filter",
            operation_parameters={**operation_parameters, "predicate_results": "False"},
        ),
    )


@documents(IDataset)
@monkeypatch(Dataset)
def split_train_test(
    self, fractions=[0.8, 0.2], seed: int = None
) -> Tuple[IDataset, IDataset]:
    """ Performs a sensible train test split.

    Keyword Arguments:
        fractions {List[float]} -- a list or tuple of floats i the interval ]0,1[ One of the items may be a -1 wildcard.  (default: {[0.8, 0.2]})
        seed {int} -- random seed (default: {None})

    Returns:
        Tuple[IDataset, IDataset] -- Train and test datasets
    """
    assert len(fractions) == 2
    return self.split(fractions, seed)


@documents(IDataset)
@monkeypatch(Dataset)
def split_train_val_test(
    self, fractions=[0.65, 0.15, 0.2], seed: int = None
) -> Tuple[IDataset, IDataset, IDataset]:
    """ Performs a sensible train val test split

    Keyword Arguments:
        fractions {List[float]} -- a list or tuple of floats i the interval ]0,1[ One of the items may be a -1 wildcard.  (default: {[0.65, 0.15, 0.2]})
        seed {int} -- random seed (default: {None})

    Returns:
        Tuple[IDataset, IDataset] -- Train and test datasets
    """
    assert len(fractions) == 3
    return self.split(fractions, seed)


# def split_k_fold(self: Dataset, k=5, seed: int = None, return_rest=False):
#     """Splits the dataset into K equally sized folds

#     Keyword Arguments:
#         k {int} -- number of splits (default: {5})
#         seed {int} -- random seed (default: {None})

#     Returns:
#         Tuple[Tuple[IDataset, IDataset], ...] -- A tuple of dataset tuples, where each sub tuple contains the selection and rest
#     """
#     splits = self.split(fractions=(1 / k,) * k, seed=seed)

#     if not return_rest:
#         return splits
#     else:
#         return tuple(
#             [
#                 (
#                     splits[i],
#                     concat([splits[j] for j in set(set(range(len(splits))) - {j})]),
#                 )
#                 for i in len(splits)
#             ]
#         )


# def split_balanced(self, key: Key, num_per_class: int, comparison_fn = lambda a,b: a == b) -> Tuple["IDataset", "IDataset"]: ... # TODO: implement
