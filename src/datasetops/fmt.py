@_warn_no_args(skip=1)
def filter(
    self,
    predicates: Optional[
        Union[DataPredicate, Sequence[Optional[DataPredicate]]]
    ] = None,
    **kwpredicates: DataPredicate
):
    """Filter a dataset using a predicate function.

    Keyword Arguments:
        predicates {Union[DataPredicate, Sequence[Optional[DataPredicate]]]} -- either a single or a list of functions taking a single dataset item and returning a bool if a single function is passed, it is applied to the whole item, if a list is passed, the functions are applied itemwise element-wise predicates can also be passed, if item_names have been named.

    Returns:
        [Dataset] -- A filtered Dataset
    """
    pass
