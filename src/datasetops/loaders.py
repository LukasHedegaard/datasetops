from .interfaces import ISampleProvider, IDataset, SampleId, Sample
from typing import Iterable, Union, Dict, Any, List
from .helpers import monkeypatch, documents
from .dataset import Dataset


class SampleProvider(ISampleProvider):
    def __init__(self, data: List[Sample], identifier: str = None):
        self._data = data
        self._identifier = identifier or str(hash(data))

    def __getitem__(self, i: SampleId) -> Sample:
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def trace(self) -> Union[str, Dict[str, Any]]:
        return self._identifier


@documents(IDataset)
@monkeypatch(Dataset)
def from_iterable(iterable: Iterable, identifier: str = None) -> IDataset:
    """Creates a SampleProvider from the elements of an iterable.

    Arguments:
        iterable {Iterable} -- an iterable containing the samples
        identifier {Optional[str]} -- unique identifier used for tracing

    Returns:
        SampleProvider
    """
    try:
        sample = next(iter(iterable))
    except StopIteration:
        raise ValueError("iterable should contain at least one element")

    if type(sample) is tuple:
        data = list(iterable)
    else:
        data = [(s,) for s in iterable]

    return Dataset(
        parent=SampleProvider(data, identifier or str(hash(tuple(data)))),
        sample_ids=range(len(data)),
        operation_name="from_iterable",
    )
