from .interfaces import ISampleProvider, IDataset, SampleId, Sample
from typing import Iterable, Union, Dict, Any
from itertools import islice
from .helpers import monkeypatch, documents
from .new_dataset import Dataset


class SampleProvider(ISampleProvider):
    def __init__(self, data: list[Sample], identifier: str = None):
        self._data = data
        self._identifier = identifier or str(hash(data))

    def __getitem__(self, i: SampleId) -> Sample:
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def trace(self) -> Union[str, Dict[str, Any]]:
        return self._identifier


@monkeypatch(Dataset)
@documents(IDataset.from_iterable)
def from_iterable(iterable: Iterable, identifier: str = None) -> SampleProvider:
    """Creates a SampleProvider from the elements of an iterable.

    Arguments:
        iterable {Iterable} -- an iterable containing the samples
        identifier {Optional[str]} -- unique identifier used for tracing

    Returns:
        SampleProvider
    """
    sample = islice(iterable, 1)

    if type(sample) is tuple:
        data = list(iterable)
    else:
        data = [tuple(s) for s in iterable]

    return SampleProvider(data, identifier or "iterable: {}".format(hash(data)))
