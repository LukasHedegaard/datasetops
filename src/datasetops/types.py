from typing import Callable, Dict, Sequence, Union, Any, List
from datasetops.abstract import AbstractDataset
from pathlib import Path

Shape = Sequence[int]
IdIndex = int
Id = int
Ids = List[Id]
Data = Any
IdIndexSet = Dict[Any, List[IdIndex]]
ItemTransformFn = Callable[[Any], Any]
DatasetTransformFn = Callable[[int, AbstractDataset], AbstractDataset]
DatasetTransformFnCreator = Union[
    Callable[[], DatasetTransformFn], Callable[[Any], DatasetTransformFn]
]
AnyPath = Union[str, Path]
DataPredicate = Callable[[Any], bool]
Key = Union[int, str]

"""Something"""
ItemNames = Dict[str, int]
