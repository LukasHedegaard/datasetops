from typing import Callable, Dict, Sequence, Union, Any, Optional, List, Tuple, Type, TypeVar
from mldatasets.abstract import AbstractDataset
from pathlib import Path

Shape = Sequence[int]
IdIndex = int
Id = int
Ids = List[Id]
Data = Any
IdIndexSet = Dict[Any, List[IdIndex]]
ItemTransformFn = Callable[[Any],Any]
DatasetTransformFn = Callable[[int, AbstractDataset], AbstractDataset] 
DatasetTransformFnCreator = Callable[[Any], DatasetTransformFn]
AnyPath = Union[str, Path]
DataPredicate = Callable[[Any], bool]