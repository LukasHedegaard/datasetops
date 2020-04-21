from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    Union,
    Iterator,
    Iterable,
    Optional,
    Collection,
    NamedTuple,
    # Literal,
)
import typing
import numpy as np

AnyPath = Union[str, Path]

SampleId = Any
SampleIds = Sequence[SampleId]

Elem = Any
Sample = Tuple[Elem, ...]

ElemShape = Tuple[int, ...]
SampleShape = Tuple[ElemShape, ...]

ElemName = str
ElemIndex = int
ElemKey = Union[ElemIndex, ElemName]
ElemNameToIndex = Dict[ElemName, ElemIndex]

SampleTransformFn = Callable[[Sample], Sample]
ElemTransformFn = Callable[[Elem], Elem]

Predicate = Callable[[Any], bool]

DatasetTransformFn = Callable[[ElemIndex, "IDataset"], "IDataset"]
Argument = Any
DatasetTransformFnCreator = Callable[[Optional[Argument]], DatasetTransformFn]


class ElemStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    axis: int = 0

    def __eq__(self, other: "ElemStats"):
        return all(
            [
                np.array_equal(self.mean, other.mean),  # type:ignore
                np.array_equal(self.std, other.std),  # type:ignore
                np.array_equal(self.min, other.min),  # type:ignore
                np.array_equal(self.max, other.max),  # type:ignore
                self.axis == other.axis,
            ]
        )


def interfacemethod(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(func.__name__)

    return wrapper


class SampleProvider(ABC):
    @abstractmethod
    def __getitem__(self, i: SampleId) -> Sample:
        ...

    @abstractmethod
    def __len__(self):
        ...


class IDataset(SampleProvider, Iterable):
    """Dataset interface"""

    # === Basics. Implemented in `dataset.py` ===

    @typing.overload
    def __getitem__(self, idx: slice) -> List[Sample]:
        ...

    @typing.overload
    def __getitem__(self, idx: int) -> Sample:
        ...

    @interfacemethod
    def __getitem__(self, idx):
        ...

    @interfacemethod
    def __len__(self) -> int:
        """Return the total number of elements in the dataset."""
        ...

    @interfacemethod
    def __iter__(self) -> Iterator[Sample]:
        ...

    @interfacemethod
    @property
    def generator(self,):
        ...

    # === Basic Info: Implemented in `dataset.py` ===

    @interfacemethod
    def named(self, first: Union[str, Sequence[str]], *rest: str) -> "IDataset":
        ...

    @interfacemethod
    @property
    def names(self) -> List[str]:
        ...

    @interfacemethod
    @property
    def shape(self) -> SampleShape:
        ...

    @interfacemethod
    def __str__(self) -> str:
        ...  # TODO: Implement

    @interfacemethod
    def __repr__(self) -> str:
        ...  # TODO: Implement

    @interfacemethod
    def trace(self) -> Any:  # Was transformation graph
        ...  # TODO: Decide return type JSON?

    # def diagram(self, storage_path: AnyPath) -> None: ... # TODO: implement

    # === Data Statistics: Implemented in `stats.py` ===

    @interfacemethod
    def counts(
        self, key_or_keys: Union[ElemKey, Sequence[ElemKey]] = None, *rest_keys: ElemKey
    ) -> List[Tuple[Any, int]]:
        ...

    @interfacemethod
    def unique(
        self, key_or_keys: Union[ElemKey, Sequence[ElemKey]] = None, *rest_keys: ElemKey
    ) -> List[Any]:
        ...

    @interfacemethod
    def statistics(
        self, key_or_keys: Union[ElemKey, Sequence[ElemKey]] = None, axis=None
    ) -> Tuple[ElemStats]:
        ...  # TODO: update impl

    # === Sampling ===

    @interfacemethod
    def sample(self, num: int, seed: int = None) -> "IDataset":
        ...

    # def sample_balanced(self, key: Key, num_per_class: int, comparison_fn = lambda a,b: a == b) -> "IDataset": ... # TODO: implement

    @interfacemethod
    def shuffle(self, seed: int = None) -> "IDataset":
        ...

    @interfacemethod
    def filter(
        self,
        predicates: Optional[Union[Predicate, Sequence[Optional[Predicate]]]],
        **kwpredicates: Predicate,
    ) -> "IDataset":
        ...  # TODO: reconsider API

    @interfacemethod
    def take(self, num: int) -> "IDataset":
        ...

    @interfacemethod
    def repeat(self, copies=1, mode="whole"):  # mode: Literal["whole", "itemwise"]
        ...

    # === Advanced sampling ===

    @interfacemethod
    def resample(
        self,
        func: Callable[[Sequence[Sample]], Sequence[Sample]],
        num_input_samples: int,
        num_output_samples: int,
    ) -> "IDataset":
        ...

    @interfacemethod
    def split_samples(
        self, split_func: Callable[[Sample], Sequence[Sample]], num_output_samples: int,
    ) -> "IDataset":
        ...  # TODO: port from subsample

    @interfacemethod
    def merge_samples(
        self,
        merge_func: Callable[[Sequence[Sample]], Sample],
        num_input_samples: int,
        stride: int = None,
    ) -> "IDataset":
        ...  # TODO: port from supersample

    # === Splitting ===

    @interfacemethod
    def split(
        self, fractions: Sequence[float], seed: int = None
    ) -> Tuple["IDataset", ...]:
        ...

    @interfacemethod
    def split_filter(
        self,
        predicates: Union[Predicate, Sequence[Optional[Predicate]]] = None,
        **kwpredicates: Predicate,
    ) -> Tuple["IDataset", ...]:
        ...

    # def split_train_test(self, ratio=[0.8,0.2], seed:int = None) -> Tuple["IDataset", "IDataset"]: ... # TODO: implement

    # def split_train_val_test(self, ratio=[0.65,0.15,0.2], seed:int = None) -> Tuple["IDataset", "IDataset", "IDataset"]: ... # TODO: implement

    # def split_k_fold(self, k=5, seed:int = None) -> Tuple["IDataset", ...]: ...

    # def split_balanced(self, key: Key, num_per_class: int, comparison_fn = lambda a,b: a == b) -> Tuple["IDataset", "IDataset"]: ... # TODO: implement

    # === Composition ===

    @interfacemethod
    def zip(self, *datasets: Sequence["IDataset"]) -> "IDataset":
        ...

    @interfacemethod
    def concat(self, *datasets: Sequence["IDataset"]) -> "IDataset":
        ...

    @interfacemethod
    def __add__(self, other: "IDataset") -> "IDataset":
        ...  # TODO: should implement concat

    @interfacemethod
    def cartesian_product(self, *datasets: Sequence["IDataset"]) -> "IDataset":
        ...

    # === Free transform ===

    @interfacemethod
    def transform(
        self,
        fns: Optional[
            Union[
                SampleTransformFn,
                Sequence[Union[SampleTransformFn, DatasetTransformFn]],
            ]
        ] = None,
        **kwfns: DatasetTransformFn,
    ) -> "IDataset":
        ...  # TODO: reconsider API

    @interfacemethod
    def reorder(
        self, key_or_keys: Union[ElemKey, Sequence[ElemKey]], *rest_keys: ElemKey
    ) -> "IDataset":
        ...

    # === Label ===

    @interfacemethod
    def categorical(
        self, key: ElemKey, mapping_fn: Callable[[Any], int] = None
    ) -> "IDataset":
        ...  # TODO: consider supporting collection

    @interfacemethod
    def one_hot(
        self,
        key: ElemKey,
        encoding_size: int = None,
        mapping_fn: Callable[[Any], int] = None,
        dtype="bool",
    ) -> "IDataset":
        ...  # TODO: consider supporting collection

    # === Numeric ===

    @interfacemethod
    def numpy(
        self, key_or_keys: Union[ElemKey, Collection[ElemKey]], *rest_keys: ElemKey
    ) -> "IDataset":
        ...  # TODO: update impl

    @interfacemethod
    def reshape(
        self, key_or_keys: Union[ElemKey, Collection[ElemKey]], new_shape: ElemShape,
    ) -> "IDataset":
        ...  # TODO: update impl

    # === Image: Implemented in `image.py` ===

    @interfacemethod
    def image(
        self, key_or_keys: Union[ElemKey, Collection[ElemKey]], *rest_keys: ElemKey
    ) -> "IDataset":
        ...  # TODO: update impl

    @interfacemethod
    def image_resize(
        self, key_or_keys: Union[ElemKey, Collection[ElemKey]], new_shape: ElemShape,
    ) -> "IDataset":
        ...  # TODO: update impl

    # def image_crop(
    #     self,
    #     key_or_keys: Union[ElemKey, Collection[ElemKey]],
    #     upper_left: Tuple[int, int],
    #     bottom_right: Tuple[int, int],
    # ) -> "IDataset":
    #     ...  # TODO: implement

    # def image_rotate(
    #     self,
    #     key_or_keys: Union[ElemKey, Collection[ElemKey]],
    #     amount: float,
    #     unit: Literal["degrees", "radians", "percentage"] = "degrees",
    #     auto_crop=True,
    # ) -> "IDataset":
    #     ...  # TODO: implement

    # def image_transform(self, pil_transform) -> "IDataset": ... # TODO: implement

    # def image_brightness(...) -> "IDataset": ... # TODO: implement

    # def image_contrast(...) -> "IDataset": ... # TODO: implement

    # def image_filter(...) -> "IDataset": ... # TODO: implement

    # === Scalers ===
    @interfacemethod
    def standardize(
        self, key_or_keys: Union[ElemKey, Collection[ElemKey]], axis=0
    ) -> "IDataset":
        ...

    @interfacemethod
    def center(
        self, key_or_keys: Union[ElemKey, Sequence[ElemKey]], axis=0
    ) -> "IDataset":
        ...

    @interfacemethod
    def minmax(
        self,
        key_or_keys: Union[ElemKey, Sequence[ElemKey]],
        axis=0,
        feature_range=(0, 1),
    ) -> "IDataset":
        ...

    @interfacemethod
    def maxabs(
        self, key_or_keys: Union[ElemKey, Sequence[ElemKey]], axis=0
    ) -> "IDataset":
        ...

    # === Loaders ===

    @interfacemethod
    @staticmethod
    def from_iterable(iterable: Iterable, identifier: str = None) -> "IDataset":
        ...

    @interfacemethod
    @staticmethod
    def from_tensorflow(tensorflow_dataset, identifier: str = None) -> "IDataset":
        ...

    @interfacemethod
    @staticmethod
    def from_pytorch(pytorch_dataset, identifier: str = None) -> "IDataset":
        ...

    @interfacemethod
    @staticmethod
    def from_folder(path: AnyPath, identifier: str = None) -> "IDataset":
        ...

    @interfacemethod
    @staticmethod
    def from_label_folders(path: AnyPath, identifier: str = None) -> "IDataset":
        ...  # Previously named from_folder_class_data

    @interfacemethod
    @staticmethod
    def from_parallel_folders(path: AnyPath, identifier: str = None) -> "IDataset":
        ...  # Previously named from_folder_group_data

    @interfacemethod
    @staticmethod
    def from_recursive_files(
        path: AnyPath,
        load_func: Callable[[AnyPath], Sample],
        predicate_func: Callable[[AnyPath], bool] = lambda x: True,
        max_depth: int = 2,
    ) -> "IDataset":
        ...

    @interfacemethod
    @staticmethod
    def from_csv(path: AnyPath) -> "IDataset":
        ...  # TODO: decide if it should be public

    @interfacemethod
    @staticmethod
    def from_mat(path: AnyPath) -> "IDataset":
        ...  # TODO: decide if it should be public

    @interfacemethod
    @staticmethod
    def from_image(path: AnyPath) -> "IDataset":
        ...  # TODO: decide if it should be public

    # def from_url(url) -> "IDataset": ... # TODO: implement

    # def from_googledrive(url) -> "IDataset": ... # TODO: implement

    # def from_saved(path: AnyPath) -> "IDataset": ... # TODO: implement

    # === Storage ===

    @interfacemethod
    def cache(self, path: AnyPath = None) -> "IDataset":
        ...

    @interfacemethod
    def save(self, path: AnyPath = None):
        ...

    @interfacemethod
    def to_tensorflow(self):
        ...

    @interfacemethod
    def to_pytorch(self):
        ...
