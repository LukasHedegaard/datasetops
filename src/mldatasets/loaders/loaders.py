from pathlib import Path
from glob import glob
from mldatasets.datasets.base import ItemGetter
from mldatasets.datasets.dataset import Dataset
from typing import Callable, Any, Union
# FolderDataItemGetter

AnyPath = Union[str, Path]


class DataGetter(ItemGetter):

    def __init__(self, getdata:Callable[[Any], Any], description='data getter'):
        self._getdata = getdata
        self._description = description

    def __getitem__(self, i: int):
        return self._getdata(i)



def load_folder_data(path: AnyPath) -> Dataset:
    p = Path(path)
    ids = [str(x.relative_to(p)) for x in p.glob('[!._]*')]

    def get_data(i):
        nonlocal p
        return str(p/i)

    datagetter = DataGetter(get_data, "Data Getter for folder with structure 'root/data'")

    ds = Dataset(downstream_getter=datagetter)
    ds._extend(ids)

    return ds



def load_folder_class_data(path: AnyPath) -> Dataset:
    p = Path(path)
    classes = [x for x in p.glob('[!._]*')]

    def get_data(i):
        nonlocal p
        return str(p/i)

    datagetter = DataGetter(get_data, "Data Getter for folder with structure 'root/classes/data'")

    ds = Dataset(downstream_getter=datagetter)
    for c in classes:
        ids = [str(x.relative_to(p)) for x in c.glob('[!._]*')]
        ds._extend(ids)

    return ds


def load_folder_dataset_class_data(path: AnyPath) -> Dataset:
    p = Path(path)
    dataset_paths = [x for x in p.glob('[!._]*')]

    return [load_folder_class_data(dsp) for dsp in dataset_paths]
