from pathlib import Path
from glob import glob
from scipy.io import loadmat
from mldatasets.datasets.abstract import ItemGetter
from mldatasets.datasets.dataset import Dataset
from typing import Callable, Any, Union, List, Dict
from PIL import Image
import numpy as np

AnyPath = Union[str, Path]


class DataGetter(ItemGetter):

    def __init__(self, getdata:Callable[[Any], Any], description='data getter'):
        self._getdata = getdata
        self._description = description

    def __getitem__(self, i: int):
        return self._getdata(i)


# def get_file_reader(file_example: AnyPath):
#     p = str(Path(file_path).absolute())

#     try:
#         im = Image.load(filename)
#         im.verify() # throws
#     except expression as identifier:
#         pass

#     file_readers = {
#         'jpg': 
#     }
    
#     try:
#         file_reader = file_readers[extention]
#     except KeyError as e:
#         raise ValueError(f'File reader for "{extention}" not supported. Currently supported file types are {file_readers.keys()}')

#     return file_reader


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
        ds._extend(ids, str(c))

    return ds


def load_folder_dataset_class_data(path: AnyPath) -> List[Dataset]:
    p = Path(path)
    dataset_paths = [x for x in p.glob('[!._]*')]
    return [load_folder_class_data(dsp) for dsp in dataset_paths]


def dataset_from_np_dict(data: Dict[str, np.ndarray], data_keys:List[str], label_key:str=None, name:str=None) -> Dataset:
    all_keys = [*data_keys, label_key]
    shapes_list = [data[k].shape for k in data_keys]
    if label_key:
        shapes_list.append(data[label_key].shape)

    # search for common dimension
    all_shapes = list(set([i for l in shapes_list for i in l]))
    common_shapes = [s for s in all_shapes if all([s in l for l in shapes_list])]

    if len(common_shapes) > 1:
        print("Warning: More than one common shape found for mat dataset. Using the largest dimension as index") # TODO: setup propper warning system

    common_shape = max(common_shapes)

    # reshape data to have the instance as first dimensions
    reshaped_data = {
        k: np.moveaxis(data[k], source=s.index(common_shape), destination=0) #type:ignore
        for k, s in zip(all_keys, shapes_list)
    }

    # prep data getter
    def get_unlabelled_data(idx: int):
        nonlocal reshaped_data
        if len(data_keys) == 1:
            return reshaped_data[data_keys[0]][idx]
        else:
            return tuple([reshaped_data[k][idx] for k in data_keys])
        

    def get_labelled_data(idx: int):
        nonlocal reshaped_data
        return (get_unlabelled_data(idx), reshaped_data[label_key][idx])

    get_data = get_labelled_data if label_key else get_unlabelled_data

    ds = Dataset(downstream_getter=DataGetter(get_data), name=name)

    # populate data getter
    if label_key:
        unique_labels = np.unique(reshaped_data[label_key])

        for lbl in unique_labels:
            lbl_inds = np.extract( #type:ignore
                condition=reshaped_data[label_key].squeeze() == lbl, 
                arr=reshaped_data[label_key].squeeze()
            )
            ds._extend(lbl_inds, lbl)
    else:
        ds._extend(list(range(common_shape)))

    return ds



def load_mat_single_mult_data(path: AnyPath) -> List[Dataset]:
    p = Path(path)
    if p.is_dir():
        file_paths = list(p.glob('[!._]*'))
        assert(len(file_paths)==1)
        p = file_paths[0]
        
    assert(p.suffix == '.mat')

    mat = loadmat(p)

    keys = [ k for k in mat.keys() if not str(k).startswith('_')]


    # group keys according to common suffixes (assuming '_' was used to divide name parts)
    suffixes = [k.split('_')[-1] for k in keys]
    keys_by_suffix = { 
        suf: [str(sk[1]) for sk in zip(suffixes, keys) if sk[0] == suf] 
        for suf in set(suffixes)
    }

    LABEL_INDICATORS = ['y','lbl', 'lbls', 'label', 'labels']

    # create a dataset for each suffix
    datasets: List[Dataset] = []
    for suffix, keys in keys_by_suffix.items():
        label_keys = list(filter(
            lambda k: any([ label_indicator in k.lower() for label_indicator in LABEL_INDICATORS]), 
            keys
        ))
        label_key = label_keys[0] if len(label_keys) > 0 else None
        data_keys = [k for k in keys if k != label_key]

        datasets.append(dataset_from_np_dict(data=mat, data_keys=data_keys, label_key=label_key, name=suffix))

    return datasets
