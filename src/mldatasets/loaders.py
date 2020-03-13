from pathlib import Path
from mldatasets.abstract import ItemGetter
from scipy.io import loadmat
from mldatasets.dataset import Dataset
from mldatasets.types import *
from PIL import Image
import numpy as np
import re
import warnings


class FunctionDataset(Dataset):

    def __init__(self,
                 getdata: Callable[[Any], Any],
                 name: str = None,
                 ):

        if(not callable(getdata)):
            raise TypeError(
                f'Invalid getdata argument. The supplied argument is not callable. Value is:\n{getdata}')

        class Getter(ItemGetter):
            def __getitem__(self, i: int):
                return getdata(i)

        super().__init__(downstream_getter=Getter(), name=name)

    def _append(self, identifier: Data, label: Optional[str] = None):

        i_new = len(self._ids)

        self._ids.append(identifier)

        if not label in self._classwise_id_inds:
            self._classwise_id_inds[label] = [i_new]
        else:
            self._classwise_id_inds[label].append(i_new)

    def _extend(self, ids: Union[List[Data], np.ndarray], label: Optional[str] = None):

        # has_duplicate_keys = not set(self._ids).isdisjoint(ids)
        # if(has_duplicate_keys):
        #     raise ValueError(
        #         'Unable to extend the with the supplied ids due to duplicate ids.')

        i_lo = len(self._ids)
        i_hi = i_lo + len(ids)
        l_new = list(range(i_lo, i_hi))

        self._ids.extend(list(ids))

        if not label in self._classwise_id_inds:
            self._classwise_id_inds[label] = l_new
        else:
            self._classwise_id_inds[label].extend(l_new)


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

    def get_data(i) -> Tuple:
        nonlocal p
        return (str(p/i),)

    ds = FunctionDataset(
        get_data, "Data Getter for folder with structure 'root/data'")
    ds._extend(ids)

    return ds


def load_folder_class_data(path: AnyPath) -> Dataset:
    p = Path(path)
    classes = [x for x in p.glob('[!._]*')]

    def get_data(i) -> Tuple:
        nonlocal p
        return (str(p/i), re.split(r'/|\\', i)[0])

    ds = FunctionDataset(
        get_data, "Data Getter for folder with structure 'root/classes/data'")

    for c in classes:
        ids = [str(x.relative_to(p)) for x in c.glob('[!._]*')]
        ds._extend(ids, str(c))

    return ds


def load_folder_dataset_class_data(path: AnyPath) -> List[Dataset]:
    p = Path(path)
    dataset_paths = [x for x in p.glob('[!._]*')]
    return [load_folder_class_data(dsp) for dsp in dataset_paths]


def dataset_from_np_dict(data: Dict[str, np.ndarray], data_keys: List[str], label_key: str = None, name: str = None) -> Dataset:
    all_keys = [*data_keys, label_key]
    shapes_list = [data[k].shape for k in data_keys]
    if label_key:
        shapes_list.append(data[label_key].shape)

    # search for common dimension
    all_shapes = list(set([i for l in shapes_list for i in l]))
    common_shapes = [s for s in all_shapes if all(
        [s in l for l in shapes_list])]

    if len(common_shapes) > 1:
        warnings.warn(
            "Warning: More than one common shape found for mat dataset. Using the largest dimension as index")

    common_shape = max(common_shapes)

    # reshape data to have the instance as first dimensions
    reshaped_data = {
        k: np.moveaxis(data[k], source=s.index(
            common_shape), destination=0)  # type:ignore
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

    ds = FunctionDataset(get_data, name=name)

    # populate data getter
    if label_key:
        unique_labels = np.unique(reshaped_data[label_key])

        for lbl in unique_labels:
            lbl_inds = np.extract(  # type:ignore
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
        assert(len(file_paths) == 1)
        p = file_paths[0]

    assert(p.suffix == '.mat')

    mat = loadmat(p)

    keys = [k for k in mat.keys() if not str(k).startswith('_')]

    # group keys according to common suffixes (assuming '_' was used to divide name parts)
    suffixes = [k.split('_')[-1] for k in keys]
    keys_by_suffix = {
        suf: [str(sk[1]) for sk in zip(suffixes, keys) if sk[0] == suf]
        for suf in set(suffixes)
    }

    LABEL_INDICATORS = ['y', 'lbl', 'lbls', 'label', 'labels']

    # create a dataset for each suffix
    datasets: List[Dataset] = []
    for suffix, keys in keys_by_suffix.items():
        label_keys = list(filter(
            lambda k: any([label_indicator in k.lower()
                           for label_indicator in LABEL_INDICATORS]),
            keys
        ))
        label_key = label_keys[0] if len(label_keys) > 0 else None
        data_keys = [k for k in keys if k != label_key]

        datasets.append(dataset_from_np_dict(
            data=mat, data_keys=data_keys, label_key=label_key, name=suffix))

    return datasets
