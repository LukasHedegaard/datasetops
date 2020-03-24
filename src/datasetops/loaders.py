from pathlib import Path
from datasetops.abstract import ItemGetter
from scipy.io import loadmat
from datasetops.dataset import Dataset
from datasetops.types import *
import numpy as np
import re
import warnings


class Loader(Dataset):
    def __init__(
        self, getdata: Callable[[Any], Any], name: str = None,
    ):
        if not callable(getdata):
            raise TypeError("get_data should be callable")

        class Getter(ItemGetter):
            def __getitem__(self, i: int):
                return getdata(i)

        super().__init__(downstream_getter=Getter(), name=name)

    def _append(self, identifier: Data):
        self._ids.append(identifier)

    def _extend(self, ids: Union[List[Data], np.ndarray]):
        self._ids.extend(list(ids))


def load_folder_data(path: AnyPath) -> Dataset:
    """Load data from a folder with the data structure:

        folder
        |- sample1.jpg
        |- sample2.jpg

    Arguments:
        path {AnyPath} -- path to folder
    
    Returns:
        Dataset -- A dataset of data paths, 
                   e.g. ('nested_folder/class1/sample1.jpg')
    """
    p = Path(path)
    ids = [str(x.relative_to(p)) for x in p.glob("[!._]*")]

    def get_data(i) -> Tuple:
        nonlocal p
        return (str(p / i),)

    ds = Loader(get_data, "Data Getter for folder with structure 'root/data'")
    ds._extend(ids)

    return ds


def load_folder_class_data(path: AnyPath) -> Dataset:
    """Load data from a folder with the data structure:

        nested_folder
        |- class1
            |- sample1.jpg
            |- sample2.jpg
        |- class2
            |- sample3.jpg

    Arguments:
        path {AnyPath} -- path to nested folder
    
    Returns:
        Dataset -- A labelled dataset of data paths and corresponding class labels, 
                   e.g. ('nested_folder/class1/sample1.jpg', 'class1')
    """
    p = Path(path)
    classes = [x for x in p.glob("[!._]*")]

    def get_data(i) -> Tuple:
        nonlocal p
        return (str(p / i), re.split(r"/|\\", i)[0])

    ds = Loader(get_data, "Data Getter for folder with structure 'root/classes/data'")

    for c in classes:
        ids = [str(x.relative_to(p)) for x in c.glob("[!._]*")]
        ds._extend(ids)

    return ds


def load_folder_dataset_class_data(path: AnyPath) -> List[Dataset]:
    """Load data from a folder with the data structure:

        nested_folder
        |- dataset1
            |- class1
                |- sample1.jpg
                |- sample2.jpg
            |- class2
                |- sample3.jpg
        |- dataset2
            |- ...

    Arguments:
        path {AnyPath} -- path to nested folder
    
    Returns:
        List[Dataset] -- A list of labelled datasets, each with data paths and corresponding class labels, 
                         e.g. ('nested_folder/class1/sample1.jpg', 'class1')
    """
    p = Path(path)
    dataset_paths = sorted([x for x in p.glob("[!._]*")])
    return [load_folder_class_data(dsp) for dsp in dataset_paths]


def _dataset_from_np_dict(
    data: Dict[str, np.ndarray],
    data_keys: List[str],
    label_key: str = None,
    name: str = None,
) -> Dataset:
    all_keys = [*data_keys, label_key]
    shapes_list = [data[k].shape for k in data_keys]
    if label_key:
        shapes_list.append(data[label_key].shape)

    # search for common dimension
    all_shapes = list(set([i for l in shapes_list for i in l]))
    common_shapes = [s for s in all_shapes if all([s in l for l in shapes_list])]

    if len(common_shapes) > 1:
        warnings.warn(
            "Warning: More than one common shape found for mat dataset. Using the largest dimension as index"
        )

    common_shape = max(common_shapes)

    # reshape data to have the instance as first dimensions
    reshaped_data = {
        k: np.moveaxis(  # type:ignore
            data[k], source=s.index(common_shape), destination=0
        )
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

    ds = Loader(get_data, name=name)

    # populate data getter
    if label_key:
        unique_labels = np.unique(reshaped_data[label_key])

        for lbl in unique_labels:
            lbl_inds = np.extract(  # type:ignore
                condition=reshaped_data[label_key].squeeze() == lbl,
                arr=reshaped_data[label_key].squeeze(),
            )
            ds._extend(lbl_inds)
    else:
        ds._extend(list(range(common_shape)))

    return ds


def load_mat_single_mult_data(path: AnyPath) -> List[Dataset]:
    """Load data from .mat file consisting of multiple data

       E.g. a .mat file with keys ['X_src', 'Y_src', 'X_tgt', 'Y_tgt']

    Arguments:
        path {AnyPath} -- path to .mat file
    
    Returns:
        List[Dataset] -- A list of datasets, where a dataset was created for each suffix
                         e.g. a dataset with data from the keys ('X_src', 'Y_src') and from ('X_tgt', 'Y_tgt')
    """
    p = Path(path)
    if p.is_dir():
        file_paths = list(p.glob("[!._]*"))
        assert len(file_paths) == 1
        p = file_paths[0]

    assert p.suffix == ".mat"

    mat = loadmat(p)

    keys = [k for k in mat.keys() if not str(k).startswith("_")]

    # group keys according to common suffixes (assuming '_' was used to divide name parts)
    suffixes = [k.split("_")[-1] for k in keys]
    keys_by_suffix = {
        suf: [str(sk[1]) for sk in zip(suffixes, keys) if sk[0] == suf]
        for suf in set(suffixes)
    }

    LABEL_INDICATORS = ["y", "lbl", "lbls", "label", "labels"]

    # create a dataset for each suffix
    datasets: List[Dataset] = []
    for suffix, keys in keys_by_suffix.items():
        label_keys = list(
            filter(
                lambda k: any(
                    [
                        label_indicator in k.lower()
                        for label_indicator in LABEL_INDICATORS
                    ]
                ),
                keys,
            )
        )
        label_key = label_keys[0] if len(label_keys) > 0 else None
        data_keys = [k for k in keys if k != label_key]

        datasets.append(
            _dataset_from_np_dict(
                data=mat, data_keys=data_keys, label_key=label_key, name=suffix
            )
        )

    return sorted(datasets, key=lambda d: d.name)
