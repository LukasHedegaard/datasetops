import pytest
import datasetops.loaders as loaders
import datasetops.dataset as mlds
from datasetops.examples import domain_adaptation_office31
import datasetops as do
import numpy as np
from testing_utils import get_test_dataset_path, DATASET_PATHS  # type:ignore
from pathlib import Path


@pytest.mark.slow
def test_domain_adaptation():
    p = Path(get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_CLASS_DATA))
    train, val, test = domain_adaptation_office31(
        source_data_path=p / "amazon", target_data_path=p / "dslr", seed=1
    )

    # prepare for tensorflow
    train, val, test = [d.to_tf().batch(16).prefetch(2) for d in [train, val, test]]

    # take an item from each and make sure it doesn't raise
    for d in [train, val, test]:
        next(iter(d))


def test_readme_examples():
    path = (
        Path(get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_CLASS_DATA)) / "amazon"
    )

    train, val, test = (
        do.load_folder_class_data(path)
        .set_item_names("data", "label")
        .as_image("data")
        .img_resize((240, 240))
        .as_numpy("data")
        .one_hot("label")
        .shuffle(seed=42)
        .split([0.6, 0.2, 0.2])
    )

