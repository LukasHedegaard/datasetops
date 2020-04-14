import pytest
import datasetops.loaders as loaders
import datasetops.dataset as mlds
from datasetops.examples import domain_adaptation_office31
import datasetops as do
from testing_utils import (
    get_test_dataset_path,
    DATASET_PATHS,
    RESOURCES_PATH,
)  # type:ignore
from pathlib import Path


@pytest.mark.slow
def test_domain_adaptation():
    p = Path(get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_CLASS_DATA))
    train, val, test = domain_adaptation_office31(
        source_data_path=p / "amazon", target_data_path=p / "dslr", seed=1
    )

    # prepare for tensorflow
    train, val, test = [
        d.to_tensorflow().batch(16).prefetch(2) for d in [train, val, test]
    ]

    # take an item from each and make sure it doesn't raise
    for d in [train, val, test]:
        next(iter(d))


def test_readme_example_1():
    path = (
        Path(get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_CLASS_DATA)) / "amazon"
    )

    train, val, test = (
        do.from_folder_class_data(path)
        .named("data", "label")
        .image_resize((240, 240))
        .one_hot("label")
        .shuffle(seed=42)
        .split([0.6, 0.2, 0.2])
    )


@pytest.mark.slow
def test_readme_example_2():
    import torchvision

    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    torch_usps = torchvision.datasets.USPS(str(RESOURCES_PATH), download=True)
    tensorflow_usps = do.from_pytorch(torch_usps).to_tensorflow()
