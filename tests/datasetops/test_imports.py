import pytest


def test_package_import():
    import datasetops as do

    do.as_image()


def test_from_package_import():
    from datasetops import Dataset
    from datasetops import FunctionDataset
    from datasetops import load_folder_data
