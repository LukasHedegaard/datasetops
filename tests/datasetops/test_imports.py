import pytest


def test_package_import():
    import datasetops as do

    do.image()


def test_from_package_import():
    from datasetops import Dataset
    from datasetops import Loader
    from datasetops import from_folder_data
