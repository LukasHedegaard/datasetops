import pytest

import datasetops as do
from datasetops.loaders import from_iterable, from_recursive_files

# see http://doc.pytest.org/en/latest/doctest.html (doctest_namespace fixture)
@pytest.fixture(autouse=True)
def add_np(doctest_namespace):

    doctest_namespace["do"] = do
    doctest_namespace["from_iterable"] = from_iterable
    doctest_namespace["load_files_recursive"] = from_recursive_files
