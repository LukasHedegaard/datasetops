import pytest

import datasetops as do
from datasetops.loaders import from_iterable

# see http://doc.pytest.org/en/latest/doctest.html (doctest_namespace fixture)
@pytest.fixture(autouse=True)
def setup_doctest_namespace(doctest_namespace):

    doctest_namespace["do"] = do
    doctest_namespace["from_iterable"] = from_iterable
