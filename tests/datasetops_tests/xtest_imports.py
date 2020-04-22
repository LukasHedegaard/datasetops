def test_package_import():
    import datasetops as do

    do.Dataset.from_iterable([1])


def test_from_package_import():
    from datasetops import Dataset

    Dataset.from_iterable([1])
