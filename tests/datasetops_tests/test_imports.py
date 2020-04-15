def test_package_import():
    import datasetops as do

    do.image()


def test_from_package_import():
    from datasetops import Dataset  # noqa: F401
    from datasetops import Loader  # noqa: F401
    from datasetops import from_folder_data  # noqa: F401
