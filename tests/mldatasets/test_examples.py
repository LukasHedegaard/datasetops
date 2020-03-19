import pytest
import mldatasets.loaders as loaders
import mldatasets.dataset as mlds
import numpy as np
from testing_utils import get_test_dataset_path, DATASET_PATHS # type:ignore
from mldatasets.examples import domain_adaptation_office31
from pathlib import Path

@pytest.mark.slow
def test_domain_adaptation():
    p = Path(get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_CLASS_DATA))
    train, val, test = domain_adaptation_office31(source_data_path=p/'amazon', target_data_path=p/'dslr', seed=1)

    # prepare for tensorflow
    train, val, test = [ d.to_tf().batch(16).prefetch(2) for d in [train, val, test] ]

    # take an item from each and make sure it doesn't raise
    for d in [train, val, test]:
        next(iter(d))

    
