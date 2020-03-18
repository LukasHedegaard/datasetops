import pytest
import mldatasets.loaders as loaders
import mldatasets.dataset as mlds
import numpy as np
from testing_utils import get_test_dataset_path, DATASET_PATHS # type:ignore

@pytest.mark.slow
def test_domain_adaptation():
    #### Prepare data ####
    seed = 42
    datasets = loaders.load_folder_dataset_class_data(get_test_dataset_path(DATASET_PATHS.FOLDER_DATASET_CLASS_DATA))
    source = datasets[0].set_item_names('s_data', 's_label')
    target = datasets[1].set_item_names('t_data', 't_label')

    source_train = source.shuffle(seed).filter(s_label=mlds.allow_unique(20))

    target_test, target_trainval  = target.split(fractions=[0.3, 0.7], seed=seed)
    target_train, target_val = target_trainval.shuffle(seed).filter_split(t_label=mlds.allow_unique(3))

    # transform all data to use a one-hot encoding for the label
    source_train, target_train, target_val, target_test = [
        d.one_hot(1) # automatically infer encoding_size
        for d in [source_train, target_train, target_val, target_test]
    ]

    # Pair up all combinations of the datasets: [(sx1, sy1, tx1, ty1), (sx1, sy1, tx2, ty2) ... ]
    train_cart = mlds.cartesian_product(source_train, target_train)
    
    # Limit the train set to have at most an 1:3 ratio of same-label and different-label pairs
    train_same, train_diff = train_cart.reorder('s_data','t_data','s_label','t_label').filter_split(lambda x: np.array_equal(x[2], x[3]))
    if len(train_diff) > 3*len(train_same):
        train_diff = train_diff.sample(3*len(train_same), seed=seed)
    train = mlds.concat(train_same, train_diff).shuffle(seed)
    
    # Pair each datapoint with itself, (x,x,y,y)  
    val, test = [
        mlds.zipped(d,d).reorder(0,2,1,3)
        for d in [target_val, target_test]
    ]

    # Change the data representation into two dicts (in, out) with an extra label in out
    train, val, test = [
        d.transform(lambda x: ({'in1':x[0], 'in2':x[1]}, {'out1':x[2], 'out2':x[3], 'out3':x[2]==x[3]}))
        for d in [train, val, test]
    ]

    # prepare for tensorflow
    train_tf = train.to_tf().repeat().batch(16).prefetch(10)
    val_tf, test_tf = [ d.to_tf().batch(16) for d in [val, test] ]

    # take an item from each and make sure it doesn't raise
    for d in [train_tf, val_tf, test_tf]:
        next(iter(d))


    
    
