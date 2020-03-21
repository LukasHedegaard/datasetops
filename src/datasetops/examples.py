import datasetops as do
from datasetops.types import *
import numpy as np


def domain_adaptation_office31(
    source_data_path: AnyPath, target_data_path: AnyPath, seed: int = 0,
) -> Tuple[do.Dataset, do.Dataset, do.Dataset]:

    # load data
    source = do.load_folder_class_data(source_data_path).set_item_names(
        "s_data", "s_label"
    )
    target = do.load_folder_class_data(target_data_path).set_item_names(
        "t_data", "t_label"
    )

    num_source_per_class = 20 if "amazon" in str(source_data_path) else 8
    num_target_per_class = 3

    source_train = source.shuffle(seed).filter(
        s_label=do.allow_unique(num_source_per_class)
    )

    target_test, target_trainval = target.split(
        fractions=[0.3, 0.7], seed=42
    )  # hard-coded seed
    target_train, target_val = target_trainval.shuffle(seed).filter_split(
        t_label=do.allow_unique(num_target_per_class)
    )

    # transform all data to use a one-hot encoding for the label
    source_train, target_train, target_val, target_test = [
        d.one_hot(1)  # automatically infer encoding_size
        for d in [source_train, target_train, target_val, target_test]
    ]

    # Pair up all combinations of the datasets: [(sx1, sy1, tx1, ty1), (sx1, sy1, tx2, ty2) ... ]
    train_cart = do.cartesian_product(source_train, target_train)

    # Limit the train set to have at most an 1:3 ratio of same-label and different-label pairs
    train_same, train_diff = train_cart.reorder(
        "s_data", "t_data", "s_label", "t_label"
    ).filter_split(lambda x: np.array_equal(x[2], x[3]))
    if len(train_diff) > 3 * len(train_same):
        train_diff = train_diff.sample(3 * len(train_same), seed=seed)
    train = do.concat(train_same, train_diff).shuffle(seed)

    # Pair each datapoint with itself, (x,x,y,y)
    val, test = [do.zipped(d, d).reorder(0, 2, 1, 3) for d in [target_val, target_test]]

    # Change the data representation into two tuples (in, out) with an extra label in out
    train, val, test = [
        d.as_image(True, True)
        .img_resize((240, 240), (240, 240))
        .as_numpy(True, True)
        .transform(lambda x: ((x[0], x[1]), (x[2], x[3], x[2] == x[3])))
        for d in [train, val, test]
    ]

    return train, val, test
