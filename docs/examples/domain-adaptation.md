# Domain Adaptation example

## Backgorund

A common dataset for domain adaptation is called Office31.
It consists of images of items commonly found in an office (31 classes) total.
The dataset consists of three sub-dataset, each with a domain of data. The domains are 'amazon' (A), 'dslr' (D), and 'webcam' (W).

Domain adaptation tries to find a common representation between source domain and target domain data with the objective of having good performance on the target dataset.

In the the domain adaptation experiments using a two-stream architecture, you need to subsample each of the datasets (so that only few data are available) and create a pairwise combination of two datasets. 
Moreover, we would like to explore many pair combinations.
The data-pairs are then fed into a two-stream network with the following structure: 
```python
( (x_source, x_target), (y_source,, y_target, labels_equal_flag) )
```
Where the first dictionary is the input data and the second is contains the targets

## Data structure
The data structure of the dataset is a nested tree structure like so:

```
Office31
|- amazon
    |- images
        |- class1
            |- sample1.jpg
            |- sample2.jpg
            |- ...
        |- class2
            |- ...
|- dslr
    |- images
|- webcam
    |- images
```

## Working example
```python
import datasetops as ds
import numpy as np

seed = 13

# load data
source = do.load_folder_class_data("Office31/amazon").named("s_data", "s_label")
target = do.load_folder_class_data("Office31/amazon").named("t_data", "t_label")

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
    d.image_resize((240, 240), (240, 240))
    .transform(lambda x: ((x[0], x[1]), (x[2], x[3], x[2] == x[3])))
    for d in [train, val, test]
]

# Port to Tensorflow
train_tf, val_tf, test_tf = [d.to_tf() for d in [train, val, test]]

```