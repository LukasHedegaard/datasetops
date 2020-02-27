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
( {'x1': x_source, 'x2':x_target}, {'y1': y_source, 'y2': y_target, 'y3':labels_equal_flag} )
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

## API suggestion
```python
import mldatasets as mlds

seed = 1
source = 'amazon'
target = 'dslr'

my_dataset_name = f"{source}-{target}-{seed}"

if mlds.exists(my_dataset_name):
    train, 
else:
    source_train = \
        mlds.create(name='amazon', path='../data/Office31/amazon/images') \
        .subsample_classwise(per_class=20, seed=seed)

    target_trainval, target_test = \
        mlds.create(name='dslr', path='../data/Office31/dslr/images') \
        .split([0.7, 0.3])

    target_train, target_val = \
        target_trainval
        .subsample_classwise(per_class=3, seed=seed, return_rest=True)

    # transform all data to use a one-hot encoding for the label
    source_train, target_train, target_val, target_test = [
        ds.transform(None, mlds.one_hot()) 
        for ds in (source_train, target_train, target_val, target_test]
    ]

    # Combine data in a pairwise manner 
    train = mlds.combine_parallel(
        datasets=(source_train, target_train), 
        cartesian_product=True # pair up all combinations of the datasets
    )
    val = mlds.combine_parallel((target_val, target_val))
    test = mlds.combine_parallel((target_test, target_test))

    # Change the data representation a bit
    train, val, test = [
        ds.transform(
            lambda x: {'i1':x[0], 'i2':x[1]},
            lambda y: {'y1':y[0], 'y2':y[1], 'y3':y[0]==y[1]} 
        )
        for ds in [train, val, test]
    ]

    train = train.shuffle()

    mlds.save([train, val, test], name=my_dataset_name)

# Port to tensorflow
train, val, test = [ds.to_tf() for ds in [train, val, test]]

```