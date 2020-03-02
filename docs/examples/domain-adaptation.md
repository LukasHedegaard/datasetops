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

processed_dataset_path = f"../data/processed/{source}-{target}-{seed}"

if not mlds.exists(processed_dataset_path):
    source_train = \
        mlds.load(path='../data/Office31/amazon/images') \
        .sample(20, per_class=True, seed=seed)

    target_test, target_rest  = \
        mlds.load(path='../data/Office31/dslr/images') \
        .split([0.3, 0.7], seed=42)

    target_train, target_val = \
        target_trainval.split_sample(3, per_class=True, seed=seed)

    # transform all data to use a one-hot encoding for the label
    source_train, target_train, target_val, target_test = \
        mlds.all(source_train, target_train, target_val, target_test) \
        .transform(None, mlds.one_hot()) 

    # Pair up all combinations of the datasets: ((xs1, xt1), (ys1, yt1)), ((xs1, xt2), (ys1, yt2)), ...
    train_cart = mlds.cartesian_product(source_train, target_train)

    # Limit the train set to have a 1:3 ratio of same-label and different-label pairs
    train_same, train_diff = train_cart.split_by(lambda _, y: y[0] == y[1])
    train_diff_sub = train_diff.sample(3*len(train_same), per_class=False, seed=seed)
    train = mlds.concat(train_same, train_diff_sub).shuffle()
    
    # Pair each datapoint with itself, ((x,x), (y,y))  
    val = mlds.zip(target_val, target_val)
    test = mlds.zip(target_test, target_test)

    # Change the data representations slightly
    train, val, test = mlds.all(train, val, test).transform(
        lambda x: {'i1':x[0], 'i2':x[1]},
        lambda y: {'y1':y[0], 'y2':y[1], 'y3':y[0]==y[1]} 
    )

    datasets = mlds.all(train, val, test)
    datasets.save(processed_dataset_path)

else:
    datasets = mlds.load(processed_dataset_path)

# Port to tensorflow
train, val, test = datasets.to_tensorflow()

```