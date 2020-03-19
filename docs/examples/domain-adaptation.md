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
    source_train =                                      \
        mlds.load(path='../data/Office31/amazon/images')\
        .set_item_names('s_data', 's_label')            \
        .shuffle(seed)                                  \
        .filter(s_label=mlds.allow_unique(20)) # 20 samples form each class

    target_test, target_trainval  =                     \
        mlds.load(path='../data/Office31/dslr/images')  \
        .set_item_names('t_data', 't_label')            \
        .target.split(fractions=[0.3, 0.7], seed)

    target_train, target_val =                          \
        target_trainval                                 \
        .shuffle(seed)                                  \
        .filter_split(t_label=mlds.allow_unique(3)) # 3 samples form each class in first dataset

    # transform all data to use a one-hot encoding for the label
    source_train, target_train, target_val, target_test = [
        d.one_hot(1) # automatically infer encoding_size
        for d in [source_train, target_train, target_val, target_test]
    ]

    # Pair up all combinations of the datasets and reorder to have inputs first and targets last
    train_cart = mlds.cartesian_product(source_train, target_train).reorder(0,2,1,3)
    
    # Limit the train set to have at most an 1:3 ratio of same-label and different-label pairs
    train_same, train_diff = train_cart.filter_split(lambda x: np.array_equal(x[2], x[3]))
    train_diff = train_diff.sample(3*len(train_same), seed=seed)
    train = mlds.concat(train_same, train_diff).shuffle(seed)
    
    # Pair each datapoint with itself, (x,x,y,y)  
    val, test = [
        mlds.zipped(d,d).reorder(0,2,1,3)
        for d in [target_val, target_test]
    ]

    # Change the data representation into two dicts (in, out) with an extra label in out
    train, val, test = [
        d.transform(lambda x: (
            { 'in1':x[0], 'in2':x[1] }, 
            { 'out1':x[2], 'out2':x[3], 'out3':x[2]==x[3] }
        ))
        for d in [train, val, test]
    ]
    mlds.save([train, val, test], processed_dataset_path)

else:
    train, val, test = mlds.load(processed_dataset_path)

# Port to tensorflow
train_tf, val_tf, test_tf = [d.to_tf() for d in [train, val, test]]

```