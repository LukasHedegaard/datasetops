# ML-Datasets
Flexible datasets for machine learning - under construction ⚙️

## Motivation 🤔
Collecting and preprocessing datasets is a tiresome and often underestimated part of the machine learning lifecycle.
While [Tensorflow](https://www.tensorflow.org/datasets) and [PyTorch](https://www.tensorflow.org/datasets) do have some useful datasets readily available, they are designed specifically with the respective frameworks in mind.
Unsuprisingly, this makes it hard to switch between frameworks, and port training-ready dataset definitions.

Moreover, they do not aid you in common standard scenarios where you want to:
- subsample your dataset, e.g with a fixed number of samples per class
- rescale, center, standardize, normalise you data
- combine multiple datasets, e.g. for parallel input in a multi-stream network
- create non-standard data splits

All of this is usually done by hand. Again and again and again... 🤒

## Idea 💡 
In a nutshell, datasets for machine learning are just a collection of samples that are usually accompanied by a label.
Even though raw datasets come in a few different formats, there are really not that many (unless you have to handle it manually every time), and they usually reduce to the same basic building blocks: samples with labels.

Two examples are the nested folder representation, i.e.
```
nested_folder
|- class1
    |- sample1.jpg
    |- sample2.jpg
|- class2
    |- sample3.jpg
```
and the .mat file, which contains data `X` and labels `Y` with maching indexes.

We should be able to read all these formats into a common representation, where most common operations can be performed.

Subsequently, we should be able to transform these into the standard formats used in Tensorflow and PyTorch.

# API 🎉
Say we have a nested folder as described above, where samples are grouped classwise

Here's an example of how we would like to:
1. Infer a dataset from a raw folder structure automatically
1. Rescale images to 16 x 16 and then normalize them (automatically computing statistics beforehand)
1. Transform target labels into a one-hot encoding
1. Split dataset into train-val-test split with ratios 0.6 : 0.1 : 0.3
1. Store the dataset
1. Create a PyTorch-ready dataset

Then later (just because you can):
1. Load the processed data again
1. Add 1 to all data
1. Transform the one_hot labels to categorical
1. Create a `tensorflow.data.Dataset` from it for use with the Tensorflow framework

```python
import MLDatasets as mlds

raw_data_path = '../data/raw/nested_folder'
processed_path = '../data/processed/my_dataset'

train, val, test = \
    mlds.create(name='MyDataset', path=raw_data_path) \       
    .transform([mlds.rescale(16,16), mlds.normalize(auto=True)], mlds.one_hot()) \
    .split({'train':0.6, 'val': 0.1, 'test': 0.3}) \      
    .save(processed_path) \
    .to_pytorch()

# Do your magic using PyTorch

train, val, test = \
    mlds.load(processed_path) \
    .transform(lambda x: x+1, mlds.to_categorical())
    .to_tf()

# Rule the world with Tensorflow

```

# Building blocks 🧱

## Raw loaders
Dataloader for most of the common dataset formats, including
- [ ] Nested folder
- [ ] Single .mat, .npy and .pkl file with multiple bundles of data
- [ ] Multiple .mat, .npy and .pkl each with one bundle of data
- [ ] A pair of a .mat, .npy and .pkl for input data and .txt with label


## Common format 🤝
Each loader should make the data available in a flexible format that enables easy downstream operations 

Proposed internal format
```python
id_set = {
    'label1': [id1],
    'label2': [id2, id3],
    'unlabelled': [id4, id5, id6]
}

def getitem(id):
    ...
    return (data, target)
```

__Questions❓__
- _Could this format also fit data with a target that is no a class-label?_
    - `id`s would just be stored in the `unlabelled` bucket. The `getitem` definition could still return a target.
- _Could this format also fit data without a label?_
    - Another `getitem` definition would be needed wher no target is returned
    - Can we do better?
- _What if splits are predefined from in the raw data?_
    - __Good question!__


## Data transformation 🌗
- Dataformat modification (e.g. conversion between float and int)
- Centering
- Rescaling 
- Normalisation
- Whitening
- Grouping of temporal data

### Data augmentation ❓
Should/could this be part of the library?


## Split generators 🤘
Split dataset according to standard practices.
This amounts to selecting the subsets from the `id_set`.  

```
no-split
----------------------------------
|   all data in one large pile   |
----------------------------------

train-test
----------------------------------
|    train (%)        | test (%) |
----------------------------------

train-val-test
----------------------------------
| train (%) | val (%) | test (%) |
----------------------------------

train (with subsplits) - test
----------------------------------
|  1 | 2 | 3 | 4 | 5  | test (%) |
----------------------------------
```

### Sampling
- Seed shoud be passed (to ensure reproducibility)
- Class-wise subsampling should be possible


## Dataset composition 🏠
Combine multiple dataset 
- Stack
- Interleave
- Parallel

## Statistics
- Samples per split
- Samples per class

## Export to common formats
All datasets should include
- `to_tf()` function for producing a Tensorflow Dataset (`tensorflow.data.Dataset`)
- `to_pytorch()` function for producing a PyTorch dataset (`torchvision.Dataset`)
    - Theres a special consideration here, because Tensorflow Dataset definitions require both shape and type information