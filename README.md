![Python package](https://github.com/LukasHedegaard/datasetops/workflows/Python%20package/badge.svg) [![Documentation Status](https://readthedocs.org/projects/datasetops/badge/?version=latest)](https://datasetops.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/LukasHedegaard/datasetops/branch/master/graph/badge.svg)](https://codecov.io/gh/LukasHedegaard/datasetops)

# Dataset Ops
Friendly dataset operations for your data science needs

## TL;DR
```python
import datasetops as do

path = '../data/nested_class_folder'

# Prepare your data
train, val, test =                                       \
    do.load_folder_class_data(path)                      \
      .set_item_names('data','label')                    \  
      .as_img('data').resize((240,240)).as_numpy('data') \
      .one_hot('label')                                  \
      .shuffle(seed=42)                                  \
      .split([0.6,0.2,0.3])      

# Do your magic using Tensorflow
train_tf = trian.to_tf() 

# Rule the world with PyTorch
train_pt = trian.to_pytorch() #coming up!

# Do your own thing
for img, label in train:
    ...
```


## Motivation 
Collecting and preprocessing datasets is a tiresome and often underestimated part of the data science and machine learning lifecycle.
While [Tensorflow](https://www.tensorflow.org/datasets) and [PyTorch](https://www.tensorflow.org/datasets) do have some useful datasets utilisites available, they are designed specifically with the respective frameworks in mind.
Unsuprisingly, this makes it hard to switch between frameworks, and port training-ready dataset definitions.

Moreover, they do not aid you in standard scenarios where you want to:
- subsample your dataset, e.g with a fixed number of samples per class
- rescale, center, standardize, normalise you data
- combine multiple datasets, e.g. for parallel input in a multi-stream network
- create non-standard data splits

All of this is usually done by hand. Again and again and again... 


## Idea
In a nutshell, datasets for data science and machine learning are just a collection of samples that are often accompanied by a label.We should be able to read all these formats into a common representation, where most common operations can be performed.Subsequently, we should be able to transform these into the standard formats used in Tensorflow and PyTorch.


## Implementation Status
The library is still under heavy development and the API may be subject to change.

What follows here is a list of implemented and planned features.

### Loaders
- [ ] `load` (load data from a path, automatically inferring type and structure)
- [x] `load_folder_data` (load flat folder with data)
- [x] `load_folder_class_data` (load nested folder with a folder for each class)
- [x] `load_folder_dataset_class_data` (load nested folder with multiple datasets, each with a nested class folder structure )
- [ ] `load_mat` (load contents of a .mat file as a single dataaset)
- [x] `load_mat_single_mult_data` (load contents of a .mat file as multiple dataasets)
- [x] `FunctionDataset` (let users define a dataset)

### Dataset information
- [x] `shape` (get shape of a dataset item)
- [x] `counts` (compute the counts of each unique item in the dataset by key)
- [x] `unique` (get a list of unique items in the dataset by key)
- [x] `item_names` (get a list of names for the elements in an item)
- [x] `set_item_names` (supply names for the item elements)
- [ ] `stats` (provide an overview of the dataset statistics)
- [ ] `origin` (provide an description of how the dataset was made)

### Sampling and splitting
- [x] `shuffle` (shuffle the items   in a dataset randomly)
- [x] `sample` (sample data at random a dataset)
- [x] `split` (split a dataset randomly based on fractions)
- [x] `filter` (filter the dataset using a predicate)
- [x] `filter_split` (split a dataset into two based on a predicate)
- [x] `allow_unique` (handy predicate used for balanced classwise filtering/sampling)
- [x] `take` (take the first items in dataset)
- [x] `repeat` (repeat the items in a dataset, either itemwise or as a whole)

### Item manipulation
- [x] `reorder` (reorder the elements of the dataset items (e.g. flip label and data order))
- [x] `transform` (transform function which takes other functions and applies them to the dataset items.)
- [x] `custom` (function wrapper enabling user-defined function to be used as a transform)
- [x] `label` (transforms an element into a integer encoded categorical label)
- [x] `one_hot` (transforms an element into a one-hot encoded categorical label)
- [x] `as_numpy` (transforms an element into a numpy.ndarray)
- [x] `reshape` (reshapes numpy.ndarray elements)
- [x] `as_image` (transforms a numpy array or path string into a PIL.Image.Image)
- [x] `img_resize` (resizes PIL.Image.Image elements)
- [ ] `center` (modify each item according to dataset statistics)
- [ ] `normalize` (modify each item according to dataset statistics)
- [ ] `standardize` (modify each item according to dataset statistics)
- [ ] `whiten` (modify each item according to dataset statistics)

### Dataset combinations 
- [x] `concat` (concatenate two datasets, placing the items of one after the other)
- [x] `zip` (zip datasets itemwise, extending the size of each item)
- [x] `cartesian_product` (create a dataset whose items are all combinations of items (zipped) of the originating datasets)

### Converters
- [x] `to_tf` (convert Dataset into tensorflow.data.Dataset)
- [ ] `to_pytroch` (convert Dataset into torchvision.Dataset)


