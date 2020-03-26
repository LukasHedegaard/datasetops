<div align="center">
  <img src="docs/pics/logo.svg"><br>
</div>

-----------------

# Dataset Ops: Fluent dataset operations, compatible with your favorite libraries

![Python package](https://github.com/LukasHedegaard/datasetops/workflows/Python%20package/badge.svg) [![Documentation Status](https://readthedocs.org/projects/datasetops/badge/?version=latest)](https://datasetops.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/LukasHedegaard/datasetops/branch/master/graph/badge.svg)](https://codecov.io/gh/LukasHedegaard/datasetops) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Dataset Ops provides a [fluent interface](https://martinfowler.com/bliki/FluentInterface.html) for _loading, filtering, transforming, splitting,_ and _combining_ datasets. 
Designed specifically with data science and machine learning applications in mind, it integrates seamlessly with [Tensorflow](https://www.tensorflow.org) and [PyTorch](https://pytorch.org).

## Appetizer
```python
import datasetops as do

# prepare your data
train, val, test = (
    do.from_folder_class_data('path/to/data/folder')
    .named("data", "label")
    .image_resize((240, 240))
    .one_hot("label")
    .shuffle(seed=42)
    .split([0.6, 0.2, 0.2])
)

# use with your favorite framework
train_tf = train.to_tensorflow() 
train_pt = train.to_pytorch() 

# or do your own thing
for img, label in train:
    ...
```

## Installation 
Baniry installers available at the [Python package index](https://pypi.org/project/datasetops/)
```bash
pip install datasetops
```


## Why? 
Collecting and preprocessing datasets is tiresome and often takes upwards of 50% of the effort spent in the data science and machine learning lifecycle.
While [Tensorflow](https://www.tensorflow.org/datasets) and [PyTorch](https://www.tensorflow.org/datasets) have some useful datasets utilities available, they are designed specifically with the respective frameworks in mind.
Unsuprisingly, this makes it hard to switch between them, and training-ready dataset definitions are bound to one or the other.
Moreover, they do not aid you in standard scenarios where you want to:
- Sample your dataset non-random ways (e.g with a fixed number of samples per class)
- Center, standardize, normalise you data
- Combine multiple datasets, e.g. for parallel input to a multi-stream network
- Create non-standard data splits

_Dataset Ops_ aims to make these processing steps easier, faster, and more intuitive to perform, while retaining full compatibility to and from the leading libraries. This also means you can grab a dataset from [torchvision datasets](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist) and use it directly with tensorflow:

```python
import do
import torchvision

torch_usps = torchvision.datasets.USPS('../dataset/path', download=True)
tensorflow_usps = do.from_pytorch(torch_usps).to_tensorflow()
```


## Development Status
The library is still under heavy development and the API may be subject to change.

What follows here is a list of implemented and planned features.

### Loaders
- [x] `Loader` (utility class used to define a dataset)
- [x] `from_pytorch` (load from a `torch.utils.data.Dataset`)
- [ ] `from_tensorflow` (load from a `tf.data.Dataset`)
- [x] `from_folder_data` (load flat folder with data)
- [x] `from_folder_class_data` (load nested folder with a folder for each class)
- [x] `from_folder_dataset_class_data` (load nested folder with multiple datasets, each with a nested class folder structure )
- [ ] `from_mat` (load contents of a .mat file as a single dataaset)
- [x] `from_mat_single_mult_data` (load contents of a .mat file as multiple dataasets)
- [ ] `load` (load data from a path, automatically inferring type and structure)

### Converters
- [x] `to_tensorflow` (convert Dataset into tensorflow.data.Dataset)
- [x] `to_pytorch` (convert Dataset into torchvision.Dataset)

### Dataset information
- [x] `shape` (get shape of a dataset item)
- [x] `counts` (compute the counts of each unique item in the dataset by key)
- [x] `unique` (get a list of unique items in the dataset by key)
- [x] `named` (supply names for the item elements)
- [x] `names` (get a list of names for the elements in an item)
- [ ] `stats` (provide an overview of the dataset statistics)
- [ ] `origin` (provide an description of how the dataset was made)

### Sampling and splitting
- [x] `shuffle` (shuffle the items   in a dataset randomly)
- [x] `sample` (sample data at random a dataset)
- [x] `filter` (filter the dataset using a predicate)
- [x] `split` (split a dataset randomly based on fractions)
- [x] `split_filter` (split a dataset into two based on a predicate)
- [x] `allow_unique` (handy predicate used for balanced classwise filtering/sampling)
- [x] `take` (take the first items in dataset)
- [x] `repeat` (repeat the items in a dataset, either itemwise or as a whole)

### Item manipulation
- [x] `reorder` (reorder the elements of the dataset items (e.g. flip label and data order))
- [x] `transform` (transform function which takes other functions and applies them to the dataset items.)
- [x] `categorical` (transforms an element into a categorical integer encoded label)
- [x] `one_hot` (transforms an element into a one-hot encoded label)
- [x] `numpy` (transforms an element into a numpy.ndarray)
- [x] `reshape` (reshapes numpy.ndarray elements)
- [x] `image` (transforms a numpy array or path string into a PIL.Image.Image)
- [x] `image_resize` (resizes PIL.Image.Image elements)
- [ ] `image_crop` (crops PIL.Image.Image elements)
- [ ] `image_rotate` (rotates PIL.Image.Image elements)
- [ ] `image_transform` (transforms PIL.Image.Image elements)
- [ ] `image_brightness` (modify brightness of PIL.Image.Image elements)
- [ ] `image_contrast` (modify contrast of PIL.Image.Image elements)
- [ ] `image_filter` (apply an image filter to PIL.Image.Image elements)
- [ ] `noise` (adds noise to the data)
- [ ] `center` (modify each item according to dataset statistics)
- [ ] `normalize` (modify each item according to dataset statistics)
- [ ] `standardize` (modify each item according to dataset statistics)
- [ ] `whiten` (modify each item according to dataset statistics)
- [ ] `randomly` (apply data transformations with some probability)

### Dataset combinations 
- [x] `concat` (concatenate two datasets, placing the items of one after the other)
- [x] `zip` (zip datasets itemwise, extending the size of each item)
- [x] `cartesian_product` (create a dataset whose items are all combinations of items (zipped) of the originating datasets)



