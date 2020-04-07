# KITTI Detection

## Backgorund

KITTI detection dataset is used for 2D/3D object detection based on RGB/Lidar/Camera calibration data.

## Data structure

When downloading the dataset, user can download only interested data and ignore other data. Each data has train and testing folders inside with additional folder that contains name of the data. So there are few ways that user can store this dataset:

1. Just unpack all archives in one folder
```
KITTI
|- image_2
    |- training
        |- image_2
            |- 000000.png
            |- 000001.png
            |- ...
    |- testing
        |- image_2
            |- 000000.png
            |- 000001.png
            |- ...
|- label_2
    |- training
        |- label_2
            |- 000000.txt
            |- 000001.txt
            |- ...
|- image_3
    |- training
        |- image_3
            |- 000000.png
            |- 000001.png
            |- ...
    |- testing
        |- image_3
            |- 000000.png
            |- 000001.png
            |- ...
|- calib
    |- training
        |- calib
            |- 000000.txt
            |- 000001.txt
            |- ...
    |- testing
        |- calib
            |- 000000.txt
            |- 000001.txt
            |- ...
|- velodyne
    |- training
        |- velodyne
            |- 000000.bin
            |- 000001.bin
            |- ...
    |- testing
        |- velodyne
            |- 000000.bin
            |- 000001.bin
            |- ...

```

2. Group them by training/testing
```
    |- training
        |- image_2
            |- 000000.png
            |- 000001.png
            |- ...
        |- image_3
            |- 000000.png
            |- 000001.png
            |- ...
        |- calib
            |- 000000.txt
            |- 000001.txt
            |- ...
        |- velodyne
            |- 000000.bin
            |- 000001.bin
            |- ...
        |- label_2
            |- 000000.txt
            |- 000001.txt
            |- ...            
    |- testing
        |- image_2
            |- 000000.png
            |- 000001.png
            |- ...
        |- image_3
            |- 000000.png
            |- 000001.png
            |- ...
        |- calib
            |- 000000.txt
            |- 000001.txt
            |- ...
        |- velodyne
            |- 000000.bin
            |- 000001.bin
            |- ...
```

Testing split has no label data

## Data types

image_2/image_3: `png images`

calib: `txt files with text and numbers`

velodyne: `bin files with lidar point clouds`

label_2: `txt files with class-name and box coordinates`

## API suggestion

We use second data storage format where data is in training/testing folders 

```python
import mldatasets as mlds

path = './KITTI'

def process_label(text):
    return process_text

def process_calib(text):
    return process_text

with mlds.seed(123):
    
    train, test = mlds.load(path) \
        .transform(["image_2", "image_3"], mlds.normalize) \
        .transform("calib", process_calib) \
        .transform("label_2", process_label) # prints "warning: mlds.transform('label2', process_label) ignored for 'testing' dataset

    train, val = train.split([0.81, -1])
    
    train_tf, test_tf, val_tf = mlds.all(train, test, val) \
        .shuffle() \
        .batch(32, remainder='cycle') \ # if datasize doesn't divide by 32, use first samples to pad (consider leaving batching to tensorflow or pytorch)
        .to_tensorflow()
```