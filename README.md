Pic2Vec
================

Featurize images using a small, contained pre-trained deep learning network


* Free software: BSD license


Features
--------

This is the prototype for image features engineering.  Currently, only
Python 2.7 is supported.

``pic2vec`` is a python package that performs automated feature extraction
for image data. It supports training models via the
DataRobot modeling API, as well as feature engineering on new image data.

## Input Specification

### Data Format

``pic2vec`` works on image data represented as either:
1. A directory of image files.
2. As URL pointers contained in a CSV.
3. Or as a directory of images with a CSV containing pointers to the image files.

If no CSV is provided with the directory, it automatically generates a CSV to store the features with the appropriate images.

Each row of the CSV represents a different image, and image rows can also have columns containing other data about the images as well. Each image's featurized representation will be appended as a series of new columns at the end of the appropriate image row.


### Constraints Specification
The goal of this project was to make the featurizer as easy to use and hard to break as possible. If working properly, it should be resistant to badly-formatted data, such as missing rows or columns in the csv, image mismatches between a CSV and an image directory, and invalid image formats.

However, for the featurizer to function optimally, it prefers certain constraints:
* The CSV should have no missing columns or rows, and there should be full overlap between images in the CSV and the image directory

* If checking predictions on a separate test set (such as on Kaggle), the filesystem needs to sort filepaths consistently with the sorting of the test set labels. The order in the CSV (whether generated automatically or passed in) will be considered the canonical order for the feature vectors.

The featurizer can only process .png, .jpeg, or .bmp image files. Any other images will be left out of the featurization by being represented by zero vectors in the image batch.

## Quick Start

The following Python code shows a typical usage of `pic2vec`:

```python
import pandas as pd
from pic2vec import ImageFeaturizer

image_column_name = 'images'
my_csv = 'path/to/data.csv'
my_image_directory = 'path/to/image/directory/'

my_featurizer = ImageFeaturizer(depth=2)

my_featurizer.load_data(image_column_name, csv_path = my_csv, image_path = my_image_directory)

my_featurizer.featurize()
```

## Examples

To get started, see the following example:

1. [Cats vs. Dogs](examples/cats_v_dogs.ipynb): Dataset from combined directory + CSV

Examples coming soon:
1. [Hotdogs](examples/hotdogs): Dataset from unsupervised directory only, with PCA visualization
1. [URLs](examples/): Dataset from CSV with URLs and no image directory


## Installation

See the [Installation Guide](docs/guides/installation.md) for details.

### Installing Keras/Tensorflow
If you run into trouble installing Keras or Tensorflow as a dependency, read the [Keras installation guide](https://keras.io/#installation) and  [Tensorflow installation guide](https://www.tensorflow.org/install/) for details about installing Keras/Tensorflow on your machine.


## Using Featurizer Output With DataRobot
``pic2vec`` generates a CSV that is ready to be dropped directly into the DataRobot application, if the data has been labelled with a variable that can be considered a target in the CSV. The image features are each treated like regular columns containing data.


### Running tests

To run the unit tests with ``pytest``, run

```
py.test tests
```



Credits
---------

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
