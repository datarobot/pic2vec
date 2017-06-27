Image Featurizer
================

Featurize images using a small, contained pre-trained deep learning network


* Free software: BSD license
* Documentation: https://image-featurizer.readthedocs.io.


Features
--------

This is the prototype for image features engineering.  Currently, only
Python 2.7 is supported.

``image_featurizer`` is a python package that performs automated feature extraction
for image data. It supports training models via the
DataRobot modeling API, as well as feature engineering on new image data.

## Input Specification

### Data Format

``image_featurizer`` works on image data represented either as a directory of image files, as URL pointers contained in a CSV, or as a directory of images with a CSV containing pointers to the image files. If no CSV is provided with the directory, it automatically generates a CSV to store the features with the appropriate images. Each
row of the CSV is associated with a different image, and they can be combined with external data as well. Each image's featurized representation will be appended to new columns at the end of the appropriate row.


### Constraints Specification
The goal of this project was to make the featurizer as easy to use and hard to break as possible. If working properly, it should be resistant to badly-formatted data, such as missing rows or columns in the csv, image mismatches between a CSV and an image directory, and invalid image formats.

However, for the featurizer to function optimally, it prefers certain constraints:
* The CSV should have no missing columns or rows, and there should be full overlap between images in the CSV and the image directory

* If checking predictions on a separate test set (such as on Kaggle), the filesystem needs to sort filepaths consistently with the sorting of the test set labels. The order in the CSV (whether generated automatically or passed in) will be considered the canonical order for the feature vectors.

The featurizer can only process .png, .jpeg, or .bmp image files– any other images will be left out of the featurization.

## Quick Start

The following Python code shows a typical usage of `image_featurizer`:

```python
import pandas as pd
from image_featurizer.image_featurizer import ImageFeaturizer

image_column_name = 'images'
my_csv = 'path/to/data.csv'
my_image_directory = 'path/to/image/directory/'

my_featurizer = ImageFeaturizer(depth=2)

my_featurizer.load_data(image_column_name, csv_path = my_csv,
                        image_path =  my_image_directory)

my_featurizer.featurize()
```

## Examples

To get started, see the following examples:

1. [Cats vs. Dogs](examples/cats_vs_dogs): Dataset from combined directory + CSV
1. [Hotdogs](examples/hotdogs): Dataset from unsupervised directory only, with PCA visualization
1. [URLs](examples/refresh-data): Dataset from CSV with URLs and no image directory

## Documentation

To generate the documentation, run the following command in the terminal:

```
cd docs
make html
```

The generated documentation can be found in the directory ``docs/_build/html``.

Some useful information:

* [List of Parameters](docs/markdowns/parameters.md)

## Installation

See the [Installation Guide](docs/guides/installation.rst) for details.


## Using Output With DataRobot
``image_featurizer`` generates a CSV that is ready to be dropped directly into the DataRobot application, if the data has been labelled with a variable that can be considered a target in the CSV. The image features are each treated like regular columns containing data.

## Development

### Setup development environment

To setup your development environment for the image_featurizer package, follow the steps within the [Installation Guide](docs/guides/install.rst)
for installing from source, but run the following to install the additional development requirements:

```
make req-dev
```

### Running tests

To run the unit tests with ``pytest``, run

```
py.test tests
```





Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
