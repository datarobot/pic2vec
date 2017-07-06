"""Test the full featurizer class"""
import os
import shutil

import numpy as np
import pytest

from image_featurizer.image_featurizer import ImageFeaturizer

# Constant paths
TEST_CSV_NAME = 'tests/ImageFeaturizer_testing/csv_tests/generated_images_csv_test'
IMAGE_LIST = ['arendt.bmp', 'borges.jpg', 'sappho.png']
CHECK_ARRAY = 'tests/ImageFeaturizer_testing/check_prediction_array_{}.npy'

# Supported models
MODELS = ['squeezenet', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']

# Arguments to load the data into the featurizers
LOAD_DATA_ARGS = {
                  'image_column_header': 'images',
                  'image_path': 'tests/feature_preprocessing_testing/test_images',
                  'new_csv_name': TEST_CSV_NAME
                 }

# Static expected attributes to compare with the featurizer attributes
COMPARE_ARGS = {
                'downsample_size': 0,
                'image_column_header': 'images',
                'automatic_downsample': False,
                'csv_path': TEST_CSV_NAME,
                'image_list': IMAGE_LIST,
                'depth': 1
               }

# Variable attributes to load the featurizer with
LOAD_PARAMS = [
               ('squeezenet', (227, 227), CHECK_ARRAY.format('squeezenet')),
               ('vgg16', (224, 224), CHECK_ARRAY.format('vgg16')),
               ('vgg19', (224, 224), CHECK_ARRAY.format('vgg19')),
               ('resnet50', (224, 224), CHECK_ARRAY.format('resnet50')),
               ('inceptionv3', (299, 299), CHECK_ARRAY.format('inceptionv3')),
               ('xception', (299, 299), CHECK_ARRAY.format('xception'))
              ]

# Remove path to the generated csv if it currently exists
if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/'):
    shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')


def compare_featurizer_class(featurizer,
                             scaled_size,
                             featurized_data,
                             downsample_size,
                             image_column_header,
                             automatic_downsample,
                             csv_path,
                             image_list,
                             depth):
    """Check the necessary assertions for a featurizer image."""
    assert featurizer.scaled_size == scaled_size
    assert np.array_equal(featurizer.featurized_data, featurized_data)
    assert featurizer.downsample_size == downsample_size
    assert featurizer.image_column_header == image_column_header
    assert featurizer.auto_sample == automatic_downsample
    assert featurizer.csv_path == csv_path
    assert featurizer.image_list == image_list
    assert featurizer.depth == depth


def test_featurize_first():
    """Test that the featurizer raises an error if featurize is called before loading data"""
    f = ImageFeaturizer()
    # Raise error if attempting to featurize before loading data
    with pytest.raises(IOError):
        f.featurize()


def testing_featurizer_build():
    """Test that the featurizer saves empty attributes correctly after initializing"""
    f = ImageFeaturizer()
    compare_featurizer_class(f, (0, 0), np.zeros((1)), 0, '', False, '', '', 1)


def test_load_data():
    """Test that the featurizer saves attributes correctly after loading data"""
    f = ImageFeaturizer()
    f.load_data(**LOAD_DATA_ARGS)
    compare_featurizer_class(f, (227, 227), np.zeros((1)), **COMPARE_ARGS)

    # Remove path to the generated csv at end of test
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')


@pytest.mark.parametrize('model,size,array_path', LOAD_PARAMS, ids=MODELS)
def test_featurizer_featurize(model, size, array_path):
    """Test that all of the featurizations and attributes for each model are correct"""
    f = ImageFeaturizer(model=model)
    f.load_and_featurize_data(**LOAD_DATA_ARGS)
    check_array = np.load(array_path)
    compare_featurizer_class(f, size, check_array, **COMPARE_ARGS)

    # Remove path to the generated csv at the end of the test
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')
