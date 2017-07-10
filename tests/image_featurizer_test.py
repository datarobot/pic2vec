"""Test the full featurizer class"""
import filecmp
import os
import pytest
import shutil

import numpy as np

from pic2vec import ImageFeaturizer

# Constant paths
TEST_CSV_NAME = 'tests/ImageFeaturizer_testing/csv_tests/generated_images_csv_test'
IMAGE_LIST = [['arendt.bmp', 'borges.jpg', 'sappho.png']]
CHECK_ARRAY = 'tests/ImageFeaturizer_testing/array_tests/check_prediction_array_{}.npy'
CHECK_CSV = 'tests/ImageFeaturizer_testing/csv_checking/{}_check_csv'

CSV_NAME_MULT = 'tests/ImageFeaturizer_testing/csv_checking/mult_check_csv'
IMAGE_LIST_MULT = [['arendt.bmp', 'sappho.png', ''], ['borges.jpg', '', '']]
CHECK_ARRAY_MULT = 'tests/ImageFeaturizer_testing/array_tests/check_prediction_array_{}_mult.npy'

# Supported models
MODELS = ['squeezenet', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']

# Arguments to load the data into the featurizers
LOAD_DATA_ARGS = {
                  'image_column_headers': 'images',
                  'image_path': 'tests/feature_preprocessing_testing/test_images',
                  'new_csv_name': TEST_CSV_NAME
                 }

# Static expected attributes to compare with the featurizer attributes
COMPARE_ARGS = {
                'downsample_size': 0,
                'image_column_headers': ['images'],
                'automatic_downsample': False,
                'csv_path': TEST_CSV_NAME,
                'image_list': IMAGE_LIST,
                'depth': 1,
               }

LOAD_DATA_ARGS_MULT_ERROR = {
                       'image_column_headers': ['images_1', 'images_2'],
                       'image_path': 'tests/feature_preprocessing_testing/test_images',
                       'new_csv_name': TEST_CSV_NAME
                      }

LOAD_DATA_ARGS_MULT = {
                       'image_column_headers': ['images_1', 'images_2'],
                       'image_path': 'tests/feature_preprocessing_testing/test_images',
                       'csv_path': CSV_NAME_MULT
                      }

COMPARE_ARGS_MULT = {
                     'downsample_size': 0,
                     'image_column_headers': ['images_1', 'images_2'],
                     'automatic_downsample': True,
                     'csv_path': CSV_NAME_MULT,
                     'image_list': IMAGE_LIST_MULT,
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

LOAD_PARAMS_MULT = [
               ('squeezenet', (227, 227), CHECK_ARRAY_MULT.format('squeezenet')),
               ('vgg16', (224, 224), CHECK_ARRAY_MULT.format('vgg16')),
               ('vgg19', (224, 224), CHECK_ARRAY_MULT.format('vgg19')),
               ('resnet50', (224, 224), CHECK_ARRAY_MULT.format('resnet50')),
               ('inceptionv3', (299, 299), CHECK_ARRAY_MULT.format('inceptionv3')),
               ('xception', (299, 299), CHECK_ARRAY_MULT.format('xception'))
              ]


# Remove path to the generated csv if it currently exists
if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
    shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')


def compare_featurizer_class(featurizer,
                             scaled_size,
                             featurized_data,
                             downsample_size,
                             image_column_headers,
                             automatic_downsample,
                             csv_path,
                             image_list,
                             depth,
                             featurized=False):
    """Check the necessary assertions for a featurizer image."""
    assert featurizer.scaled_size == scaled_size
    assert np.array_equal(featurizer.featurized_data, featurized_data)
    assert featurizer.downsample_size == downsample_size
    assert featurizer.image_column_headers == image_column_headers
    assert featurizer.auto_sample == automatic_downsample
    assert featurizer.csv_path == csv_path
    assert featurizer.image_list == image_list
    assert featurizer.depth == depth
    if featurized:
        assert filecmp.cmp('{}_full'.format(csv_path), CHECK_CSV.format(featurizer.model_name))


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


def test_load_data_single_column():
    """Test that the featurizer saves attributes correctly after loading data"""
    f = ImageFeaturizer()
    f.load_data(**LOAD_DATA_ARGS)
    compare_featurizer_class(f, (227, 227), np.zeros((1)), **COMPARE_ARGS)

    # Remove path to the generated csv at end of test
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')

def test_load_data_multiple_columns_no_csv():
    """Test featurizer raises error if multiple columns passed with only a directory"""
    f = ImageFeaturizer()
    with pytest.raises(ValueError):
        f.load_data(**LOAD_DATA_ARGS_MULT_ERROR)

def test_load_data_multiple_columns():
    """Test featurizer loads data correctly with multiple image columns"""
    f = ImageFeaturizer(auto_sample=True)
    f.load_data(**LOAD_DATA_ARGS_MULT)
    compare_featurizer_class(f, (227, 227), np.zeros((1)), **COMPARE_ARGS_MULT)


@pytest.mark.parametrize('model,size,array_path', LOAD_PARAMS_MULT, ids=MODELS)
def test_load_and_featurize_data_multiple_columns(model, size, array_path):
    """Test featurizations and attributes for each model are correct with multiple image columns"""
    f = ImageFeaturizer(model=model, auto_sample=True)
    f.load_and_featurize_data(**LOAD_DATA_ARGS_MULT)
    check_array = np.load(array_path)
    compare_featurizer_class(f, size, check_array, **COMPARE_ARGS_MULT)

    # Remove path to the generated csv at the end of the test
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')

    if os.path.isfile('{}_full'.format(CSV_NAME_MULT)):
        os.remove('{}_full'.format(CSV_NAME_MULT))
        pass
    if os.path.isfile('{}_features_only'.format(CSV_NAME_MULT)):
        os.remove('{}_features_only'.format(CSV_NAME_MULT))

@pytest.mark.parametrize('model,size,array_path', LOAD_PARAMS, ids=MODELS)
def test_load_and_featurize_single_column(model, size, array_path):
    """Test that all of the featurizations and attributes for each model are correct"""
    f = ImageFeaturizer(model=model)
    f.load_and_featurize_data(**LOAD_DATA_ARGS)
    check_array = np.load(array_path)
    compare_featurizer_class(f, size, check_array, **COMPARE_ARGS)

    # Remove path to the generated csv at the end of the test
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')
