"""Test the full featurizer class"""
import filecmp
import os
import pytest
import shutil

import numpy as np
import pandas as pd

from pic2vec import ImageFeaturizer
from .build_featurizer_test import ATOL

# Constant paths
TEST_CSV_NAME = 'tests/ImageFeaturizer_testing/csv_tests/generated_images_csv_test'
IMAGE_LIST = ['arendt.bmp', 'borges.jpg', 'sappho.png']
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
    'image_dict': {'images': IMAGE_LIST},
    'depth': 1
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
    'image_dict': {'images_1': IMAGE_LIST_MULT[0], 'images_2': IMAGE_LIST_MULT[1]},
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


# Model fixtures
@pytest.fixture(scope='module', autouse=True)
def model_dict():
    # Initialize the dictionary
    model_dict = {'sample': {}, 'no_sample': {}}

    # For each of the sampling options, and for each model, build the model and save it to the dict
    for sample in ['sample', 'no_sample']:
        for model in MODELS:
            auto_sample = (sample == 'sample')
            model_dict[sample][model] = ImageFeaturizer(model=model, auto_sample=auto_sample)

    return model_dict


def compare_featurizer_class(featurizer,
                             scaled_size,
                             featurized_data,
                             downsample_size,
                             image_column_headers,
                             automatic_downsample,
                             csv_path,
                             image_dict,
                             depth,
                             featurized=False):
    """Check the necessary assertions for a featurizer image."""
    assert featurizer.scaled_size == scaled_size
    assert np.allclose(featurizer.features.as_matrix(), featurized_data, atol=ATOL)
    assert featurizer.downsample_size == downsample_size
    assert featurizer.image_column_headers == image_column_headers
    assert featurizer.auto_sample == automatic_downsample
    assert featurizer.csv_path == csv_path
    assert featurizer.image_dict == image_dict
    assert featurizer.depth == depth
    if featurized:
        assert filecmp.cmp('{}_full'.format(csv_path), CHECK_CSV.format(featurizer.model_name))
        assert featurizer.full_dataframe == pd.read_csv(CHECK_CSV.format(featurizer.model_name))


def test_featurize_first():
    """Test that the featurizer raises an error if featurize is called before loading data"""
    f = ImageFeaturizer()
    # Raise error if attempting to featurize before loading data
    with pytest.raises(IOError):
        f.featurize()


def testing_featurizer_build():
    """Test that the featurizer saves empty attributes correctly after initializing"""
    f = ImageFeaturizer()
    compare_featurizer_class(f, (0, 0), np.zeros((1)), 0, '', False, '', {}, 1)


def test_load_data_single_column(model_dict):
    """Test that the featurizer saves attributes correctly after loading data"""
    f = model_dict['no_sample']['squeezenet']
    f.clear_input(confirm=True)
    f.load_data(**LOAD_DATA_ARGS)
    compare_featurizer_class(f, (227, 227), np.zeros((1)), **COMPARE_ARGS)

    # Remove path to the generated csv at end of test
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')


def test_load_data_multiple_columns_no_csv(model_dict):
    """Test featurizer raises error if multiple columns passed with only a directory"""
    f = model_dict['no_sample']['squeezenet']
    f.clear_input(confirm=True)
    with pytest.raises(ValueError):
        f.load_data(**LOAD_DATA_ARGS_MULT_ERROR)


def test_load_data_multiple_columns(model_dict):
    """Test featurizer loads data correctly with multiple image columns"""
    f = model_dict['sample']['squeezenet']
    f.clear_input(confirm=True)
    f.load_data(**LOAD_DATA_ARGS_MULT)
    compare_featurizer_class(f, (227, 227), np.zeros((1)), **COMPARE_ARGS_MULT)


def test_save_csv(model_dict):
    """Make sure the featurizer writes the name correctly to csv with robust naming config"""
    f = model_dict['no_sample']['squeezenet']
    f.clear_input(confirm=True)
    f.load_and_featurize_data(save_csv=True, save_features=True, omit_time=True,
                              **LOAD_DATA_ARGS_MULT)
    check_array_path = '{}_squeezenet_depth-1_output-512'.format(CSV_NAME_MULT)
    full_check = '{}_full'.format(check_array_path)
    feature_check = '{}_features_only'.format(check_array_path)
    f.save_csv(save_features=True, omit_time=True)
    try:
        assert os.path.isfile(full_check)
        assert os.path.isfile(feature_check)
    finally:
        if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
            shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')

        if os.path.isfile('{}_full'.format(check_array_path)):
            os.remove('{}_full'.format(check_array_path))
            pass
        if os.path.isfile('{}_features_only'.format(check_array_path)):
            os.remove('{}_features_only'.format(check_array_path))


@pytest.mark.parametrize('model,size,array_path', LOAD_PARAMS_MULT, ids=MODELS)
def test_load_then_featurize_data_multiple_columns(model, size, array_path, model_dict):
    """Test featurizations and attributes for each model are correct with multiple image columns"""
    feat = model_dict['sample'][model]
    feat.clear_input(confirm=True)
    feat.load_data(**LOAD_DATA_ARGS_MULT)
    feat.featurize(save_features=True, omit_time=True, omit_model=True, omit_depth=True,
                   omit_output=True, save_csv=True)
    check_array = np.load(array_path)

    try:
        compare_featurizer_class(feat, size, check_array, **COMPARE_ARGS_MULT)
    finally:
        # Remove path to the generated csv at the end of the test
        if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
            shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')

        if os.path.isfile('{}_full'.format(CSV_NAME_MULT)):
            os.remove('{}_full'.format(CSV_NAME_MULT))
            pass
        if os.path.isfile('{}_features_only'.format(CSV_NAME_MULT)):
            os.remove('{}_features_only'.format(CSV_NAME_MULT))


@pytest.mark.parametrize('model,size,array_path', LOAD_PARAMS_MULT, ids=MODELS)
def test_load_and_featurize_multiple_columns_no_batch(model, size, array_path, model_dict):
    """Test featurizations and attributes for each model are correct with multiple image columns"""
    feat = model_dict['sample'][model]
    feat.clear_input(confirm=True)
    feat.load_and_featurize_data(save_features=True, omit_time=True, omit_model=True,
                                 omit_depth=True, omit_output=True, **LOAD_DATA_ARGS_MULT)
    check_array = np.load(array_path)

    try:
        print(check_array.shape)
        print(feat.num_features)
        compare_featurizer_class(feat, size, check_array, **COMPARE_ARGS_MULT)
    finally:
        # Remove path to the generated csv at the end of the test
        if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
            shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')

        if os.path.isfile('{}_full'.format(CSV_NAME_MULT)):
            os.remove('{}_full'.format(CSV_NAME_MULT))
            pass
        if os.path.isfile('{}_features_only'.format(CSV_NAME_MULT)):
            os.remove('{}_features_only'.format(CSV_NAME_MULT))


@pytest.mark.parametrize('model,size,array_path', LOAD_PARAMS_MULT, ids=MODELS)
def test_load_and_featurize_data_multiple_columns_with_batch(model, size, array_path, model_dict):
    """Test featurizations and attributes for each model are correct with multiple image columns"""
    feat = model_dict['sample'][model]
    feat.clear_input(confirm=True)
    feat.load_and_featurize_data(batch_size=2, save_features=True, omit_time=True, omit_model=True,
                                 omit_depth=True, omit_output=True, **LOAD_DATA_ARGS_MULT)
    check_array = np.load(array_path)

    try:
        print(check_array.shape)
        print(feat.num_features)
        compare_featurizer_class(feat, size, check_array, **COMPARE_ARGS_MULT)
    finally:
        # Remove path to the generated csv at the end of the test
        if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
            shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')

        if os.path.isfile('{}_full'.format(CSV_NAME_MULT)):
            os.remove('{}_full'.format(CSV_NAME_MULT))
            pass
        if os.path.isfile('{}_features_only'.format(CSV_NAME_MULT)):
            os.remove('{}_features_only'.format(CSV_NAME_MULT))


@pytest.mark.parametrize('model,size,array_path', LOAD_PARAMS, ids=MODELS)
def test_load_and_featurize_single_column_no_batch(model, size, array_path, model_dict):
    """Test that all of the featurizations and attributes for each model are correct"""
    feat = model_dict['no_sample'][model]
    feat.clear_input(confirm=True)
    feat.load_and_featurize_data(save_features=True, omit_time=True, omit_model=True,
                                 omit_depth=True, omit_output=True, **LOAD_DATA_ARGS)

    check_array = np.load(array_path)

    try:
        compare_featurizer_class(feat, size, check_array, **COMPARE_ARGS)
    finally:
        # Remove path to the generated csv at the end of the test
        if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
            shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')


@pytest.mark.parametrize('model,size,array_path', LOAD_PARAMS, ids=MODELS)
def test_load_and_featurize_single_column_with_batch(model, size, array_path, model_dict):
    """Test that all of the featurizations and attributes for each model are correct"""
    feat = model_dict['no_sample'][model]
    feat.clear_input(confirm=True)
    feat.load_and_featurize_data(batch_size=2, save_features=True, omit_time=True, omit_model=True,
                                 omit_depth=True, omit_output=True, **LOAD_DATA_ARGS)

    check_array = np.load(array_path)

    try:
        compare_featurizer_class(feat, size, check_array, **COMPARE_ARGS)
    finally:
        # Remove path to the generated csv at the end of the test
        if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests'):
            shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests')
        del feat
