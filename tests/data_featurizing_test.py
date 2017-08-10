"""Test data_featurizing module"""
import filecmp
import os

import numpy as np
import pandas as pd
import pytest
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential

from .build_featurizer_test import ATOL
from pic2vec.data_featurizing import (featurize_data,
                                      _named_path_finder,
                                      _features_to_csv,
                                      _create_features_df)

np.random.seed(5102020)

# The paths to the toy csvs
CHECK_CSV_IMAGES_PATH = 'tests/data_featurizing_testing/csv_testing/featurize_data_check_csv_images'
CHECK_CSV_FULL_PATH = 'tests/data_featurizing_testing/csv_testing/featurize_data_check_csv_full'
CHECK_CSV_FEATURES_ONLY_PATH = ('tests/data_featurizing_testing/csv_testing/'
                                'featurize_data_check_csv_features_only')

# The image list from the csv
CHECK_IMAGE_LIST = ['borges.jpg', 'arendt.bmp', 'sappho.png']


# The mock array being treated as the vectorized data
check_data_temp = np.ones((4, 2, 2, 2))
check_data_temp[2] = np.zeros((2, 2, 2))
CHECK_DATA = check_data_temp

# The mock array being treated as the "full featurized data"
CHECK_ARRAY = np.array([[1., 2., 3.],
                        [4., 5., 6.],
                        [0., 0., 0.],
                        [7., 8., 9.]
                        ])

# Create model
MODEL = Sequential([
                    Conv2D(5, (3, 3), input_shape=(5, 5, 3), activation='relu'),
                    Flatten(),
                    Dense(5)
                   ])


def test_featurize_data_bad_array():
    """Test errors with a badly formatted array"""
    error_array = np.ones((5, 5, 10))

    with pytest.raises(ValueError):
        featurize_data(MODEL, error_array)


def test_featurize_data():
    """
    Test that the featurize_data model correctly outputs the features of a toy
    network on a toy tensor
    """
    # Create the checked array
    init_array = np.ones((5, 5, 5, 3))

    for i in range(5):
        init_array[i] = init_array[i] * i

    # Check the prediction vs. the saved array
    check_array = np.load('tests/data_featurizing_testing/array_testing/check_featurize.npy')
    assert np.allclose(featurize_data(MODEL, init_array), check_array, atol=ATOL)


def test_named_path_finder():
    """Check that named_path_finder produces the correct output (without time)"""
    check_named_path = 'csv_name_modelstring_depth-n_output-x'
    test_named_path = _named_path_finder('csv_name', 'modelstring', 'n', 'x',
                                         omit_model=False, omit_depth=False, omit_output=False,
                                         omit_time=True)
    assert check_named_path == test_named_path

def test_features_to_csv_bad_feature_array():
    """
    Test that the model raises an error when a bad array
    is passed in (i.e. wrong shape)
    """
    # An error array with the wrong size
    error_array = np.zeros((4, 3, 2))
    with pytest.raises(ValueError):
        _features_to_csv(CHECK_DATA, error_array, CHECK_CSV_IMAGES_PATH, 'image', CHECK_IMAGE_LIST,
                         model_output='', model_str='', model_depth='', omit_time=True,
                         omit_depth=True, omit_output=True, save_features=True)


def test_features_to_csv_bad_column_header():
    """Raise an error when the column header is not found in the csv"""
    with pytest.raises(ValueError):
        _features_to_csv(CHECK_DATA, CHECK_ARRAY, CHECK_CSV_IMAGES_PATH, 'derp', CHECK_IMAGE_LIST,
                         model_output='', model_str='', model_depth='', omit_time=True,
                         omit_depth=True, omit_output=True, save_features=True)

def test_features_to_csv_bad_data_array():
    """Raise error when a bad data array is passed (i.e. wrong shape)"""
    # An error array with the wrong size
    error_array = np.zeros((4, 3, 2))
    with pytest.raises(ValueError):
        _features_to_csv(error_array, CHECK_ARRAY, CHECK_CSV_IMAGES_PATH, 'image', CHECK_IMAGE_LIST,
                         model_output='', model_str='', model_depth='', omit_time=True,
                         omit_depth=True, omit_output=True, save_features=True)

def test_create_features_df():
    """Test that the correct full array is created to be passed to the features_to_csv function"""
    df = pd.read_csv(CHECK_CSV_IMAGES_PATH)
    full_df_test = _create_features_df(CHECK_DATA, CHECK_ARRAY, 'image', df)[0]
    assert full_df_test.equals(pd.read_csv(CHECK_CSV_FULL_PATH))

def test_features_to_csv():
    """Test that the model creates the correct csvs from a toy array, csv, and image list"""
    # Check and remove the generated csvs if they already exist
    if os.path.isfile('{}_full'.format(CHECK_CSV_IMAGES_PATH)):
        os.remove('{}_full'.format(CHECK_CSV_IMAGES_PATH))
    if os.path.isfile('{}_features_only'.format(CHECK_CSV_IMAGES_PATH)):
        os.remove('{}_features_only'.format(CHECK_CSV_IMAGES_PATH))

    # Create the test
    full_test_dataframe = _features_to_csv(CHECK_DATA, CHECK_ARRAY, CHECK_CSV_IMAGES_PATH,
                                           'image', CHECK_IMAGE_LIST, '', '', '', omit_model=True,
                                           omit_depth=True, omit_output=True, omit_time=True,
                                           save_features=True)

    # Assert that the dataframe returned is correct, and the csv was generated correctly
    try:
        assert np.array_equal(full_test_dataframe, pd.read_csv(CHECK_CSV_FULL_PATH))
        assert filecmp.cmp('{}_features_only'.format(CHECK_CSV_IMAGES_PATH),
                           CHECK_CSV_FEATURES_ONLY_PATH)
        assert filecmp.cmp('{}_full'.format(CHECK_CSV_IMAGES_PATH), CHECK_CSV_FULL_PATH)

    # Remove the generated files
    finally:
        if os.path.isfile('{}_full'.format(CHECK_CSV_IMAGES_PATH)):
            os.remove('{}_full'.format(CHECK_CSV_IMAGES_PATH))
        if os.path.isfile('{}_features_only'.format(CHECK_CSV_IMAGES_PATH)):
            os.remove('{}_features_only'.format(CHECK_CSV_IMAGES_PATH))
