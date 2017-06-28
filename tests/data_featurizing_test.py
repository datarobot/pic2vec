import filecmp
import os

import numpy as np
import pandas as pd
import pytest
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential

from image_featurizer.data_featurizing import featurize_data, features_to_csv


def test_featurize_data():
    """
    Test that the featurize_data model correctly outputs the features of a toy
    network on a toy tensor
    """
    np.random.seed(5102020)

    # Create model
    model = Sequential([
        Conv2D(5, (3, 3), input_shape=(5, 5, 3), activation='relu'),
        Flatten(),
        Dense(5)
    ])

    # Create the checked array
    init_array = np.ones((5, 5, 5, 3))

    for i in xrange(5):
        init_array[i] = init_array[i] * i

    error_array = np.ones((5, 5, 10))

    # Check that it raises errors with badly formatted models or arrays
    with pytest.raises(TypeError):
        featurize_data(model, 4)
    with pytest.raises(TypeError):
        featurize_data(4, init_array)
    with pytest.raises(ValueError):
        featurize_data(model, error_array)

    # Check the prediction vs. the saved array
    check_array = np.load('tests/data_featurizing_testing/array_testing/check_featurize.npy')
    assert np.array_equal(featurize_data(model, init_array), check_array)


def test_features_to_csv():
    """
    Test that the model creates the correct csvs from a toy array, csv, and
     image list
    """

    # The paths to the toy csvs
    check_csv_images = 'tests/data_featurizing_testing/csv_testing/featurize_data_check_csv_images'
    check_csv_full = 'tests/data_featurizing_testing/csv_testing/featurize_data_check_csv_full'
    check_csv_features_only = ('tests/data_featurizing_testing/csv_testing/'
                               'featurize_data_check_csv_features_only')
    # Build the array to treat as the "full featurized data"
    check_array = np.array([[1., 2., 3.],
                            [4., 5., 6.],
                            [0., 0., 0.],
                            [7., 8., 9.]
                            ])

    # An error array with the wrong size
    error_array = np.zeros((4, 3, 2))

    # The image list from the csv
    check_image_list = ['borges.jpg', 'arendt.bmp', 'sappho.png']

    # Check and remove the generated csvs if they already exist
    if os.path.isfile('{}_full'.format(check_csv_images)):
        os.remove('{}_full'.format(check_csv_images))
    if os.path.isfile('{}_features_only'.format(check_csv_images)):
        os.remove('{}_features_only'.format(check_csv_images))

    # Raise errors with badly formatted input: a bad array, or an image column header
    # not present in the csv
    with pytest.raises(ValueError):
        features_to_csv(check_array, check_csv_images, 'derp', check_image_list)
    with pytest.raises(ValueError):
        features_to_csv(error_array, check_csv_images, 'images', check_image_list)

    # Create the test
    full_test_dataframe = features_to_csv(check_array, check_csv_images, 'images', check_image_list)

    # Assert that the dataframe returned is correct, and the csv was generated correclty
    assert np.array_equal(full_test_dataframe, pd.read_csv(check_csv_full))
    assert filecmp.cmp('{}_features_only'.format(check_csv_images), check_csv_features_only)
    assert filecmp.cmp('{}_full'.format(check_csv_images), check_csv_full)

    # Remove the generated files
    os.remove('{}_full'.format(check_csv_images))
    os.remove('{}_features_only'.format(check_csv_images))
