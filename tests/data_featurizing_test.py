from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten

import pytest
import random
import filecmp
import os

import pandas as pd
import numpy as np

from image_featurizer.data_featurizing import featurize_data, features_to_csv



def test_featurize_data():
    np.random.seed(5102020)

    # Create model
    model = Sequential([
        Conv2D(5, (3, 3), input_shape=(5,5,3), activation='relu'),
        Flatten(),
        Dense(5)
        ])

    #model.load_weights('tests/test_model_featurize.h5')

    init_array = np.ones((5,5,5,3))

    for i in xrange(5):
        init_array[i]  = init_array[i]*i

    error_array = np.ones((5,5,10))

    with pytest.raises(TypeError):
        featurize_data(model, 4)
    with pytest.raises(ValueError):
        featurize_data(model, error_array)

    check_array = np.load('tests/data_featurizing_testing/array_testing/check_featurize.npy')
    assert np.array_equal(featurize_data(model, init_array), check_array)

def test_features_to_csv():
    check_csv_images = 'tests/data_featurizing_testing/csv_testing/featurize_data_check_csv_images'
    check_csv_full = 'tests/data_featurizing_testing/csv_testing/featurize_data_check_csv_full'


    check_array = np.array([[1.,2.,3.],
                            [4.,5.,6.],
                            [0.,0.,0.],
                            [7.,8.,9.]
                            ])

    error_array = np.zeros((4,3,2))

    check_image_list = ['borges.jpg', 'arendt.bmp', 'sappho.png']

    if os.path.isfile('{}_full'.format(check_csv_images)):
        os.remove('{}_full'.format(check_csv_images))
    if os.path.isfile('{}_features_only'.format(check_csv_images)):
        os.remove('{}_features_only'.format(check_csv_images))

    with pytest.raises(ValueError):
        error_df = features_to_csv(check_array, check_csv_images, 'derp',check_image_list)
    with pytest.raises(ValueError):
        error_df = features_to_csv(error_array, check_csv_images, 'images',check_image_list)


    full_test_dataframe = features_to_csv(check_array, check_csv_images, 'images',check_image_list)



    assert np.array_equal(full_test_dataframe, pd.read_csv(check_csv_full))
    assert filecmp.cmp('{}_full'.format(check_csv_images),check_csv_full)

    os.remove('{}_full'.format(check_csv_images))
    os.remove('{}_features_only'.format(check_csv_images))
