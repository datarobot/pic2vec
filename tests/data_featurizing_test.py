from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten

import pytest
import numpy as np
import random

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
    check_csv = 'tests/data_featurizing_testing/csv_testing/featurize_data_test_csv'
