from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation

import pytest
import numpy as np

from image_featurizer.data_featurizing import featurize_data, features_to_csv


def test_featurize_data():
    # Create model
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(30,30,3), activation='relu'),
        Conv2D(32, (3,3), activation='relu'),
        Dense(5),
        ])


    array = np.zeros((10,30,30,3))
    error_array = np.zeros((10,30,30))
    with pytest.raises(TypeError):
        featurize_data(model, 4)
    with pytest.raises(ValueError):
        featurize_data(model, error_array)

    assert featurize_data(model, array).shape == (10,5)

#def test_features_to_csv():
