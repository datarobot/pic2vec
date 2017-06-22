from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten

import pytest
import numpy as np
import random

from image_featurizer.data_featurizing import featurize_data, features_to_csv

np.random.seed(5102020)
model = Sequential([
    Conv2D(5, (3, 3), input_shape=(5,5,3), activation='relu'),
    Flatten(),
    Dense(5)
    ])

#model.load_weights('tests/test_model_featurize.h5')

array = np.ones((5,5,5,3))
for i in xrange(5):
    array[i]  = array[i]*i
model.predict(array)

check_array = np.array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [-0.01559529,  0.48380297, -1.26519454, -0.39580452, -0.75129372],
       [-0.03119057,  0.96760595, -2.53038907, -0.79160905, -1.50258744],
       [-0.04678613,  1.45140922, -3.79558277, -1.18741381, -2.25388145],
       [-0.06238115,  1.9352119 , -5.06077814, -1.5832181 , -3.00517464]], dtype='float32')
def test_featurize_data():
# Create model
    model = Sequential([
        Conv2D(5, (3, 3), input_shape=(5,5,3), activation='relu'),
        Flatten(),
        Dense(5)
        ])

    #model.load_weights('tests/test_model_featurize.h5')

    array = np.ones((5,5,5,3))
    for i in xrange(5):
        array[i]  = array[i]*i
    model.predict(array)
    error_array = np.ones((5,5,10))

    with pytest.raises(TypeError):
        featurize_data(model, 4)
    with pytest.raises(ValueError):
        featurize_data(model, error_array)

    assert np.array_equal(featurize_data(model, array), check_prediction)

#def test_features_to_csv():
