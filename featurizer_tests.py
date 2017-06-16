from keras.applications.inception_v3 import InceptionV3
from keras.layers import Lambda, Dense, Activation
from keras.layers.merge import average, add
from keras.models import Sequential
import keras.backend as K
import numpy as np
import pytest

from image_featurizer.build_featurizer import *



def test_decapitate_model():
    # Create model
    model = Sequential([
        Dense(40, input_shape=(784,)),
        Activation('relu'),
        Dense(20),
        Activation('relu'),
        Dense(10),
        Activation('relu'),
        Dense(5),
        Activation('softmax'),])

    decapitate_model(model, 5)

    # Check for Type Error when model is passed something that isn't a Model
    with pytest.raises(TypeError):
        decapitate_model(K.constant(3, shape=(3,4)), 4)
    with pytest.raises(TypeError):
        decapitate_model(model.layers[-1],1)

    # Check for TypeError when depth is not passed an integer
    with pytest.raises(TypeError):
        decapitate_model(model,2.0)

    # Check for Value Error when passed a depth >= (# of layers in network) - 1
    with pytest.raises(ValueError):
        decapitate_model(model,7)


    # Make checks for all of the necessary features: the model outputs, the
    # last layer, the last layer's connections, and the last layer's shape
    assert model.layers[-1] == model.layers[2]
    assert model.layers[2].outbound_nodes == []
    assert model.outputs == [model.layers[2].output]
    assert model.layers[-1].output_shape == (None, 20)


def test_splice_layer():
    tensor = K.constant(3, shape=(3,12))

    # Check for Value Error with non-integer number of slices
    with pytest.raises(ValueError):
        splice_layer(tensor, 1.5)

    # Check for Value Error when # slices is not an integer divisor of the
    # total number of features
    with pytest.raises(ValueError):
        splice_layer(tensor, 5)
    with pytest.raises(ValueError):
        splice_layer(tensor,24)

    # Create spliced and added layers via splicing function
    tensor = K.constant(3, shape=(3,12))

    list_of_spliced_layers = splice_layer(tensor, 3)

    # Add each of the layers together
    x = add(list_of_spliced_layers)


    # Create the spliced and added layers by hand
    check_layer = K.constant(9, shape=(3,4))

    # Check the math is right by hand
    assert np.array_equal(K.eval(check_layer), K.eval(x))


def test_find_pooling_constant():
    features = K.constant(2, shape=(3,60))

    # Check for Value Error when user tries to upsample
    with pytest.raises(ValueError):
        find_pooling_constant(features,120)

    # Check for Type Error when pool is not a divisor of the number of features
    with pytest.raises(ValueError):
        find_pooling_constant(features,40)

    # Check for Type Error when number of pooled features is not an integer
    with pytest.raises(TypeError):
        find_pooling_constant(features, 1.5)

    # Check that it gives the right answer when formatted correctly
    assert find_pooling_constant(features, 6) == 10

def test_downsample_model_features():

    # Create the spliced and averaged tensor via downsampling function
    array = np.array([[1,2,3,4,5,6,7,8,9,10],
                      [11,12,13,14,15,16,17,18,19,20],
                      [21,22,23,24,25,26,27,28,29,30]
                      ])
    tensor = K.variable(array)

    x = downsample_model_features(tensor, 5)

    # Create the spliced and averaged tensor by hand
    check_array=np.array([[1.5,3.5,5.5,7.5,9.5],
                          [11.5,13.5,15.5,17.5,19.5],
                          [21.5,23.5,25.5,27.5,29.5]
                         ])
    check_tensor = K.variable(check_array)

    # Check that they are equal: that it returns the correct tensor!
    assert np.array_equal(K.eval(check_tensor), K.eval(x))

def test_initialize_model():

    # Initialize the model
    model = initialize_model()

    # Create the test array to be predicted on
    test_array = np.zeros((1,299,299,3))

    # I created this prediction earlier with the full model
    check_prediction = np.load('inception_test_prediction.npy')

    # Check that it predicts correctly to see if weights were correctly loaded
    assert np.array_equal(model.predict_on_batch(test_array), check_prediction)

# def test_build_featurizer():
#     model = build_featurizer(1,True,1024)
#     model.summary()


if __name__ == '__main__':
    test_decapitate_model()
    test_splice_layer()
    test_find_pooling_constant()
    test_downsample_model_features()
    test_initialize_model()
    #test_build_featurizer()
