from keras.layers import Dense, Activation
from keras.layers.merge import average, add
from keras.models import Sequential

import keras.backend as K
import numpy as np

import pytest
import random
import os

from image_featurizer.build_featurizer import \
    _decapitate_model, _find_pooling_constant, _splice_layer, _downsample_model_features, \
    _initialize_model, _check_downsampling_mismatch, build_featurizer

random.seed(5102020)

def test_decapitate_model():
    '''
    This test creates a toy network, and checks that it calls the right errors
    and checks that it decapitates the network correctly:
    '''
    # Create model
    model = Sequential([
        Dense(40, input_shape=(100,)),
        Activation('relu'),
        Dense(20),
        Activation('relu'),
        Dense(10),
        Activation('relu'),
        Dense(5),
        Activation('softmax'),])

    _decapitate_model(model, 5)

    # Check for Type Error when model is passed something that isn't a Model
    with pytest.raises(TypeError):
        _decapitate_model(K.constant(3, shape=(3,4)), 4)
    with pytest.raises(TypeError):
        _decapitate_model(model.layers[-1],1)

    # Check for TypeError when depth is not passed an integer
    with pytest.raises(TypeError):
        _decapitate_model(model,2.0)

    # Check for Value Error when passed a depth >= (# of layers in network) - 1
    with pytest.raises(ValueError):
        _decapitate_model(model,7)


    # Make checks for all of the necessary features: the model outputs, the
    # last layer, the last layer's connections, and the last layer's shape
    assert model.layers[-1] == model.layers[2]
    assert model.layers[2].outbound_nodes == []
    assert model.outputs == [model.layers[2].output]
    assert model.layers[-1].output_shape == (None, 20)


def test_splice_layer():
    '''
    Test method splices tensors correctly
    '''
    # Create toy tensor
    tensor = K.constant(3, shape=(3,12))

    # Check for Value Error with non-integer number of slices
    with pytest.raises(ValueError):
        _splice_layer(tensor, 1.0)

    # Check for Value Error when # slices is not an integer divisor of the
    # total number of features
    with pytest.raises(ValueError):
        _splice_layer(tensor, 5)
    with pytest.raises(ValueError):
        _splice_layer(tensor,24)

    # Create spliced and added layers via splicing function
    tensor = K.constant(3, shape=(3,12))

    list_of_spliced_layers = _splice_layer(tensor, 3)

    # Add each of the layers together
    x = add(list_of_spliced_layers)


    # Create the spliced and added layers by hand
    check_layer = K.constant(9, shape=(3,4))

    # Check the math is right by hand
    assert np.array_equal(K.eval(check_layer), K.eval(x))


def test_find_pooling_constant():
    '''
    Test method returns correct pooling constant, and raises errors with
    badly formatted or incorrectly sized inputs
    '''
    features = K.constant(2, shape=(3,60))

    # Check for Value Error when user tries to upsample
    with pytest.raises(ValueError):
        _find_pooling_constant(features,120)

    # Check for Type Error when pool is not a divisor of the number of features
    with pytest.raises(ValueError):
        _find_pooling_constant(features,40)

    # Check for Type Error when pool is not a divisor of the number of features
    with pytest.raises(ValueError):
        _find_pooling_constant(features,0)

    # Check for Type Error when number of pooled features is not an integer
    with pytest.raises(TypeError):
        _find_pooling_constant(features, 1.5)

    # Check that it gives the right answer when formatted correctly
    assert _find_pooling_constant(features, 6) == 10

def test_downsample_model_features():
    '''
    Test creates a toy numpy array, and checks that the method
    correctly downsamples the array into a hand-checked tensor
    '''
    # Create the spliced and averaged tensor via downsampling function
    array = np.array([[1,2,3,4,5,6,7,8,9,10],
                      [11,12,13,14,15,16,17,18,19,20],
                      [21,22,23,24,25,26,27,28,29,30]
                      ])
    tensor = K.variable(array)

    x = _downsample_model_features(tensor, 5)

    # Create the spliced and averaged tensor by hand
    check_array=np.array([[1.5,3.5,5.5,7.5,9.5],
                          [11.5,13.5,15.5,17.5,19.5],
                          [21.5,23.5,25.5,27.5,29.5]
                         ])
    check_tensor = K.variable(check_array)

    # Check that they are equal: that it returns the correct tensor!
    assert np.array_equal(K.eval(check_tensor), K.eval(x))

def test_check_downsampling_mismatch():
    '''
    Test method correctly returns from mismatched downsample flags and inputs
    '''
    # Testing automatic downsampling at each depth
    # Depth 1
    assert _check_downsampling_mismatch(True,0,1) == (True, 1024)
    assert _check_downsampling_mismatch(False,0,1) == (False, 0)
    assert _check_downsampling_mismatch(False,512,1) == (True, 512)

    # Depth 2
    assert _check_downsampling_mismatch(True,0,2) == (True, 1024)
    assert _check_downsampling_mismatch(False,0,2) == (False, 0)
    assert _check_downsampling_mismatch(False,512,2) == (True, 512)

    # Depth 3
    assert _check_downsampling_mismatch(True,0,3) == (True, 1024)
    assert _check_downsampling_mismatch(False,0,3) == (False, 0)
    assert _check_downsampling_mismatch(False,512,3) == (True, 512)

    # Depth 4
    assert _check_downsampling_mismatch(True,0,4) == (True, 640)
    assert _check_downsampling_mismatch(False,0,4) == (False, 0)
    assert _check_downsampling_mismatch(False,640,4) == (True, 640)



def test_initialize_model():
    '''
    Test initializes the non-decapitated network, and checks that it correctly
    loaded the weights by checking its batch prediction on a pre-calculated, saved tensor.
    '''
    weight_path = 'image_featurizer/model/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
    # Initialize the model
    model = _initialize_model()

    if os.path.isfile(weight_path):
        try:
            os.rename(weight_path,'image_featurizer/model/testing_download_weights')
            model_downloaded_weights= _initialize_model()
            os.rename('image_featurizer/model/testing_download_weights',weight_path)
        except:
             raise AssertionError('Problem loading weights from keras, or changing' \
                                  'name of weight file!')

    # Testing models are the same from loaded weights and downloaded from keras
    assert len(model_downloaded_weights.layers) ==  len(model.layers)

    for layer in range(len(model.layers)):
        for array in range(len(model.layers[layer].get_weights())):
            assert np.array_equal(model.layers[layer].get_weights()[array], model_downloaded_weights.layers[layer].get_weights()[array])


    # Create the test array to be predicted on
    test_array = np.zeros((1,299,299,3))

    # I created this prediction earlier with the full model
    check_prediction = np.load('tests/featurizer_testing/test_initializer/inception_test_prediction.npy')

    # Check that it predicts correctly to see if weights were correctly loaded
    assert np.array_equal(model.predict_on_batch(test_array), check_prediction)

def test_build_featurizer():
    '''
    This integration test builds the full featurizer, and checks that it
    correctly builds the model with multiple options
    '''
    def check_featurizer(model,length, output_shape):
        assert len(model.layers)==length
        assert model.layers[-1].output_shape == output_shape

    ## Checking Depth 1 ##
    # Checking with downsampling
    model = build_featurizer(1,False,1024)
    check_featurizer(model, 315, (None,1024))

    model = build_featurizer(1,True,512)
    check_featurizer(model, 317, (None,512))

    # Checking without downsampling
    model = build_featurizer(1,False, 0)
    check_featurizer(model, 312, (None,2048))


    ## Checking Depth 2 ##
    # Checking with downsampling
    model = build_featurizer(2,False,1024)
    check_featurizer(model, 285, (None,1024))

    model = build_featurizer(2,True,512)
    check_featurizer(model, 287, (None,512))

    # Checking without downsampling
    model = build_featurizer(2,False, 0)
    check_featurizer(model, 282, (None,2048))


    ## Checking Depth 3 ##
    # Checking with downsampling
    model = build_featurizer(3,False,1024)
    check_featurizer(model, 284, (None,1024))

    model = build_featurizer(3,True,512)
    check_featurizer(model, 286, (None,512))

    # Checking without downsampling
    model = build_featurizer(3,False, 0)
    check_featurizer(model, 281, (None,2048))


    ## Checking Depth 4 ##
    # Checking with downsampling
    model = build_featurizer(4,False,640)
    check_featurizer(model, 254, (None,640))

    model = build_featurizer(4,True,320)
    check_featurizer(model, 256, (None,320))

    # Checking without downsampling
    model = build_featurizer(4,False, 0)
    check_featurizer(model, 251, (None,1280))



if __name__ == '__main__':
    test_decapitate_model()
    test_splice_layer()
    test_find_pooling_constant()
    test_downsample_model_features()
    test_initialize_model()
    #test_build_featurizer()
