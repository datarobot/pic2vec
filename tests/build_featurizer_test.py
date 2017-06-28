from keras.layers import Dense, Activation
from keras.layers.merge import average, add
from keras.models import Sequential

import keras.backend as K
import numpy as np

import pytest
import random
import os
import copy

from image_featurizer.build_featurizer import \
    _decapitate_model, _find_pooling_constant, _splice_layer, _downsample_model_features, \
    _initialize_model, _check_downsampling_mismatch, build_featurizer

from image_featurizer.squeezenet import SqueezeNet

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

    model = _decapitate_model(model, 5)

    # Check for Type Error when model is passed something that isn't a Model
    with pytest.raises(TypeError):
        error = _decapitate_model(K.constant(3, shape=(3,4)), 4)
    with pytest.raises(TypeError):
        error = _decapitate_model(model.layers[-1],1)

    # Check for TypeError when depth is not passed an integer
    with pytest.raises(TypeError):
        error = _decapitate_model(model,2.0)

    # Check for Value Error when passed a depth >= (# of layers in network) - 1
    with pytest.raises(ValueError):
        error = _decapitate_model(model,7)


    # Make checks for all of the necessary features: the model outputs, the
    # last layer, the last layer's connections, and the last layer's shape
    assert model.layers[-1] == model.layers[3]
    assert model.layers[3].outbound_nodes == []
    assert model.outputs == [model.layers[3].output]
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
    # Raises error with no
    with pytest.raises(ValueError):
        _check_downsampling_mismatch(True,0, 2049)

    # Testing automatic downsampling at each depth
    assert _check_downsampling_mismatch(True,0,2048) == (True, 1024)
    assert _check_downsampling_mismatch(False,0,2048) == (False, 0)
    assert _check_downsampling_mismatch(False,512,2048) == (True, 512)


def check_model_equal(model1, model2):
    '''
    Checks whether two models are equal
    '''
    # Testing models are the same from loaded weights and downloaded from keras
    assert len(model1.layers) ==  len(model2.layers)

    for layer in range(len(model1.layers)):
        for array in range(len(model1.layers[layer].get_weights())):
            assert np.array_equal(model1.layers[layer].get_weights()[array], model2.layers[layer].get_weights()[array])

def test_initialize_model_squeezenet():
    '''
    Test the initialization of the loaded SqueezeNet model
    '''

    with pytest.raises(TypeError):
        _initialize_model(4)
    with pytest.raises(ValueError):
        _initialize_model('error!')

    weight_path = 'image_featurizer/model/squeezenet_weights_tf_dim_ordering_tf_kernels.h5'
    changed_weights_test = 'image_featurizer/model/changed_weight_name'
    if os.isfile(weight_path):
        os.rename(weight_path, changed_weights_test)
        try:
            with pytest.raises(ValueError):
                model = _initialize_model('squeezenet')
            os.rename(changed_weights_test, weight_path)
        except Exception as err:
            os.rename(changed_weights_test, weight_path)
            raise err


    # Initialize the model
    model = _initialize_model('SqueEzEneT')

    try:
        model_downloaded_weights= SqueezeNet()

    except:
        raise AssertionError('Problem loading squeezenet weights!')


    check_model_equal(model, model_downloaded_weights)

    # Create the test array to be predicted on
    test_array = np.zeros((1,227,227,3))

    # I created this prediction earlier with the full model
    check_prediction = np.load('tests/featurizer_testing/test_initializer/squeezenet_test_prediction.npy')

    # Check that it predicts correctly to see if weights were correctly loaded
    assert np.array_equal(model.predict_on_batch(test_array), check_prediction)


def test_initialize_model_inceptionv3():
    '''
    Test the initialization of the InceptionV3 model
    '''
    # Initialize the model
    model = _initialize_model('inceptionv3')

    assert len(model.layers) == 313

    # Create the test array to be predicted on
    test_array = np.zeros((1,299,299,3))

    # I created this prediction earlier with the full model
    check_prediction = np.load('tests/featurizer_testing/test_initializer/inception_test_prediction.npy')

    # Check that it predicts correctly to see if weights were correctly loaded
    assert np.array_equal(model.predict_on_batch(test_array), check_prediction)

def test_initialize_model_vgg16():
    '''
    Test the initialization of the VGG16 model
    '''
    # Initialize the model
    model = _initialize_model('vgg16')


    assert len(model.layers) == 23

    # Create the test array to be predicted on
    test_array = np.zeros((1,224,224,3))

    # I created this prediction earlier with the full model
    check_prediction = np.load('tests/featurizer_testing/test_initializer/vgg16_test_prediction.npy')

    # Check that it predicts correctly to see if weights were correctly loaded
    assert np.array_equal(model.predict_on_batch(test_array), check_prediction)


def test_initialize_model_vgg19():
    '''
    Test the initialization of the VGG19 model
    '''
    # Initialize the model
    model = _initialize_model('vgg19')

    assert len(model.layers) == 26

    # Create the test array to be predicted on
    test_array = np.zeros((1,224,224,3))

    # I created this prediction earlier with the full model
    check_prediction = np.load('tests/featurizer_testing/test_initializer/vgg19_test_prediction.npy')

    # Check that it predicts correctly to see if weights were correctly loaded
    assert np.array_equal(model.predict_on_batch(test_array), check_prediction)


def test_initialize_model_resnet50():
    '''
    Test the initialization of the ResNet50 model
    '''
    # Initialize the model
    model = _initialize_model('resnet50')

    assert len(model.layers) == 177

    # Create the test array to be predicted on
    test_array = np.zeros((1,224,224,3))

    # I created this prediction earlier with the full model
    check_prediction = np.load('tests/featurizer_testing/test_initializer/resnet50_test_prediction.npy')

    # Check that it predicts correctly to see if weights were correctly loaded
    assert np.array_equal(model.predict_on_batch(test_array), check_prediction)

def test_initialize_model_xception():
    '''
    Test the initialization of the Xception model
    '''
    # Initialize the model
    model = _initialize_model('xception')

    assert len(model.layers) == 134

    # Create the test array to be predicted on
    test_array = np.zeros((1,299,299,3))

    # I created this prediction earlier with the full model
    check_prediction = np.load('tests/featurizer_testing/test_initializer/xception_test_prediction.npy')

    # Check that it predicts correctly to see if weights were correctly loaded
    assert np.array_equal(model.predict_on_batch(test_array), check_prediction)


def test_build_featurizer_squeezenet():
    '''
    This integration test builds the full featurizer, and checks that it
    correctly builds the squeezenet model with multiple options
    '''

    squeezenet = _initialize_model('squeezenet')

    ## Checking Depth 1 ##
    # Checking with downsampling
    model = build_featurizer(1, False, 128, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,128)

    model = build_featurizer(1, True, 0, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,256)

    # Checking without downsampling
    model = build_featurizer(1, False, 0, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,512)


    ## Checking Depth 2 ##
    # Checking with downsampling
    model = build_featurizer(2, False, 128, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,128)

    model = build_featurizer(2, True, 0, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,256)

    # Checking without downsampling
    model = build_featurizer(2, False, 0, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,512)


    ## Checking Depth 3 ##
    # Checking with downsampling
    model = build_featurizer(3,False,96, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,96)

    model = build_featurizer(3,True,0, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,192)


    # Checking without downsampling
    model = build_featurizer(3,False, 0, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,384)


    ## Checking Depth 4 ##
    # Checking with downsampling
    model = build_featurizer(4,False,96, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,96)

    model = build_featurizer(4,True,0, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,192)

    # Checking without downsampling
    model = build_featurizer(4,False, 0, model_str='squeezenet', loaded_model=squeezenet)
    assert model.layers[-1].output_shape == (None,384)

def test_build_featurizer_vgg16():
    '''
    This integration test builds the full featurizer, and checks that it
    correctly builds the VGG16 model with multiple options
    '''
    vgg16 = _initialize_model('vgg16')

    # Check for error with badly passed loaded_model
    with pytest.raises(TypeError):
        error = build_featurizer(1, False, 1024, model_str='vgg16', loaded_model=4)

    ## Checking Depth 1 ##
    # Checking with downsampling
    model = build_featurizer(1, False, 1024, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,1024)

    model = build_featurizer(1, True, 0, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,2048)

    # Checking without downsampling
    model = build_featurizer(1, False, 0, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,4096)


    ## Checking Depth 2 ##
    # Checking with downsampling
    model = build_featurizer(2, False, 1024, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,1024)

    model = build_featurizer(2, True, 0, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,2048)

    # Checking without downsampling
    model = build_featurizer(2, False, 0, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,4096)


    ## Checking Depth 3 ##
    # Checking with downsampling
    model = build_featurizer(3,False,128, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,128)

    model = build_featurizer(3,True,0, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,256)

    # Checking without downsampling
    model = build_featurizer(3,False, 0, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,512)


    ## Checking Depth 4 ##
    # Checking with downsampling
    model = build_featurizer(4,False,128, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,128)

    model = build_featurizer(4,True,0, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,256)

    # Checking without downsampling
    model = build_featurizer(4,False, 0, model_str='vgg16', loaded_model=vgg16)
    assert model.layers[-1].output_shape == (None,512)

def test_build_featurizer_vgg19():
    '''
    This integration test builds the full featurizer, and checks that it
    correctly builds the VGG19 model with multiple options
    '''

    vgg19 = _initialize_model('vgg19')

    ## Checking Depth 1 ##
    # Checking with downsampling
    model = build_featurizer(1, False, 1024, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,1024)

    model = build_featurizer(1, True, 0, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,2048)

    # Checking without downsampling
    model = build_featurizer(1, False, 0, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,4096)


    ## Checking Depth 2 ##
    # Checking with downsampling
    model = build_featurizer(2, False, 1024, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,1024)

    model = build_featurizer(2, True, 0, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,2048)

    # Checking without downsampling
    model = build_featurizer(2, False, 0, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,4096)


    ## Checking Depth 3 ##
    # Checking with downsampling
    model = build_featurizer(3, False, 128, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,128)

    model = build_featurizer(3, True, 0, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,256)


    # Checking without downsampling
    model = build_featurizer(3, False, 0, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,512)


    ## Checking Depth 4 ##
    # Checking with downsampling
    model = build_featurizer(4,False, 128, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,128)

    model = build_featurizer(4,True,0, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,256)

    # Checking without downsampling
    model = build_featurizer(4,False, 0, model_str='vgg19', loaded_model=vgg19)
    assert model.layers[-1].output_shape == (None,512)

def test_build_featurizer_resnet50():
    '''
    This integration test builds the full featurizer, and checks that it
    correctly builds the model with multiple options
    '''

    resnet50 = _initialize_model('resnet50')

    ## Checking Depth 1 ##
    # Checking with downsampling
    model = build_featurizer(1, False, 512, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,512)

    model = build_featurizer(1, True, 0, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,1024)

    # Checking without downsampling
    model = build_featurizer(1, False, 0, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,2048)


    ## Checking Depth 2 ##
    # Checking with downsampling
    model = build_featurizer(2, False, 512, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,512)

    model = build_featurizer(2, True, 0, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,1024)

    # Checking without downsampling
    model = build_featurizer(2, False, 0, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,2048)


    ## Checking Depth 3 ##
    # Checking with downsampling
    model = build_featurizer(3,False,512, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,512)

    model = build_featurizer(3,True,0, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,1024)

    # Checking without downsampling
    model = build_featurizer(3,False, 0, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,2048)


    ## Checking Depth 4 ##
    # Checking with downsampling
    model = build_featurizer(4,False,512, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,512)

    model = build_featurizer(4,True,0, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,1024)

    # Checking without downsampling
    model = build_featurizer(4,False, 0, model_str='resnet50', loaded_model=resnet50)
    assert model.layers[-1].output_shape == (None,2048)

def test_build_featurizer_inceptionv3():
    '''
    This integration test builds the full featurizer, and checks that it
    correctly builds the model with multiple options
    '''
    def check_featurizer(model, output_shape):
        assert model.layers[-1].output_shape == output_shape

    inceptionv3 = _initialize_model('inceptionv3')

    ## Checking Depth 1 ##
    # Checking with downsampling
    model = build_featurizer(1, False, 512, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,512)

    model = build_featurizer(1, True, 0, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,1024)

    # Checking without downsampling
    model = build_featurizer(1, False, 0, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,2048)


    ## Checking Depth 2 ##
    # Checking with downsampling
    model = build_featurizer(2, False, 512, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,512)

    model = build_featurizer(2, True, 0, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,1024)

    # Checking without downsampling
    model = build_featurizer(2, False, 0, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,2048)


    ## Checking Depth 3 ##
    # Checking with downsampling
    model = build_featurizer(3, False, 512, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,512)

    model = build_featurizer(3, True, 0, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,1024)

    # Checking without downsampling
    model = build_featurizer(3, False, 0, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,2048)


    ## Checking Depth 4 ##
    # Checking with downsampling
    model = build_featurizer(4, False, 320, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,320)

    model = build_featurizer(4,True,0, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,640)

    # Checking without downsampling
    model = build_featurizer(4,False, 0, model_str='inceptionv3', loaded_model=inceptionv3)
    assert model.layers[-1].output_shape == (None,1280)

def test_build_featurizer_xception():
    '''
    This integration test builds the full featurizer, and checks that it
    correctly builds the model with multiple options
    '''
    xception = _initialize_model('xception')

    ## Checking Depth 1 ##
    # Checking with downsampling
    model = build_featurizer(1, False, 512, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,512)

    model = build_featurizer(1, True, 0, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,1024)

    # Checking without downsampling
    model = build_featurizer(1, False, 0, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,2048)


    ## Checking Depth 2 ##
    # Checking with downsampling
    model = build_featurizer(2, False, 256, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,256)

    model = build_featurizer(2, True, 0, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,512)

    # Checking without downsampling
    model = build_featurizer(2, False, 0, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,1024)


    ## Checking Depth 3 ##
    # Checking with downsampling
    model = build_featurizer(3,False,182, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,182)


    model = build_featurizer(3,True,0, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,364)

    # Checking without downsampling
    model = build_featurizer(3,False, 0, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,728)


    ## Checking Depth 4 ##
    # Checking with downsampling
    model = build_featurizer(4,False,182, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,182)

    model = build_featurizer(4,True,0, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,364)

    # Checking without downsampling
    model = build_featurizer(4,False, 0, model_str='xception', loaded_model=xception)
    assert model.layers[-1].output_shape == (None,728)



if __name__ == '__main__':
    test_decapitate_model()
    test_splice_layer()
    test_find_pooling_constant()
    test_downsample_model_features()
    test_initialize_model()
    #test_build_featurizer()
