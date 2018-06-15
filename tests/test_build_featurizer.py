"""Test the build_featurizer code."""
import os
import random
import warnings
import logging

import keras.backend as K
import numpy as np
import pytest
from keras.layers import Dense, Activation, Input
from keras.layers.merge import add
from keras.models import Sequential, Model

from pic2vec.build_featurizer import (_decapitate_model, _find_pooling_constant,
                                      _splice_layer, _downsample_model_features,
                                      _initialize_model, _check_downsampling_mismatch,
                                      build_featurizer)

from pic2vec.squeezenet import SqueezeNet

random.seed(5102020)

# Tolerance for prediction error
ATOL = 0.00001


@pytest.fixture(scope='module')
def check_model():
    # Building the checking model
    input_layer = Input(shape=(100, ))
    layer = Dense(40)(input_layer)
    layer = Activation('relu')(layer)
    layer = Dense(20)(layer)
    layer = Activation('relu')(layer)
    layer = Dense(10)(layer)
    layer = Activation('relu')(layer)
    layer = Dense(5)(layer)
    output_layer = Activation('softmax')(layer)

    check_model = Model(inputs=input_layer, outputs=output_layer)

    return check_model


# Create tensor for splicing
SPLICING_TENSOR = K.constant(3, shape=(3, 12))

# Create featurization for finding the pooling constant
POOLING_FEATURES = K.constant(2, shape=(3, 60))

# Path to checking prediction arrays for each model in _initialize_model
INITIALIZE_ARRAY = 'tests/build_featurizer_testing/{}_test_prediction.npy'

MODELS = ['squeezenet', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']


def test_decapitate_model_lazy_input():
    """Test an error is raised when the model has a lazy input layer initialization"""
    # Raise warning when model has lazy input layer initialization
    error_model = Sequential([
        Dense(40, input_shape=(100,)),
        Dense(20),
        Activation('softmax')])

    with warnings.catch_warnings(record=True) as warning_check:
        _decapitate_model(error_model, 1)
        assert len(warning_check) == 1
        assert "depth issues" in str(warning_check[-1].message)


def test_decapitate_model_too_deep(check_model):
    """Test error raised when model is decapitated too deep"""
    # Check for Value Error when passed a depth >= (# of layers in network) - 1
    with pytest.raises(ValueError):
        _decapitate_model(check_model, 8)


def test_decapitate_model(check_model):
    """
    This test creates a toy network, and checks that it calls the right errors
    and checks that it decapitates the network correctly:
    """
    # Create test model
    test_model = _decapitate_model(check_model, 5)

    # Make checks for all of the necessary features: the model outputs, the
    # last layer, the last layer's connections, and the last layer's shape
    assert test_model.layers[-1] == test_model.layers[3]
    assert test_model.layers[3].outbound_nodes == []
    assert test_model.outputs == [test_model.layers[3].output]
    assert test_model.layers[-1].output_shape == (None, 20)


def test_splice_layer_bad_split():
    """Check error with bad split on the tensor"""
    with pytest.raises(ValueError):
        _splice_layer(SPLICING_TENSOR, 5)


def test_splice_layer():
    """Test method splices tensors correctly"""
    # Create spliced and added layers via splicing function
    list_of_spliced_layers = _splice_layer(SPLICING_TENSOR, 3)
    # Add each of the layers together
    x = add(list_of_spliced_layers)
    # Create the spliced and added layers by hand
    check_layer = K.constant(9, shape=(3, 4))
    # Check the math
    assert np.allclose(K.eval(check_layer), K.eval(x), atol=ATOL)


def test_find_pooling_constant_upsample():
    """Test error when trying to upsample"""
    with pytest.raises(ValueError):
        _find_pooling_constant(POOLING_FEATURES, 120)


def test_find_pooling_constant_bad_divisor():
    """Test error when trying to downsample to a non-divisor of the features"""
    with pytest.raises(ValueError):
        _find_pooling_constant(POOLING_FEATURES, 40)

    with pytest.raises(ValueError):
        _find_pooling_constant(POOLING_FEATURES, 0)


def test_find_pooling_constant():
    """Test that pooling constant given correct answer with good inputs"""
    assert _find_pooling_constant(POOLING_FEATURES, 6) == 10


def test_downsample_model_features():
    """
    Test creates a toy numpy array, and checks that the method
    correctly downsamples the array into a hand-checked tensor
    """
    # Create the spliced and averaged tensor via downsampling function
    array = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
                      ])
    tensor = K.variable(array)

    x = _downsample_model_features(tensor, 5)

    # Create the spliced and averaged tensor by hand
    check_array = np.array([[1.5, 3.5, 5.5, 7.5, 9.5],
                            [11.5, 13.5, 15.5, 17.5, 19.5],
                            [21.5, 23.5, 25.5, 27.5, 29.5]
                            ])
    check_tensor = K.variable(check_array)
    # Check that they are equal: that it returns the correct tensor
    assert np.allclose(K.eval(check_tensor), K.eval(x), atol=ATOL)


def test_check_downsampling_mismatch_bad_num_features():
    """Raises error with autodownsampling an odd number of features"""
    with pytest.raises(ValueError):
        _check_downsampling_mismatch(True, 0, 2049)


def test_check_downsampling_mismatch_autosample():
    """Test method correctly autosamples"""
    # Testing automatic downsampling
    assert _check_downsampling_mismatch(True, 0, 2048) == (True, 1024)


def test_check_downsampling_mismatch_no_sample():
    """Test method correctly returns with no sampling"""
    # Testing no downsampling
    assert _check_downsampling_mismatch(False, 0, 2048) == (False, 0)


def test_check_downsampling_mismatch_manual_sample():
    """Test method correctly returns with manual sampling"""
    # Testing manual downsampling
    assert _check_downsampling_mismatch(False, 512, 2048) == (True, 512)


def check_model_equal(model1, model2):
    """Check whether two models are equal"""
    # Testing models are the same from loaded weights and downloaded from keras
    assert len(model1.layers) == len(model2.layers)

    for layer in range(len(model1.layers)):
        for array in range(len(model1.layers[layer].get_weights())):
            assert np.allclose(model1.layers[layer].get_weights()[array],
                               model2.layers[layer].get_weights()[array], atol=ATOL)


def test_initialize_model_weights_not_found():
    """Test error raised when the model can't find weights to load"""
    error_weight = 'htraenoytinutroppodnocesaevahtondideduti\losfosraeyderdnuhenootdenmednocsecar'
    try:
        assert not os.path.isfile(error_weight)
    except AssertionError:
        logging.error('Whoops, that mirage exists. '
                      'Change error_weight to a file path that does not exist.')

    with pytest.raises(IOError):
        _initialize_model('squeezenet', error_weight)


def test_initialize_model_bad_weights():
    """
    Test error raised when the model finds the weights file,
    but it's not the right format
    """
    bad_weights_file = open('bad_weights_test', 'w')
    bad_weights_file.write('this should fail')
    bad_weights_file.close()
    error_weight = 'bad_weights_test'

    try:
        with pytest.raises(IOError):
            _initialize_model('squeezenet', error_weight)
    finally:
        os.remove(error_weight)


def test_initialize_model_wrong_weights():
    """Test error raised when weights exist but don't match model"""
    squeeze_weight_path = 'pic2vec/saved_models/squeezenet_weights_tf_dim_ordering_tf_kernels.h5'
    assert os.path.isfile(squeeze_weight_path)

    with pytest.raises(ValueError):
        _initialize_model('vgg16', squeeze_weight_path)


INITIALIZE_MODEL_CASES = [
    ('squeezenet', 67, (1, 227, 227, 3)),
    ('vgg16', 23, (1, 224, 224, 3)),
    ('vgg19', 26, (1, 224, 224, 3)),
    ('resnet50', 176, (1, 224, 224, 3)),
    ('inceptionv3', 313, (1, 299, 299, 3)),
    ('xception', 134, (1, 299, 299, 3)),
]


@pytest.mark.parametrize('model_str, expected_layers, test_size',
                         INITIALIZE_MODEL_CASES, ids=MODELS)
def test_initialize_model(model_str, expected_layers, test_size):
    """Test the initializations of each model"""
    model = _initialize_model(model_str)

    if model_str == 'squeezenet':
        try:
            model_downloaded_weights = SqueezeNet()
        except Exception:
            raise AssertionError('Problem loading SqueezeNet weights.')
        check_model_equal(model, model_downloaded_weights)

    assert len(model.layers) == expected_layers

    # Create the test array to be predicted on
    test_array = np.zeros(test_size)

    # Pre-checked prediction
    check_prediction = np.load(INITIALIZE_ARRAY.format(model_str))

    # Check that each model predicts correctly to see if weights were correctly loaded
    assert np.allclose(model.predict_on_batch(test_array), check_prediction, atol=ATOL)
    del model


FEATURIZER_MODEL_DICT = dict.fromkeys(MODELS)
FEAT_CASES = [  # squeezenet
    (1, False, 128, 128, 'squeezenet'), (1, False, 0, 512, 'squeezenet'),
    (1, True, 0, 256, 'squeezenet'), (2, True, 0, 256, 'squeezenet'),
    (2, False, 128, 128, 'squeezenet'), (2, False, 0, 512, 'squeezenet'),
    (3, False, 96, 96, 'squeezenet'), (3, False, 0, 384, 'squeezenet'),
    (3, True, 0, 192, 'squeezenet'), (4, True, 0, 192, 'squeezenet'),
    (4, False, 96, 96, 'squeezenet'), (4, False, 0, 384, 'squeezenet'),

    # vgg16
    (1, False, 1024, 1024, 'vgg16'), (1, False, 0, 4096, 'vgg16'),
    (1, True, 0, 2048, 'vgg16'), (2, True, 0, 2048, 'vgg16'),
    (2, False, 1024, 1024, 'vgg16'), (2, False, 0, 4096, 'vgg16'),
    (3, False, 128, 128, 'vgg16'), (3, False, 0, 512, 'vgg16'),
    (3, True, 0, 256, 'vgg16'), (4, True, 0, 256, 'vgg16'),
    (4, False, 128, 128, 'vgg16'), (4, False, 0, 512, 'vgg16'),

    # vgg19
    (1, False, 1024, 1024, 'vgg19'), (1, False, 0, 4096, 'vgg19'),
    (1, True, 0, 2048, 'vgg19'), (2, True, 0, 2048, 'vgg19'),
    (2, False, 1024, 1024, 'vgg19'), (2, False, 0, 4096, 'vgg19'),
    (3, False, 128, 128, 'vgg19'), (3, False, 0, 512, 'vgg19'),
    (3, True, 0, 256, 'vgg19'), (4, True, 0, 256, 'vgg19'),
    (4, False, 128, 128, 'vgg19'), (4, False, 0, 512, 'vgg19'),

    # resnet50
    (1, False, 512, 512, 'resnet50'), (1, False, 0, 2048, 'resnet50'),
    (1, True, 0, 1024, 'resnet50'), (2, True, 0, 1024, 'resnet50'),
    (2, False, 512, 512, 'resnet50'), (2, False, 0, 2048, 'resnet50'),
    (3, False, 512, 512, 'resnet50'), (3, False, 0, 2048, 'resnet50'),
    (3, True, 0, 1024, 'resnet50'), (4, True, 0, 1024, 'resnet50'),
    (4, False, 512, 512, 'resnet50'), (4, False, 0, 2048, 'resnet50'),

    # inceptionv3
    (1, False, 512, 512, 'inceptionv3'), (1, False, 0, 2048, 'inceptionv3'),
    (1, True, 0, 1024, 'inceptionv3'), (2, True, 0, 1024, 'inceptionv3'),
    (2, False, 512, 512, 'inceptionv3'), (2, False, 0, 2048, 'inceptionv3'),
    (3, False, 512, 512, 'inceptionv3'), (3, False, 0, 2048, 'inceptionv3'),
    (3, True, 0, 1024, 'inceptionv3'), (4, True, 0, 640, 'inceptionv3'),
    (4, False, 320, 320, 'inceptionv3'), (4, False, 0, 1280, 'inceptionv3'),

    # xception
    (1, False, 512, 512, 'xception'), (1, False, 0, 2048, 'xception'),
    (1, True, 0, 1024, 'xception'), (2, True, 0, 512, 'xception'),
    (2, False, 256, 256, 'xception'), (2, False, 0, 1024, 'xception'),
    (3, False, 182, 182, 'xception'), (3, False, 0, 728, 'xception'),
    (3, True, 0, 364, 'xception'), (4, True, 0, 364, 'xception'),
    (4, False, 182, 182, 'xception'), (4, False, 0, 728, 'xception')
]


@pytest.mark.parametrize('depth, autosample, sample_size, expected_size, model_str', FEAT_CASES)
def test_build_featurizer(depth, autosample, sample_size, expected_size, model_str):
    """Test all of the model iterations"""
    if FEATURIZER_MODEL_DICT[model_str] is None:
        FEATURIZER_MODEL_DICT[model_str] = _initialize_model(model_str)

    model = build_featurizer(depth, autosample, sample_size,
                             model_str=model_str, loaded_model=FEATURIZER_MODEL_DICT[model_str])
    assert model.layers[-1].output_shape == (None, expected_size)
    del model


if __name__ == '__main__':
    test_decapitate_model()
    test_splice_layer()
    test_find_pooling_constant()
    test_downsample_model_features()
    test_initialize_model()
    test_build_featurizer()
