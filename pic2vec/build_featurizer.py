"""
This file deals with building the actual featurizer:
1. Initializing the InceptionV3 model
2. Decapitating it to the appropriate depth
3. Downsampling, if desired

The integrated function is the build_featurizer function, which takes the depth,
a flag signalling downsampling, and the number of features to downsample to.
"""

import logging
import os
import warnings

import trafaret as t
from keras.applications import InceptionV3, ResNet50, VGG16, VGG19, Xception
from keras.engine.topology import InputLayer
from keras.layers import GlobalAvgPool2D, Lambda, average
from keras.models import Model
import keras.backend as K

from .squeezenet import SqueezeNet

if K.backend() != 'tensorflow':
    logging.warn('Without a tensorflow backend, SqueezeNet and Xception will not be '
                 ' available. Please initialize ImageFeaturizer with either vgg16, vgg19, '
                 'resnet50, or inceptionv3.')

supported_model_types = {
    'squeezenet': {
        'label': 'SqueezeNet',
        'class': SqueezeNet,
        'kwargs': {'weights': None},
        'depth': {1: 5, 2: 12, 3: 19, 4: 26}
    },
    'inceptionv3': {
        'label': 'InceptionV3',
        'class': InceptionV3,
        'kwargs': {},
        'depth': {1: 2, 2: 19, 3: 33, 4: 50}
    },
    'vgg16': {
        'label': 'VGG16',
        'class': VGG16,
        'kwargs': {},
        'depth': {1: 1, 2: 2, 3: 4, 4: 8}
    },
    'vgg19': {
        'label': 'VGG19',
        'class': VGG19,
        'kwargs': {},
        'depth': {1: 1, 2: 2, 3: 4, 4: 9}
    },
    'resnet50': {
        'label': 'ResNet50',
        'class': ResNet50,
        'kwargs': {},
        'depth': {1: 2, 2: 5, 3: 13, 4: 23}
    },
    'xception': {
        'label': 'Xception',
        'class': Xception,
        'kwargs': {},
        'depth': {1: 1, 2: 8, 3: 18, 4: 28}
    }
}


@t.guard(model_str=t.Enum(*supported_model_types.keys()),
         loaded_weights=t.String(allow_blank=True))
def _initialize_model(model_str, loaded_weights=''):
    """
    Initialize the InceptionV3 model with the saved weights, or
    if the weight file can't be found, load them automatically through Keras.

    Parameters:
    ----------
        model_str : str
            String deciding which model to use for the featurizer

    Returns:
    -------
        model : keras.models.Model
            The initialized model loaded with pre-trained weights
    """
    logging.info('Loading/downloading {model_label} model weights. '
                 'This may take a minute first time.'
                 .format(model_label=supported_model_types[model_str]['label']))

    if loaded_weights != '':
        model = supported_model_types[model_str]['class'](weights=None)
        try:
            model.load_weights(loaded_weights)
        except IOError as err:
            logging.error('Problem loading the custom weights. If not an advanced user, please '
                          'leave loaded_weights unconfigured.')
            raise err
    else:
        model = supported_model_types[model_str]['class'](**supported_model_types
                                                          [model_str]['kwargs'])

        if model_str == 'squeezenet':
            # Special case for squeezenet - we already have weights for it
            this_dir, this_filename = os.path.split(__file__)
            model_path = os.path.join(this_dir,
                                      'saved_models',
                                      'squeezenet_weights_tf_dim_ordering_tf_kernels.h5')
            if not os.path.isfile(model_path):
                raise ValueError('Could not find the weights. Download another model'
                                 ' or replace the SqueezeNet weights in the model folder.')
            model.load_weights(model_path)

    logging.info('Model successfully initialized.')
    return model


@t.guard(model=t.Type(Model), depth=t.Int(gte=1))
def _decapitate_model(model, depth):
    """
    Cut off end layers of a model equal to the depth of the desired outputs,
    and then remove the links connecting the new outer layer to the old ones.

    Parameters:
    ----------
    model: keras.models.Model
        The model being decapitated. Note: original model is not changed, method returns new model.
    depth: int
        The number of layers to pop off the top of the network

    Returns:
    -------
    model: keras.models.Model
        Decapitated model.
    """
    # -------------- #
    # ERROR CHECKING #

    # Make sure the depth isn't greater than the number of layers (minus input)
    if depth >= len(model.layers) - 1:
        raise ValueError('Can\'t go deeper than the number of layers in the model. Tried to pop '
                         '{} layers, but model only has {}'.format(depth, len(model.layers) - 1))

    if not isinstance(model.layers[0], InputLayer):
        warnings.warn('First layer of the model is not an input layer. Beware of depth issues.')
    # -------------------------------------------------------- #

    # Get the intermediate output
    new_model_output = model.layers[(depth + 1) * -1].output
    new_model = Model(inputs=model.input, outputs=new_model_output)
    new_model.layers[-1].outbound_nodes = []
    return new_model


@t.guard(features=t.Any(), num_pooled_features=t.Int(gte=1))
def _find_pooling_constant(features, num_pooled_features):
    """
    Given a tensor and an integer divisor for the desired downsampled features,
    this will downsample the tensor to the desired number of features

    Parameters:
    ----------
    features : Tensor
        the layer output being downsampled
    num_pooled_features : int
        the desired number of features to downsample to

    Returns:
    -------
    int
        the integer pooling constant required to correctly splice the layer output for downsampling
    """
    # Initializing the outputs
    num_features = features.shape[-1].__int__()

    # Find the pooling constant
    pooling_constant = num_features / float(num_pooled_features)

    # -------------- #
    # ERROR CHECKING #

    if pooling_constant < 1:
        raise ValueError(
            'You can\'t downsample to a number bigger than the original feature space.')

    # Check that the number of downsampled features is an integer divisor of the original output
    if not pooling_constant.is_integer():
        # Store recommended downsample
        recommended_downsample = num_features / int(pooling_constant)
        raise ValueError('Trying to downsample features to non-integer divisor: '
                         'from {} to {}.\n\n Did you mean to downsample to'
                         ' {}? Regardless, please choose an integer divisor.'
                         .format(num_features, num_pooled_features, recommended_downsample))
    # -------------------------------------------------------- #
    # Cast the pooling constant back to an int from a float if it passes the tests
    return int(pooling_constant)


@t.guard(tensor=t.Any(), number_splices=t.Int(gte=1))
def _splice_layer(tensor, number_splices):
    """
    Splice a layer into a number of even slices through skipping. This downsamples the layer,
    and allows for operations to be performed over neighbors.

    Parameters:
    ----------
    layer: Tensor
        the layer output being spliced
    number_splices: int
        the number of new layers the original layer is being spliced into.
        NOTE: must be integer divisor of layer

    Returns:
    -------
    list_of_spliced_layers : list of Tensor
        a list of the spliced tensor sections of the original layer, with neighboring nodes
        occupying the same indices across splices
    """
    # -------------- #
    # ERROR CHECKING #
    # Need to check that the number of splices is an integer divisor of the feature
    # size of the layer
    num_features = tensor.shape[-1].__int__()
    if num_features % number_splices:
        raise ValueError('Number of splices needs to be an integer divisor of'
                         ' the number of features. Tried to split {} features into'
                         ' {} equal parts.'.format(num_features, number_splices))
    # ------------------------------------------ #
    # Split the tensor into equal parts by skipping nodes equal to the number
    # of splices. This allows for merge operations over neighbor features

    return [Lambda(lambda features: features[:, i::number_splices])(tensor) for i in
            range(number_splices)]


@t.guard(features=t.Any(), num_pooled_features=t.Int(gte=1))
def _downsample_model_features(features, num_pooled_features):
    """
    Take in a layer of a model, and downsample the layer to a specified size.

    Parameters:
    ----------
    features : Tensor
        the final layer output being downsampled
    num_pooled_features : int
        the desired number of features to downsample to

    Returns:
    -------
    downsampled_features : Tensor
        a tensor containing the downsampled features with size = (?, num_pooled_features)
    """
    # Find the pooling constant needed
    pooling_constant = _find_pooling_constant(features, num_pooled_features)
    # Splice the top layer into n layers, where n = pooling constant.
    list_of_spliced_layers = _splice_layer(features, pooling_constant)
    # Average the spliced layers to downsample
    downsampled_features = average(list_of_spliced_layers)
    return downsampled_features


def _check_downsampling_mismatch(downsample, num_pooled_features, output_layer_size):
    """
    If downsample is flagged True, but no downsampling size is given, then automatically
    downsample model. If downsample flagged false, but there is a size given, set downsample
    to true.

    Parameters:
    ----------
    downsample : bool
        Boolean flagging whether model is being downsampled
    num_pooled_features : int
        the desired number of features to downsample to
    output_layer_size : int
        number of nodes in the output layer being downsampled
    Returns:
    -------
    downsample : boolean
        Updated boolean flagging whether model is being downsampled
    num_pooled_features : int
        Updated number of features model output is being downsample to
    """
    # If num_pooled_features left uninitialized, and they want to downsample,
    # perform automatic downsampling
    if num_pooled_features == 0 and downsample:
        if output_layer_size % 2 == 0:
            num_pooled_features = output_layer_size // 2
            logging.warning('Automatic downsampling to {}. If you would like to set custom '
                            'downsampling, pass in an integer divisor of {} to '
                            'num_pooled_features.'.format(num_pooled_features, output_layer_size))
        else:
            raise ValueError('Sorry, no automatic downsampling available for this model.')

    # If they have initialized num_pooled_features, but not turned on
    # downsampling, downsample to what they entered
    elif num_pooled_features != 0 and not downsample:
        logging.info('Downsampling to {}.'.format(num_pooled_features))
        downsample = True

    return downsample, num_pooled_features


@t.guard(depth_of_featurizer=t.Int(gte=1, lte=4),
         downsample=t.Bool,
         num_pooled_features=t.Int(gte=0),
         model_str=t.Enum(*supported_model_types.keys()),
         loaded_model=t.Type(Model) | t.Null)
def build_featurizer(depth_of_featurizer, downsample, num_pooled_features=0,
                     model_str='squeezenet', loaded_model=None):
    """
    Create the full featurizer.

    Initialize the model, decapitate it to the appropriate depth, and check if downsampling
    top-layer featurization. If so, downsample to the desired feature space

    Parameters:
    ----------
    depth_of_featurizer : int
        How deep to cut the network. Can be 1, 2, 3, or 4.
    downsample : bool
        Boolean flagging whether to perform downsampling
    num_pooled_features : int
        If we downsample, integer determining how small to downsample.
        NOTE: Must be integer divisor of original number of features
        or 0 if we don't want to specify exact number
    model_str : str
        String deciding which model to use for the featurizer
    loaded_model : keras.models.Model, optional
        If specified - use the model for featurizing, istead of creating new one.

    Returns:
    -------
    model: keras.models.Model
        The decapitated, potentially downsampled, pre-trained image featurizer.
        With no downsampling, the output features are equal to the top densely-
        connected layer of the network, which depends on the depth of the model.
        With downsampling, the output is equal to a downsampled average of
        multiple splices of the last densely connected layer.
    """
    # BUILDING INITIAL MODEL #
    if loaded_model is not None:
        model = loaded_model
    else:
        model = _initialize_model(model_str=model_str)

    # DECAPITATING MODEL #
    # Find the right depth from the dictionary and decapitate the model
    model = _decapitate_model(model, supported_model_types[model_str]['depth'][depth_of_featurizer])
    model_output = model.layers[-1].output
    # Add pooling layer to the top of the now-decapitated model as the featurizer,
    # if it needs to be downsampled
    if len(model.layers[-1].output_shape) > 2:
        model_output = GlobalAvgPool2D(name='featurizer')(model_output)

    # Save the model output
    num_output_features = model_output.shape[-1].__int__()
    logging.info("Model decapitated.")

    # DOWNSAMPLING FEATURES #
    # Checking that the user's downsampling flag matches the initialization of the downsampling
    (downsample, num_pooled_features) = _check_downsampling_mismatch(downsample,
                                                                     num_pooled_features,
                                                                     num_output_features)

    # If we are downsampling the features, we add a pooling layer to the outputs
    # to bring it to the correct size.
    if downsample:
        model_output = _downsample_model_features(model_output, num_pooled_features)
    logging.info("Model downsampled.")

    # Finally save the model
    model = Model(inputs=model.input, outputs=model_output)
    logging.info("Full featurizer is built.")
    if downsample:
        logging.info("Final layer feature space downsampled to {}".format(num_pooled_features))
    else:
        logging.info("No downsampling. Final layer feature space has size {}"
                     .format(num_output_features))

    return model
