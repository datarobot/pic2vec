"""
This file deals with building the actual featurizer:
1. Initializing the InceptionV3 model
2. Decapitating it to the appropriate depth
3. Downsampling, if desired

The integrated function is the build_featurizer function, which takes the depth,
a flag signalling downsampling, and the number of features to downsample to.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.applications.inception_v3 import InceptionV3  # noqa: E402
from keras.layers import GlobalAvgPool2D, Lambda, average  # noqa: E402
from keras.models import Model  # noqa: E402


def _initialize_model(model_str):
    """
    This function initializes the InceptionV3 model with the saved weights, or
    if it can't find the weight file, it loads them automatically through Keras.

    Parameters:
    ----------
        model_str : str
            String deciding which model to use for the featurizer

    Returns:
    -------
        model : keras.model.Model
            The initialized model loaded with pre-trained weights
    """

    # -------------- #
    # ERROR CHECKING #
    valid_models = ['squeezenet', 'inceptionv3', 'vgg16', 'vgg19', 'resnet50', 'xception']

    if not isinstance(model_str, str):
        raise TypeError('model_str must be a string!')


    if model_str.lower() not in valid_models:
        raise TypeError('model_str must be one of the following model names:'
                        ' squeezenet, inceptionv3, vgg16, vgg19, resnet50, xception')
    # ---------------------------- #

    # Special case for Squeezenet, because we have the weight file in the package
    if model_str.lower() == 'squeezenet':

        # Create path to the saved model
        this_dir, this_filename = os.path.split(__file__)
        model_path = os.path.join(this_dir, "model", "squeezenet_weights_tf_dim_ordering_tf_kernels.h5")

        # Initialize the Squeezenet model. If weights are already downloaded, pull them.
        if os.path.isfile(model_path):
            model = SqueezeNet(weights=None)
            model.load_weights(model_path)
            print("\nModel initialized and weights loaded successfully!")

        # Otherwise, download them automatically
        else:
            raise ValueError('Could not find the weights! Download another model' \
                             ' or replace the SqueezeNet weights in the model folder.')

    # Initializing all the other models automatically from Keras weights
    elif model_str.lower() == 'vgg16':
        print('Need to download VGG16 weights from Keras!')
        model = VGG16()
        print("\nModel successfully initialized.")

    elif model_str.lower() == 'vgg19':
        print('Need to download VGG19 weights from Keras!')
        model = VGG19()
        print("\nModel successfully initialized.")

    elif model_str.lower() == 'resnet50':
        print('Need to download ResNet50 weights from Keras!')
        model = ResNet50()
        print("\nModel successfully initialized.")

    elif model_str.lower() == 'inceptionv3':
        print('Need to download InceptionV3 weights from Keras!')
        model = InceptionV3()
        print("\nModel successfully initialized.")

    elif model_str.lower() == 'xception':
        print('Need to download Xception weights from Keras!')
        model = Xception()
        print("\nModel successfully initialized.")


    return model


def _decapitate_model(model, depth):
    """
    This cuts off end layers of a model equal to the depth of the desired outputs,
    and then removes the links connecting the new outer layer to the old ones.
``
    Parameters:
    ----------
        model: The model being decapitated
        depth: The number of layers to pop off the top of the network

    Returns:
    -------
        No output. This function operates on the model directly.
    """

    # -------------- #
    # ERROR CHECKING #

    # Make sure they actually passed a keras model
    if not isinstance(model, Model):
        raise TypeError('Please pass a model to the function. This is not a model.')

    # Make sure depth is an integer
    if not isinstance(depth, int):
        raise TypeError('Depth is not an integer! Must have integer decapitation depth.')

    # Make sure the depth isn't greater than the number of layers (minus input)
    if depth >= len(model.layers)-1:
        raise ValueError('Can\'t go deeper than the number of layers in the model! Tried to pop '
                         '{} layers, but model only has {}'.format(depth, len(model.layers) - 1))

    if not isinstance(model.layers[0], InputLayer):
        warnings.warn('First layer of the model is not an input layer. Beware of depth issues.')
    # -------------------------------------------------------- #

    # Get the intermediate output
    new_model_output = model.layers[(depth+1) * -1].output

    new_model = Model(inputs=model.input, outputs=new_model_output)
    new_model.layers[-1].outbound_nodes = []

    return new model

def _find_pooling_constant(features, num_pooled_features):
    """
    Given a tensor and an integer divisor for the desired downsampled features,
    this will downsample the tensor to the desired number of features

    Parameters:
    ----------
    features: the tensor to be downsampled
    num_pooled_features: the desired number of features to downsample to

    Returns:
    -------
    int(pooling_constant): the integer pooling constant required to correctly
                           splice the tensor layer for downsampling
    """
    # Initializing the outputs
    output_shape = features.shape
    num_features = output_shape[-1].__int__()

    if not num_pooled_features:
        raise ValueError('Can\'t pool to zero! Something wrong with parents functions-'
                         ' should not be able to pass 0 to this function. Check traceback.')

    # Find the pooling constant
    pooling_constant = num_features / float(num_pooled_features)

    # -------------- #
    # ERROR CHECKING #
    if not isinstance(num_pooled_features, int):
        raise TypeError('Number of features after pooling has to be an integer!')

    # Throw an error if they try to "downsample" up
    if pooling_constant < 1:
        raise ValueError(
            'You can\'t downsample to a number bigger than the original feature space!')

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


def _splice_layer(tensor, number_splices):
    """
    This helper function takes a layer, and splices it into a number of
    even slices through skipping. This downsamples the layer, and allows for
    operations to be performed over neighbors.

    Parameters:
    ----------
        layer: the layer being spliced

        number_splices: the number of new layers the original layer is being spliced into.
                        NOTE: must be integer divisor of layer

    Returns:
    -------
        list_of_spliced_layers: a list of the spliced sections of the original
                                layer, with neighboring nodes occupying the same
                                indices across  splices
    """

    # -------------- #
    # ERROR CHECKING #
    # Need to check that the number of splices is an integer divisor of the feature
    # size of the layer
    num_features = tensor.shape[-1].__int__()

    if not isinstance(number_splices, int):
        raise ValueError('Must have integer number of splices! Trying to splice into '
                         '{} parts.'.format(number_splices))

    if num_features % number_splices:
        raise ValueError('Number of splices needs to be an integer divisor of'
                         ' the number of features! Tried to split {} features into'
                         ' {} equal parts.'.format(num_features, number_splices))
    # ------------------------------------------ #

    # Split the tensor into equal parts by skipping nodes equal to the number
    # of splices. This allows for merge operations over neighbor features
    return [Lambda(lambda features: features[:, i::number_splices])(tensor) for i in
            xrange(number_splices)]


def _downsample_model_features(features, num_pooled_features):
    """
    This takes in a layer of a model, and downsamples layer to a specified size

    Parameters:
    ----------
        features: the layer being downsampled

        size_of_downsample: the size that the features are being downsampled to

    Returns:
    -------
        downsampled_features: a tensor containing the downsampled features with
                              size = (?, num_pooled_features)
    """

    # Find the pooling constant needed
    pooling_constant = _find_pooling_constant(features, num_pooled_features)

    # Splice the top layer into n layers, where n = pooling constant.
    list_of_spliced_layers = _splice_layer(features, pooling_constant)

    # Average the spliced layers to downsample!
    downsampled_features = average(list_of_spliced_layers)

    return downsampled_features

def _check_downsampling_mismatch(downsample, num_pooled_features, output_layer_size):

    # If num_pooled_features left uninitialized, and they want to downsample,
    # perform automatic downsampling
    if num_pooled_features == 0 and downsample:
        if output_layer_size % 2 == 0:
            num_pooled_features = output_layer_size // 2
            print('Automatic downsampling to {}. If you would like to set custom '
                  'downsampling, pass in an integer divisor of {} to '
                  'num_pooled_features!'.format(num_pooled_features,output_layer_size))
        else:
            raise ValueError('Sorry, no automatic downsampling available for this model!')

    # If they have initialized num_pooled_features, but not turned on
    # downsampling, downsample to what they entered!
    elif num_pooled_features != 0 and not downsample:
        print('\n \n Downsampling to {}.'.format(num_pooled_features))
        downsample = True

    return downsample, num_pooled_features


def build_featurizer(depth_of_featurizer, downsample, num_pooled_features,
                     model_str='squeezenet', loaded_model=None):
    """
    Create the full featurizer:
        Initialize the model
        Decapitate it to the appropriate depth
        Check if downsampling top-layer featurization
        If so, downsample to the desired feature space

    Parameters:
    ----------
        depth_of_featurizer: How deep to cut the network. Can be 1, 2, 3, or 4.

        downsample: Boolean indicating whether to perform downsampling

        num_pooled_features: If we downsample, integer determining how small to downsample.
                             NOTE: Must be integer divisor of original number of features

    Returns:
    -------
        model: The decapitated, potentially downsampled, pre-trained image featurizer.

               With no downsampling, the output features are equal to the top densely-
               connected layer of the network, which depends on the depth of the model.

               With downsampling, the output is equal to a downsampled average of
               multiple splices of the last densely connected layer.
    """

    if not (isinstance(loaded_model, Model) or isinstance(loaded_model, type(None))):
        raise TypeError('loaded_model is only for testing functionality. ' \
                        'Needs to be either a Model or None type.' )

    ### BUILDING INITIAL MODEL ###
    if loaded_model != None:
        model = loaded_model

    else:
        model = _initialize_model(model_str=model_str)

    ### DECAPITATING MODEL ###

    # Choosing model depth:
    squeezenet_dict = {1: 5, 2: 12, 3: 19, 4: 26}
    vgg16_dict = {1: 1, 2: 2, 3: 4, 4: 8}
    vgg19_dict = {1: 1, 2: 2, 3: 4, 4: 9}
    resnet50_dict = {1: 2, 2: 5, 3: 13, 4: 23}
    inceptionv3_dict = {1: 2, 2: 19, 3: 33, 4: 50}
    xception_dict = {1: 1, 2: 8, 3: 18, 4: 28 }

    depth_dict = {'squeezenet': squeezenet_dict, 'vgg16': vgg16_dict, 'vgg19':\
                   vgg19_dict, 'resnet50': resnet50_dict, 'inceptionv3': \
                   inceptionv3_dict, 'xception': xception_dict}

    # Find the right depth from the dictionary and decapitate the model
    model = _decapitate_model(model, depth_dict[model_str][depth_of_featurizer])
    model_output = model.layers[-1].output
    # Add pooling layer to the top of the now-decapitated model as the featurizer,
    # if it needs to be downsampled
    if len(model.layers[-1].output_shape) > 2:
        model_output = GlobalAvgPool2D(name='featurizer')(model_output)

    # Save the model output
    num_output_features = model_output.shape[-1].__int__()
    print("Model decapitated!")

    # DOWNSAMPLING FEATURES #

    # Checking that the user's downsampling flag matches the initialization of the downsampling
    (downsample, num_pooled_features) = _check_downsampling_mismatch(downsample,
                                                                     num_pooled_features,
                                                                     num_output_features)

    # If we are downsampling the features, we add a pooling layer to the outputs
    # to bring it to the correct size.
    if downsample:
        model_output = _downsample_model_features(model_output, num_pooled_features)

    print("Model downsampled!")

    # Finally save the model!
    model = Model(inputs=model.input, outputs=model_output)

    print("Full featurizer is built!")
    if downsample:
        print("Final layer feature space downsampled to " + str(num_pooled_features))
    else:
        print("No downsampling! Final layer feature space has size " + str(num_output_features))

    return model
