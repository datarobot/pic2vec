import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import GlobalAvgPool2D, Lambda, average


def _initialize_model():
    '''
    This function initializes the InceptionV3 model with the saved weights, or
    if it can't find the weight file, it loads them automatically through Keras.

    Parameters:
    ----------
        None

    Returns:
    -------
        model: The initialized InceptionV3 model loaded with pre-trained weights
    '''

    # Create path to the saved model
    this_dir, this_filename = os.path.split(__file__)
    model_path = os.path.join(this_dir, "model", "inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    # Initialize the model. If weights are already downloaded, pull them.
    if os.path.isfile(model_path):
        model = InceptionV3(weights=None)
        model.load_weights(model_path)
        print("\nModel initialized and weights loaded successfully!")

    # Otherwise, download them automatically
    else:
        print('\nCan\'t find weight file. Need to download weights from Keras!')
        model = InceptionV3()
        print("\nModel successfully initialized.")

    return model


def _decapitate_model(model, depth):
    '''
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
    '''

    #------------------------------------------------#
    ### ERROR CHECKING ###
    # Make sure they actually passed a keras model
    if not isinstance(model, Model):
        raise TypeError('Please pass a model to the function. This is not a model.')

    # Make sure depth is an integer
    if not isinstance(depth, int):
        raise TypeError('Depth is not an integer! Must have integer decapitation depth.')

    # Make sure the depth isn't greater than the number of layers (minus input)
    if depth >= len(model.layers):
        raise ValueError('Can\'t go deeper than the number of layers in the model!' \
                         ' Tried to pop {} layers, but model only has {}'.format(depth, len(model.layers)-1))
    #------------------------------------------------#


    # Pop the layers
    for layer in xrange(depth):
        model.layers.pop()

    # Break the connections
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []



def _find_pooling_constant(features, num_pooled_features):
    '''
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
    '''
    # Initializing the outputs
    output_shape = features.shape
    num_features = output_shape[-1].__int__()

    if not num_pooled_features:
        raise ValueError('Can\'t pool to zero! Something wrong with parents functions-' \
                         ' should not be able to pass 0 to this function. Check traceback.')

    # Find the pooling constant
    pooling_constant = num_features/float(num_pooled_features)

    #------------------------------------------------#
    ### ERROR CHECKING ###

    if not isinstance(num_pooled_features,int):
        raise TypeError('Number of features after pooling has to be an integer!')

    # Throw an error if they try to "downsample" up
    if pooling_constant < 1:
        raise ValueError('You can\'t downsample to a number bigger than the original feature space!')


    # Check that the number of downsampled features is an integer divisor of the original output
    if not pooling_constant.is_integer():

        # Store recommended downsample
        recommended_downsample = num_features/int(pooling_constant)

        raise ValueError('Trying to downsample features to non-integer divisor: ' \
                         'from {} to {}.\n\n Did you mean to downsample to' \
                         ' {}? Regardless, please choose an integer divisor.'\
                         .format(num_features, num_pooled_features, recommended_downsample))
    #------------------------------------------------#

    # Cast the pooling constant back to an int from a float if it passes the tests
    return int(pooling_constant)

def _splice_layer(tensor, number_splices):
    '''
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
    '''

    #------------------------------------------------#
    ### ERROR CHECKING ###

    # Need to check that the number of splices is an integer divisor of the feature
    # size of the layer
    num_features = tensor.shape[-1].__int__()

    if not isinstance(number_splices, int):
        raise ValueError('Must have integer number of splices! Trying to splice into ' \
                         '{} parts.'.format(number_splices))

    if num_features % number_splices:
        raise ValueError('Number of splices needs to be an integer divisor of' \
                         ' the number of features! Tried to split {} features into' \
                         ' {} equal parts.'.format(num_features, number_splices))
    #------------------------------------------------#


    # Split the tensor into equal parts by skipping nodes equal to the number
    # of splices. This allows for merge operations over neighbor features

    return [Lambda(lambda features: features[:, i::number_splices])(tensor) for i in xrange(number_splices)]


def _downsample_model_features(features, num_pooled_features):
    '''
    This takes in a layer of a model, and downsamples layer to a specified size

    Parameters:
    ----------
        features: the layer being downsampled

        size_of_downsample: the size that the features are being downsampled to

    Returns:
    -------
        downsampled_features: a tensor containing the downsampled features with
                              size = (?, num_pooled_features)
    '''

    # Find the pooling constant needed
    pooling_constant = _find_pooling_constant(features, num_pooled_features)

    # Splice the top layer into n layers, where n = pooling constant.
    list_of_spliced_layers = _splice_layer(features, pooling_constant)

    # Average the spliced layers to downsample!
    downsampled_features = average(list_of_spliced_layers)

    return downsampled_features

def _check_downsampling_mismatch(downsample, num_pooled_features, depth):

    # If num_pooled_features left uninitialized, and they want to downsample
    # perform automatic downsampling
    if num_pooled_features == 0 and downsample == True:
        if depth == 4:
            temp_features=1280
            num_pooled_features = 640
        else:
            temp_features = 2048
            num_pooled_features = 1024

        print('Automatic downsampling to {}. If you would like to set custom '
              'downsampling, pass in an integer divisor of {} to '
              'num_pooled_features!'.format(num_pooled_features,temp_features))

    # If they have initialized num_pooled_features, but not turned on
    # downsampling, check that they don't actually want to downsample!
    elif num_pooled_features != 0 and downsample == False:
        print('\n \n Downsampling to {}.'.format(num_pooled_features))
        downsample = True

    return (downsample, num_pooled_features)

def build_featurizer(depth_of_featurizer, downsample, num_pooled_features):
    '''
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
    '''

    ### BUILDING INITIAL MODEL ###
    model = _initialize_model(target_size)
    ### DECAPITATING MODEL ###

    # Choosing model depth:
    depth_to_number_of_layers = {1: 2, 2: 19, 3: 33, 4:50}

    # Find the right depth from the dictionary and decapitate the model
    _decapitate_model(model, depth_to_number_of_layers[depth_of_featurizer])

    # Add pooling layer to the top of the now-decapitated model as the featurizer
    out = GlobalAvgPool2D(name='featurizer')(model.layers[-1].output)
    model = Model(inputs=model.input, outputs=out)

    # Save the model output
    model_output = model.layers[-1].output
    num_output_features = model_output.shape[-1].__int__()
    print("Model decapitated!")




    ### DOWNSAMPLING FEATURES ###

    # Checking that the user's downsampling flag matches the initialization of the downsampling
    (downsample, num_pooled_features) = _check_downsampling_mismatch(downsample, num_pooled_features, depth_of_featurizer)

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
