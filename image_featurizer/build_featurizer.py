from keras.applications.inception_v3 import InceptionV3
import os
from keras.layers import Lambda
from keras.layers.merge import average
import pkg_resources
import time
from keras.models import Model
from keras.datasets import cifar10


def decapitate_model(model, depth):
    '''
    This cuts off end layers of a model equal to the depth of the desired outputs,
    and then removes the links connecting the new outer layer to the old ones.

    ## Parameters: ###
        model: The model being decapitated
        depth: The number of layers to pop off the top of the network

    ### Output: ###
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
        raise ValueError('Can\'t go deeper than the number of layers in the model!' +
                         ' Tried to pop ' + str(depth) + 'layers, but model only has ' + str(len(model.layers)-1))
    #------------------------------------------------#


    # Pop the layers
    for layer in range(depth):
        model.layers.pop()

    # Break the connections
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []


def splice_layer(tensor, number_splices):
    '''
    This helper function takes a layer, and splices it into a number of
    even slices through skipping. This downsamples the layer, and allows for
    operations to be performed over neighbors.

    ### Parameters: ###
        layer: the layer being spliced

        number_splices: the number of new layers the original layer is being spliced into.
                        NOTE: must be integer divisor of layer

    ### Output: ###
        list_of_spliced_layers: a list of the spliced sections of the original
                                layer, with neighboring nodes occupying the same
                                indices across  splices
    '''
    # Initializing list of spliced layers
    list_of_spliced_layers=[]

    #------------------------------------------------#
    ### ERROR CHECKING ###

    # Need to check that the number of splices is an integer divisor of the feature
    # size of the layer
    num_features = tensor.shape[-1].__int__()

    if not number_splices == int(number_splices):
        raise ValueError('Must have integer number of splices! Trying to splice into ' +
                         str(number_splices) + ' parts.')

    if not num_features % number_splices == 0:
        raise ValueError('Number of splices needs to be an integer divisor of' +
                         ' the number of features! Tried to split ' + str(num_features)
                         + ' features into ' + str(number_splices) + ' equal parts.')
    #------------------------------------------------#


    # Split the tensor into equal parts by skipping nodes equal to the number
    # of splices. This allows for merge operations over neighbor features

    for i in range(number_splices):
        spliced_output = Lambda(lambda features: features[:, i::number_splices])(tensor)
        list_of_spliced_layers.append(spliced_output)

    return list_of_spliced_layers


def find_pooling_constant(features, num_pooled_features):
    # Initializing the outputs
    output_shape = features.shape
    num_features = output_shape[-1].__int__()

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

        raise ValueError('Trying to downsample features to non-integer divisor: from ' +
                         str(num_features)+ ' features to ' + str(num_pooled_features) + '.' +
                         ' \n \n We recommend you downsample to: ' + str(recommended_downsample))
    #------------------------------------------------#

    # Cast the pooling constant back to an int from a float if it passes the tests
    return int(pooling_constant)


def downsample_model_features(features, num_pooled_features):
    '''
    This takes in a layer of a model, and downsamples layer to a specified size

    ### Parameters: ###
        features: the layer being downsampled

        size_of_downsample: the size that the features are being downsampled to

    ### Output: ###
        downsampled_features: a tensor containing the downsampled features with
                              size = (?, num_pooled_features)
    '''

    # Find the pooling constant needed
    pooling_constant = find_pooling_constant(features, num_pooled_features)

    # Splice the top layer into n layers, where n = pooling constant.
    list_of_spliced_layers = splice_layer(features, pooling_constant)

    # Average the spliced layers to downsample!
    downsampled_features = average(list_of_spliced_layers)

    return downsampled_features

def initialize_model():
    this_dir, this_filename = os.path.split(__file__)
    model_path = os.path.join(this_dir, "model", "inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    # Initialize the model. If weights are already downloaded, pull them.
    print model_path

    if os.path.isfile(model_path):
        model = InceptionV3(weights=None)
        model.load_weights(model_path)
        print "\n \nModel initialized and weights loaded successfully!"

    # Otherwise, download them automatically
    else:
        print "Can't find weight file. Need to download weights from Keras!"
        model = InceptionV3()
        print "Model successfully initialized."

    return model

def build_featurizer(depth_of_featurizer, downsample_features, num_pooled_features):
    '''
from image_featurizer.model import build_featurizer
model = build_featurizer(1, False, 1024)
    '''
    ### INITIALIZING MODEL ###
    this_dir, this_filename = os.path.split(__file__)
    model_path = os.path.join(this_dir, "model", "inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    # Initialize the model. If weights are already downloaded, pull them.
    print model_path

    if os.path.isfile(model_path):
        model = InceptionV3(weights=None)
        model.load_weights(model_path)
        print "\n \nModel initialized and weights loaded successfully!"

    # Otherwise, download them automatically
    else:
        print "Can't find weight file. Need to download weights from Keras!"
        model = InceptionV3()
        print "Model successfully initialized."


    ### DECAPITATING MODEL ###

    # Choosing model depth:
    depth_to_number_of_layers = {1: 1, 2: 19, 3: 33, 4:50}

    # Find the right depth from the dictionary and decapitate the model
    decapitated_layers = depth_to_number_of_layers[depth_of_featurizer]
    decapitate_model(model, decapitated_layers)

    # If depth is 1, we don't add a pool because it's already there.
    if depth_of_featurizer != 1:
        model_output = GlobalAveragePooling2D(name='avg_pool')(model.layers[-1].output)

    # Save the model output
    model_output = model.layers[-1].output
    num_output_features = model_output.shape[-1].__int__()

    # Check that the model's output shape = (None, number_of_features)
    if not model.layers[-1].output_shape == (None, num_output_features):
        raise ValueError('Something wrong with output! Should have shape: (None, '
                        + str(num_pooled_features)+ ')' + '. Actually has shape: '
                        + str(model.layers[-1].output_shape))


    ### DOWNSAMPLING FEATURES ###

    # If we are downsampling the features, we add a pooling layer to the outputs
    # to bring it to the correct size.
    if downsample_features:
        model_output = downsample_model_features(model, num_pooled_features)

    # Finally save the model! Input is the same as usual.
    # With no downsampling, output is equal to the last layer, which depends
    # on the depth of the model. WITH downsampling, output is equal to a
    # downsampled average of multiple splices of the last layer.
    model = Model(input=model.input, output=model_output)
    return model
