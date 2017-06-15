# -*- coding: utf-8 -*-

"""Main module."""
from keras.applications.inception_v3 import InceptionV3


class ImageFeaturizer:
    '''
    This object can load images, rescale, crop, and vectorize them into a
    uniform batch, and then featurize the images for use with custom classifiers.
    '''

    def __init__(self,
                images_files = None,
                model_string = "InceptionV3"
                scaled_size = (299, 299),
                crop_size = (299, 299),
                number_crops = 1,
                random_crop = False
                isotropic_scaling = True,
                top_layer = True,
                pool_features = False
                num_pooled_features = 1024
                ):

        # The pretrained models that can be loaded through Keras. The InceptionV3
        # network is loaded into this package– using any others will automatically
        # load the weights online.
        list_of_possible_models = ['InceptionV3', 'Xception', 'VGG16', 'VGG19',
                                   'ResNet50']

        # A dictionary of the boolean set for error-checking
        dict_of_booleans = {'random_crop': random_crop, 'isotropic_scaling': isotropic_scaling
                            'top_layer': top_layer, 'pool_features': pool_features}

        # Type checking
        if image_files == None:
            raise ValueError('Image files required.')

        if not isinstance(model_string, str):
            raise ValueError('model_string is not a string! Please pass a string \
                            of one of the available models: \
                            InceptionV3, Xception, VGG16, VGG19, or ResNet50.')

        if not model_string in list_of_possible_models:
            raise ValueError('model_string is not one of the available models: \
                            InceptionV3, Xception, VGG16, VGG19, or ResNet50. \
                            Please reformulate string')

        if not isinstance(scaled_size, tuple):
            raise ValueError('scaled_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(crop_size, tuple):
            raise ValueError('crop_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(number_crops, int):
            raise ValueError('number_crops is not an integer! Please specify the \
                            number of random crops you would like to average')

        for key in dict_of_booleans:
            if not isinstance(dict_of_booleans[key], bool):
                raise ValueError(key + ' is not a boolean! Please set to True or False, \
                                 or leave blank for default configuration')


        # Create the model!
        model = InceptionV3(weights=None)
        model.load_weights('../model/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')


        # If top_layer is set to True, we just use the top layer of the network
        if top_layer:
            # Pop off the last layer and get rid of the links
            model.layers.pop()
            model.outputs = [model.layers[-1].output]
            model.layers[-1].outbound_nodes = []

        # Otherwise, we use a deeper layer as the output
        # TODO: Should we be able to decide which deep layer?
        else:
            x

        # If we are pooling the features, we add a pooling layer to the outputs
        # to bring it to the correct size. TODO: Should correct size be customizable,
        # or just left at 1024?

        if pool_features:
            x



        # Images
        self.image_files = image_files
        self.vectorized_images = vectorized_images
        self.processed_images=  image_files

        # Model type
        self.model_string = model_string

        # Image scaling and cropping
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        self.number_crops = number_crops
        self.isotropic_scaling = isotropic_scaling

        # Save the model
        self.model = model
