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
                scaled_size = (299, 299),
                crop_size = (299, 299),
                number_crops = 1,
                random_crop = False
                isotropic_scaling = True,
                top_layer = True,
                depth_of_output = 2,
                downsample_features = False
                num_pooled_features = 1024
                ):


        #### TYPE CHECKING ####

        # A dictionary of the boolean set for error-checking
        dict_of_booleans = {'random_crop': random_crop, 'isotropic_scaling': isotropic_scaling
                            'top_layer': top_layer, 'downsample_features': downsample_features}

        if image_files == None:
            raise ValueError('Image files required.')

        if not isinstance(scaled_size, tuple):
            raise ValueError('scaled_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(crop_size, tuple):
            raise ValueError('crop_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(number_crops, int):
            raise ValueError('number_crops is not an integer! Please specify the \
                            number of random crops you would like to average')

        if not isinstance(depth_of_output, int)
            raise ValueError('depth_of_output is not an integer! Please specify the \
                            number of layers you would like to remove from the top \
                            of the network for featurization. Can be between 1 and 4.')

        for key in dict_of_booleans:
            if not isinstance(dict_of_booleans[key], bool):
                raise ValueError(key + ' is not a boolean! Please set to True or False, \
                                 or leave blank for default configuration')

        if not isinstance(num_pooled_features, int):
                raise ValueError('num_pooled_features is not an integer! Please set \
                                to an integer. Recommended value is 1024, but can \
                                be set to any integer divisor of the number of \
                                unpooled features.')


        #### HELPER FUNCTIONS ####

        def decapitate_model(model, depth):
            '''
            This is a method that cuts off end layers of a model equal to the depth
            of the desired outputs, and then removes the links connecting the new
            outer layer to the old ones.

            ## Parameters: ###
                model: The model being decapitated

                depth: The number of layers to pop off the top of the network
            '''

            # Pop the layers
            for layer in range(depth):
                model.layers.pop()

            # Break the connections
            model.outputs = [model.layers[-1].output]

            model.layers[-1].outbound_nodes = []


        def downsample_features(layer, size_of_downsample):



        #### BUILDING THE MODEL ####

        # Initialize the model
        model = InceptionV3(weights=None)
        model.load_weights('../model/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
        model_input = model.input

        # Choosing model depth:
        # If top_layer is set to True or depth is 1, we just use the top layer of the network
        if top_layer or depth_of_output==1:
            # Output layer here should already be (None, 2048)
            decapitate(model, 1)

        # Otherwise, we decapitate the model to the appropriate layers
        elif depth_of_output == 2:
            decapitate_model(model, 19)
            x = model.layers[-1].output

            # Output layer after global average should be (None, 2048)
            x = GlobalAveragePooling2D(name='avg_pool')(x)


        elif depth_of_output == 3:
            decapitate_model(model, 33)
            x = model.layers[-1].output

            # Output layer after global average should be (None, 2048)
            x = GlobalAveragePooling2D(name='avg_pool')(x)

        elif depth_of_output == 4:
            decapitate_model(model, 50)
            x = model.layers[-1].output

            # The output after global average should be (None, 1280)
            x = GlobalAveragePooling2D(name='avg_pool')(x)


        model = Model(input=model_input, output=x)
        # Check output length
        if not len(model.layers[-1].output_shape) == 2:
            raise ValueError('Something wrong with output! Should be a tuple of \
                            length 2, with the second value being equal to the \
                            number of features desired. It is not of length 2.')


        # If we are pooling the features, we add a pooling layer to the outputs
        # to bring it to the correct size.
        if downsample_features:
            shape_out = model.layers[-1].output_shape

            # Check that the number of downsampled features is an integer divisor
            # of the original output
            if not isinstance(shape_out[1]/num_pooled_features, int):
                raise ValueError('The desired number of pooled features is not an \
                                integer divisor of the regular output! Output shape \
                                = ' + str(shape_out[1]) + '. The desired features \
                                = '+ str(num_pooled_features) + '.')

            # Find the pooling constant: I.E. If we want half the number of features,
            # pooling constant will be 2.
            pooling_constant = shape_out[1]/num_pooled_features



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
