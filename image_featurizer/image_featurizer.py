# -*- coding: utf-8 -*-

"""Main module."""
from keras.applications.inception_v3 import InceptionV3
from keras.layers.merge import average

class ImageFeaturizer:
    '''
    This object can load images, rescale, crop, and vectorize them into a
    uniform batch, and then featurize the images for use with custom classifiers.
    '''


    def __init__(self,
                images_files = None,
                scaled_size = (299, 299),
                crop_size = (299, 299),
                number_crops = 0,
                random_crop = False,
                isotropic_scaling = True,
                depth_of_featurizer = 1,
                downsample_features = False,
                num_pooled_features = 1024
                ):

    '''
    Initializer:

    Loads an initial InceptionV3 pretrained network, decapitates it and downsamples
    according to user specifications. Loads image file and csv, and stores model
    to perform image featurization.

    ### Parameters: ###
        images_files: The file containing the images

        scaled_size: The size that the images get scaled to

        crop_size: If the image gets cropped, decides the size of the crop

        number_crops: If 0, no cropping. Otherwise, the number of crops taken of the image

        random_crop: If False, only take the center crop. If True, take random crops

        isotropic_scaling: If False, the image is scaled non-proportionally. If true,
                       image is scaled keeping proportions, and then cropped to correct size

        depth_of_featurizer: How many layers deep we're taking the model

        downsample_features: If True, feature layer is downsampled

        num_pooled_features: If feature layer is downsampled, chooses number of features
                            to downsample it to
    '''

        #------------------------------------------------#
                    ### ERROR CHECKING ###
        # A dictionary of the boolean set for error-checking
        dict_of_booleans = {'random_crop': random_crop, 'isotropic_scaling': isotropic_scaling,
                            'downsample_features': downsample_features}

        if image_files == None:
            raise ValueError('Image files required.')

        if not isinstance(scaled_size, tuple):
            raise TypeError('scaled_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(crop_size, tuple):
            raise TypeError('crop_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(number_crops, int):
            raise TypeError('number_crops is not an integer! Please specify the' +
                            ' number of random crops you would like to average')

        if not isinstance(depth_of_featurizer, int)
            raise TypeError('depth_of_featurizer is not an integer! Please' +
                            'specify the number of layers you would like to ' +
                            'remove from the top of the network for featurization.'+
                            ' Can be between 1 and 4.')

        for key in dict_of_booleans:
            if not isinstance(dict_of_booleans[key], bool):
                raise TypeError(key + ' is not a boolean! Please set to True '+
                                'or False, or leave blank for default configuration')

        if not isinstance(num_pooled_features, int):
                raise TypeError('num_pooled_features is not an integer!' +
                                ' Please set to an integer. Recommended value ' +
                                'is 1024, but can be set to any integer divisor' +
                                ' of the number of unpooled features.')

        #----------------------------------------------------------------------#


        ---###### BUILDING THE MODEL ######---
        model = build_featurizer(depth_of_featurizer, downsample_features,
                                 num_pooled_features)

        # Images
        self.image_files = image_files
        self.vectorized_images = vectorized_images
        self.processed_images=  image_files

        # ADD THIS IF WE MAKE CHOICE OF MODEL
        # Model type
        #self.model_string = model_string

        # Image scaling and cropping
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        self.number_crops = number_crops
        self.isotropic_scaling = isotropic_scaling


        # Save the model
        self.model = model
