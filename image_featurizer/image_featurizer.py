# -*- coding: utf-8 -*-

"""Main module."""
from build_featurizer import build_featurizer
from keras.layers.merge import average

class ImageFeaturizer:
    '''
    This object can load images, rescale, crop, and vectorize them into a
    uniform batch, and then featurize the images for use with custom classifiers.
    '''


    def __init__(self,
                image_filepath = None,
                scaled_size = (299, 299),
                crop_size = (299, 299),
                number_crops = 0,
                random_crop = False,
                isotropic_scaling = True,
                depth = 1,
                downsample = False,
                downsample_size = None
                ):

        '''
        Initializer:

        Loads an initial InceptionV3 pretrained network, decapitates it and
        downsamples according to user specifications. Loads image file and csv,
        and stores model to perform image featurization.

        ### Parameters: ###
            images_files: The file containing the images

            scaled_size: The size that the images get scaled to

            crop_size: If the image gets cropped, decides the size of the crop

            number_crops: If 0, no cropping. Otherwise, it represents the number
                          of crops taken of the image

            random_crop: If False, only take the center crop. If True, take random crops

            isotropic_scaling: If False, the image is scaled non-proportionally.
                            If true, image is scaled keeping proportions, and
                            then cropped to correct size

            depth: How many layers deep we're taking the model

            downsample: If True, feature layer is downsampled

            downsample_size: If feature layer is downsampled, chooses number of
                            features to downsample it to
        '''

        #------------------------------------------------#
                    ### ERROR CHECKING ###
        # A dictionary of the boolean set for error-checking
        dict_of_booleans = {'random_crop': random_crop, 'isotropic_scaling': isotropic_scaling,
                            'downsample': downsample}

        if image_files == None:
            raise ValueError('Image files required.')

        if not isinstance(scaled_size, tuple):
            raise TypeError('scaled_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(crop_size, tuple):
            raise TypeError('crop_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(number_crops, int):
            raise TypeError('number_crops is not an integer! Please specify the' +
                            ' number of random crops you would like to average')

        if not isinstance(depth, int):
            raise TypeError('depth is not an integer! Please' +
                            'specify the number of layers you would like to ' +
                            'remove from the top of the network for featurization.'+
                            ' Can be between 1 and 4.')

        for key in dict_of_booleans:
            if not isinstance(dict_of_booleans[key], bool):
                raise TypeError(key + ' is not a boolean! Please set to True '+
                                'or False, or leave blank for default configuration')

        if not (isinstance(downsample_size, int) or isinstance(downsample_size, type(None))):
                raise TypeError('Tried to set downsample_size to a non-integer value!' +
                                ' Please set to an integer or leave uninitialized.'+
                                ' Recommended value is 1024, but can be set to' +
                                'any integer divisor of the number of unpooled features.')

        #----------------------------------------------------------------------#


        ###### BUILDING THE MODEL ######
        print "\nBuilding the featurizer!"
        model = build_featurizer(depth, downsample,
                                 downsample_size)

        # Images
        self.image_files = image_files
        self.vectorized_images = None
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
        self.visualize=model.summary
