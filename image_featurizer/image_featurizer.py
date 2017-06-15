# -*- coding: utf-8 -*-

"""Main module."""
from keras.applications.xception import Xception


class ImageFeaturizer:
    '''
    This object can load images, rescale, crop, and vectorize them into a
    uniform batch, and then featurize the images for use with custom classifiers.
    '''

    def __init__(self,
                images_files = None,
                scaled_size = (256, 256),
                crop_size = (224, 224),
                number_crops = 1,
                isotropic_scaling = True,
                whitening = True,
                std_dev = True,
                contrast_norm = True,
                vertical_flip = False,
                horizontal_flip = True,
                rotation = False):


        # Type checking
        booleans = {'isotropic scaling': isotropic_scaling, 'whitening': whitening,
                    'std_dev': std_dev, 'contrast_norm': contrast_norm, 'vertical_flip': vertical_flip,
                    'horizontal_flip': horizontal_flip, 'rotation': rotation]

        if image_files == None:
            raise ValueError('Image files required.')

        for key in booleans:
            if not isinstance(booleans[key], bool):
                raise ValueError(key + ' is not boolean! Must be set to True or False.')

        if not isinstance(scaled_size, tuple):
            raise ValueError('scaled_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(crop_size, tuple):
            raise ValueError('crop_size is not a tuple! Please list dimensions as a tuple')

        if not isinstance(number_crops, int):
            raise ValueError('number_crops is not an integer! Please specify the \
                number of random crops you would like to average')

        # Create the model!
        model = Xception(weights=None)
        model.load_weights('..')

        # Images
        self.image_files = image_files
        self.vectorized_images = vectorized_images

        # Model type
        self.model_string = model_string

        # Image scaling and cropping
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        self.number_crops = number_crops
        self.isotropic_scaling = isotropic_scaling

        ##### Types of data normalization #####

        # Statistical norms
        self.whitening = whitening
        self.std_dev = std_dev
        self.contrast_norm = contrast_norm

        # Flipping the images
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self.rotation = rotation
