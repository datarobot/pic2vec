# -*- coding: utf-8 -*-

"""Main module."""
import os
import numpy as np

from .build_featurizer import build_featurizer
from .feature_preprocessing import preprocess_data
from .data_featurizing import featurize_data, features_to_csv



class ImageFeaturizer:
    '''
    This object can load images, rescale, crop, and vectorize them into a
    uniform batch, and then featurize the images for use with custom classifiers.

    Methods
    -------
    load_data : No return
        Load the data into the model

    featurize : pandas.DataFrame
        Featurize the loaded data, append the features to the csv, and return the full dataframe

    load_and_featurize_data : pandas.DataFrame
        Load and featurize the data in one function
    '''

    def __init__(self,
                depth = 1,
                downsample = False,
                downsample_size = 0
                ):

        '''
        Initializer:

        Loads an initial InceptionV3 pretrained network, decapitates it and
        downsamples according to user specifications.

        Parameters:
        ----------
            depth : int
                How deep to decapitate the model. Deeper means less specific but
                also less complex

            downsample : bool
                If True, feature layer is downsampled.

            downsample_size: int
                The number of features to downsample the featurizer to

        Returns:
        --------
        None. Initializes and saves the featurizer object attributes.
        '''

        #------------------------------------------------#
                    ### ERROR CHECKING ###

        # Acceptable depths for decapitation
        acceptable_depths = [1,2,3,4]

        if not isinstance(depth, int):
            raise TypeError('depth is not set to an integer! Please ' \
                            'specify the number of layers you would like to ' \
                            'remove from the top of the network for featurization.' \
                            ' Can be between 1 and 4.')

        if not depth in acceptable_depths:
            raise ValueError('Depth can be set to 1, 2, 3, or 4. Otherwise, ' \
                             'leave it blank for default configuration of 1.')

        if not isinstance(downsample, bool):
            raise TypeError('downsample is not set to a boolean! If you would like to' \
                            ' downsample the featurizer, please set to True. ' \
                            'Otherwise, leave blank for default configuration.')

        if not isinstance(downsample_size, int):
                raise TypeError('Tried to set downsample_size to a non-integer value!' \
                                ' Please set to an integer or leave uninitialized.' \
                                ' Recommended value is 1024 for depths 1,2, or 3,' \
                                'but can be set to any integer divisor of the' \
                                ' number of unpooled features.')
        #------------------------------------------------#


        ###### BUILDING THE MODEL ######
        print("\nBuilding the featurizer!")

        model = build_featurizer(depth, downsample,
                                 downsample_size)


        # Saving initializations of model
        self.depth = depth
        self.downsample = downsample
        self.downsample_size = downsample_size

        # Save the model
        self.model = model
        self.visualize = model.summary

        # Initializing preprocessing variables for after we load the images
        self.data = np.zeros((1))
        self.featurized_data = np.zeros((1))
        self.csv_path = ''
        self.image_list = ''
        self.image_column_header = ''


        # Image scaling and cropping
        self.scaled_size = (0,0)
        self.crop_size = (0,0)
        self.number_crops = 0
        self.isotropic_scaling = False

    def load_and_featurize_data(self,
                  image_column_header,
                  image_directory_path='',
                  csv_path='',
                  new_csv_name='featurizer_csv/generated_images_csv',
                  scaled_size = (299, 299),
                  grayscale=False

                   #crop_size = (299, 299),
                   #number_crops = 0,
                   #random_crop = False,
                   #isotropic_scaling = True
                  ):
        '''
        Load and featurize the data in one pass.


        '''
        self.load_data(image_column_header,image_directory_path,csv_path,new_csv_name, \
                       scaled_size,grayscale)
        return self.featurize()


    def load_data(self,
                  image_column_header,
                  image_directory_path='',
                  csv_path='',
                  new_csv_name='featurizer_csv/generated_images_csv',
                  scaled_size = (299, 299),
                  grayscale=False

                  #crop_size = (299, 299),
                  #number_crops = 0,
                  #random_crop = False,
                  #isotropic_scaling = True
                  ):
        '''
        Loads image directory and/or csv, and vectorizes the images for input
        into the featurizer.

        Parameters:
        ----------
            image_column_header : str
                the name of the column holding the image data

            image_directory_path : str
                the path to the folder containing the images

            scaled_size : tuple
                The size that the images get scaled to.


            ### These features haven't been implemented yet!
            # isotropic_scaling: If False, the image is scaled non-proportionally.
            #                 If true, image is scaled keeping proportions, and
            #                 then cropped to correct size
            # crop_size: If the image gets cropped, decides the size of the crop
            #
            # random_crop: If False, only take the center crop. If True, take random crop
            #
        '''

        # If new csv_path is being generated, make sure
        # the folder exists!
        if (csv_path==''):
            if not os.path.isdir(os.path.dirname(new_csv_name)):
                os.makedirs(os.path.dirname(new_csv_name))

        # Save the full image tensor, the path to the csv, and the list of image paths
        (full_image_data, csv_path, list_of_image_paths) = \
            preprocess_data(image_column_header, image_directory_path, csv_path,
                            new_csv_name, scaled_size, grayscale)

        # Save all of the necessary data to the featurizer!
        self.data = full_image_data
        self.csv_path = csv_path
        self.image_list = list_of_image_paths
        self.image_column_header = image_column_header
        self.scaled_size = scaled_size



    def featurize(self):
        print("Checking array initialized.")
        if np.array_equal(self.data, np.zeros((1))):
            raise IOError('Must load data into the model first! Call load_data.')

        print("Trying to featurize data!")
        self.featurized_data = featurize_data(self.model, self.data)

        return features_to_csv(self.featurized_data, self.csv_path, self.image_column_header, self.image_list)
