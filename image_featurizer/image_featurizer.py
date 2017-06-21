# -*- coding: utf-8 -*-

"""Main module."""
from ./build_featurizer import build_featurizer
from keras.layers.merge import average
import os

class ImageFeaturizer:
    '''
    This object can load images, rescale, crop, and vectorize them into a
    uniform batch, and then featurize the images for use with custom classifiers.
    '''

    def load_data(image_column_header,
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

        ### Parameters: ###
            image_column_header: the name of the column holding the image data

            image_directory_path: The path to the folder containing the images

            scaled_size: The size that the images get scaled to

            # crop_size: If the image gets cropped, decides the size of the crop
            #
            # number_crops: If 0, no cropping. Otherwise, it represents the number
            #               of crops taken of the image
            #
            # random_crop: If False, only take the center crop. If True, take random crops
            #
            # isotropic_scaling: If False, the image is scaled non-proportionally.
            #                 If true, image is scaled keeping proportions, and
            #                 then cropped to correct size
        '''

        #------------------------------------------------#
                    ### ERROR CHECKING ###

        # # Raise an error if image_column_header is not a string
        # if not isinstance(image_column_header, str):
        #     raise TypeError('image_column_header must be passed a string! This '+
        #                     'determines where to look for (or create) the column'+
        #                     ' of image paths in the csv.')
        #
        # # Raise an error if image_directory_path is not a string
        # if not isinstance(image_directory_path, str) or isinstance(image_directory_path, type(None)):
        #     raise TypeError('image_directory_path must be passed a string, or left'+
        #                     ' as None! This determines where to look for the folder of images,'+
        #                     ' or says if it doesn\'t exist.')
        #
        # # Raise an error if the image_directory_path doesn't point to a directory
        # if image_directory_path != None:
        #     if not os.path.isdir(image_directory_path):
        #         raise TypeError('image_directory_path must lead to a directory if '+
        #                         'it is initialized! It is where the images are stored.')
        #
        # # Raise an error if csv_path is not a string
        # if not isinstance(csv_path, str) or isinstance(csv_path, type(None)):
        #     raise TypeError('csv_path must be passed a string, or left as None!'+
        #                     ' This determines where to look for the csv,'+
        #                     ' or says if it doesn\'t exist.')
        #
        # # Raise an error if the csv_path doesn't point to a file
        # if csv_path != None:
        #     if not os.path.isfile(csv_path):
        #         raise TypeError('csv_path must lead to a file if it is initialized!'+
        #                         ' This is the csv containing pointers to the images.')
        #
        # # Raise an error if image_column_header is not a string
        # if not isinstance(new_csv_name, str):
        #     raise TypeError('new_csv_name must be passed a string! This '+
        #                     'determines where to create the new csv from images'+
        #                     'if it doesn\'t already exist!.')
        #
        # # Raise an error if scaled_size is not a tuple of integers
        # if not isinstance(scaled_size, tuple):
        #     raise TypeError('scaled_size is not a tuple! Please list dimensions as a tuple')
        #
        # for element in scaled_size:
        #     if not isinstance(element, int):
        #         raise TypeError('scaled_size must be a tuple of integers!')
        #
        # if not isinstance(grayscale, bool):
        #     raise TypeError('grayscale must be a boolean! This determines if the'+
        #                     'images are grayscale or in color. Default is False')

            # Error checking for options I have not yet implemented!
        # # Just a dictionary of the two booleans for easy error checking
        # dict_of_booleans = {'random_crop': random_crop, 'isotropic_scaling': isotropic_scaling,
        #
        # if not isinstance(crop_size, tuple):
        #     raise TypeError('crop_size is not a tuple! Please list dimensions as a tuple')
        #
        # for element in crop_size:
        #     if not isinstance(element, int):
        #         raise ValueError('crop_size must be a tuple of integers!')
        #
        #
        # if not isinstance(number_crops, int):
        #     raise TypeError('number_crops is not an integer! Please specify the' +
        #                     ' number of random crops you would like to average')
        #
        # for key in dict_of_booleans:
        #     if not isinstance(dict_of_booleans[key], bool):
        #         raise TypeError(key + ' is not a boolean! Please set to True '+
        #                         'or False, or leave blank for default configuration')
        #------------------------------------------------#


        # Save the full image tensor, the path to the csv, and the list of image paths
        (full_image_data, csv_path, list_of_image_paths) = \
            preprocess_data(image_column_header, image_directory_path, csv_path,
                            new_csv_name, scaled_size, grayscale)

        # Save all of the necessary data to the featurizer!
        self.data = full_image_data
        self.csv_path = csv_path
        self.image_paths = list_of_image_paths
        self.image_column_header = image_column_header

    def featurize():
        num_images = self.data.shape[0]
        df = pd.read_csv(self.csv_path)

        for i in range(num_images):
            image = self.data[i,:,:,:]
            image_path = self.image_paths[i]

            featurized_tensor = self.model.predict(self.data)


            csv_row = df.loc[df[image_column_header] == image_path]).index

            df_row =



    def __init__(self,
                depth = 1,
                downsample = False,
                downsample_size = 0
                ):

        '''
        Initializer:

        Loads an initial InceptionV3 pretrained network, decapitates it and
        downsamples according to user specifications.

        ### Parameters: ###
            depth: How many layers deep we're taking the model

            downsample: If True, feature layer is downsampled

            downsample_size: If feature layer is downsampled, chooses number of
                            features to downsample it to
        '''

        #------------------------------------------------#
                    ### ERROR CHECKING ###

        # Acceptable depths for decapitation
        acceptable_depths = [1,2,3,4]

        if not isinstance(depth, int):
            raise TypeError('depth is not set to an integer! Please ' +
                            'specify the number of layers you would like to ' +
                            'remove from the top of the network for featurization.'+
                            ' Can be between 1 and 4.')

        if not depth in acceptable_depths:
            raise ValueError('Depth can be set to 1, 2, 3, or 4. Otherwise, ' +
                             'leave it blank for default configuration of 1.')

        if not isinstance(downsample, bool):
            raise TypeError('downsample is not set to a boolean! If you would like to' +
                            ' downsample the featurizer, please set to True. '+
                            'Otherwise, leave blank for default configuration.')

        if not isinstance(downsample_size, int):
                raise TypeError('Tried to set downsample_size to a non-integer value!' +
                                ' Please set to an integer or leave uninitialized.'+
                                ' Recommended value is 1024 for depths 1,2, or 3,' +
                                'but can be set to any integer divisor of the'+
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
        self.data = np.empty((1))
        self.csv_path = ''
        self.image_paths = ''


        # Image scaling and cropping
        self.scaled_size = (0,0)
        self.crop_size = (0,0)
        self.number_crops = 0
        self.isotropic_scaling = False
