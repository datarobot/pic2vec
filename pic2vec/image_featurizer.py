"""
This file contains the full ImageFeaturizer class, which allows users to upload
an image directory, a csv containing a list of image URLs, or a directory with a
csv containing names of images in the directory.

It featurizes the images using pretrained, decapitated InceptionV3 model, and
saves the featurized data to a csv, as well as within the ImageFeaturizer class
itself. This allows data scientists to easily analyze image data using simpler models.

Functionality:

    1. Build the featurizer model. The class initializer ImageFeaturizer() takes as input:
        depth : int
            1, 2, 3, or 4, depending on how far down you want to sample the featurizer layer

        auto_sample : bool
            a boolean flag signalling automatic downsampling

        downsample_size : int
            desired number of features to downsample the final layer to. Must be an
            integer divisor of the number of features in the layer.

    2. Load the data. The self.load_data() function takes as input:
            image_column_headers : str
                the name of the column holding the image data, if a csv exists,
                or what the name of the column will be, if generating the csv
                from a directory

            image_path : str
                the path to the folder containing the images. If using URLs, leave blank

            csv_path : str
                the path to the csv. If just using a directory, leave blank, and
                specify the path for the generated csv in new_csv_name.
                If csv exists, this is the path where the featurized csv will be
                generated.

            new_csv_name : str
                the path to the new csv, if one is being generated from a directory.
                If no csv exists, this is the path where the featurized csv will
                be generated

            scaled_size : tuple
                The size that the images get scaled to. Default is (299, 299)

            grayscale : bool
                Decides if image is grayscale or not. May get deprecated. Don't
                think it works on the InceptionV3 model due to input size.

    3. Featurize the data. The self.featurize_preloaded_data() function takes no input, and
       featurizes the loaded data, writing the new csvs to the same path as the loaded csv
       Also adds a binary "image_missing" column automatically, for any images that are missing
       from the image list.

    3a. Users can also load and featurize the data in one pass, with the
        self.featurize_data function, which takes the same input as the
        load_data function and performs the featurization automatically.

"""

import logging
import os
import math
import time
import numpy as np
import trafaret as t
import pandas as pd

from .build_featurizer import build_featurizer, supported_model_types
from .feature_preprocessing import preprocess_data, _image_paths_finder, _create_csv_path
from .data_featurizing import featurize_data, create_features, _named_path_finder


logger = logging.getLogger(__name__)

SIZE_DICT = {'squeezenet': (227, 227), 'vgg16': (224, 224), 'vgg19': (224, 224),
             'resnet50': (224, 224), 'inceptionv3': (299, 299), 'xception': (299, 299)}


class ImageFeaturizer:
    """
    This object can load images, rescale, crop, and vectorize them into a
    uniform batch, and then featurize the images for use with custom classifiers.

          Methods
    ------------------
        __init__(depth, auto_sample,
                 downsample_size):
            --------------------------------
            Initialize the ImageFeaturizer. Build the featurizer model with the
            depth and feature downsampling specified by the inputs.



        featurize_data(image_column_headers, image_path,
                                csv_path, new_csv_name, scaled_size, grayscale):
            --------------------------------
            Loads image directory and/or csv into the model, and
            featurizes the images



        load_data(image_column_headers, image_path, csv_path,
                  new_csv_name, scaled_size, grayscale):
            --------------------------------
            Loads image directory and/or csv into the model, and vectorize the
            images for input into the featurizer



        featurize_preloaded_data():
            --------------------------------
            Featurize the loaded data, append the features to the csv, and
            return the full dataframe


    """

    @t.guard(depth=t.Int(gte=1, lte=4),
             auto_sample=t.Bool,
             downsample_size=t.Int(gte=0),
             model=t.Enum(*supported_model_types.keys()))
    def __init__(self,
                 depth=1,
                 auto_sample=False,
                 downsample_size=0,
                 model='squeezenet'
                 ):
        """
        Initializer.

        Loads an initial InceptionV3 pretrained network, decapitates it and
        downsamples according to user specifications.

        Parameters:
        ----------
            depth : int
                How deep to decapitate the model. Deeper means less specific but
                also less complex

            auto_sample : bool
                If True, feature layer is automatically downsampled to the right size.

            downsample_size: int
                The number of features to downsample the featurizer to

        Returns:
        --------
        None. Initializes and saves the featurizer object attributes.

        """
        # BUILDING THE MODEL #
        logging.info("Building the featurizer.")

        featurizer = build_featurizer(depth, auto_sample,
                                      downsample_size, model_str=model.lower())

        # Saving initializations of model
        self.depth = depth
        self.auto_sample = auto_sample
        self.downsample_size = downsample_size
        self.num_features = featurizer.layers[-1].output_shape[-1]

        # Save the model
        self.model_name = model.lower()
        self.featurizer = featurizer
        self.visualize = featurizer.summary

        # Initializing preprocessing variables for after we load and featurize the images
        self.data = np.zeros((1))
        self.features = pd.DataFrame()
        self.df_original = pd.DataFrame()
        self.full_dataframe = pd.DataFrame()
        self.df_features = pd.DataFrame()
        self.csv_path = ''
        self.image_dict = {}
        self.image_column_headers = ''
        self.image_path = ''

        # Image scaling and cropping
        self.scaled_size = (0, 0)
        self.crop_size = (0, 0)
        self.number_crops = 0
        self.isotropic_scaling = False

    def load_data(self,
                  image_column_headers,
                  image_path='',
                  image_dict='',
                  csv_path='',
                  new_csv_name='~/Downloads/featurized_images.csv',
                  grayscale=False,
                  save_data=True
                  # crop_size = (299, 299),
                  # number_crops = 0,
                  # random_crop = False,
                  # isotropic_scaling = True
                  ):
        """
        Load image directory and/or csv, and vectorize the images for input into the featurizer.

        Parameters:
        ----------
            image_column_headers : str
                the name of the column holding the image data, if a csv exists,
                or what the name of the column will be, if generating the csv
                from a directory

            image_path : str
                the path to the folder containing the images. If using URLs, leave blank

            csv_path : str
                the path to the csv. If just using a directory, leave blank, and
                specify the path for the generated csv in new_csv_name.
                If csv exists, this is the path where the featurized csv will be
                generated.

            new_csv_name : str
                the path to the new csv, if one is being generated from a directory.
                If no csv exists, this is the path where the featurized csv will
                be generated

            grayscale : bool
                Decides if image is grayscale or not. May get deprecated. Don't
                think it works on the InceptionV3 model due to input size.

            # These features haven't been implemented yet.
            # isotropic_scaling : bool
            #     if True, images are scaled keeping proportions and then cropped
            #
            # crop_size: tuple
            #     if the image gets cropped, decides the size of the crop
            #
            # random_crop: bool
            #    If False, only take the center crop. If True, take random crop
            #

        """
        # Fix column headers and image path if they haven't been done, build path for new csv
        image_column_headers, image_path = self._input_fixer(image_column_headers, image_path)

        # If there's no dataframe, build it!
        if csv_path == '':
            if len(image_column_headers) > 1:
                raise ValueError('If building the dataframe from an image directory, the featurizer'
                                 'can only create a single image column. If two image columns are '
                                 'needed, please create a csv to pass in.')

        # If the image_dict hasn't been passed in (which only happens in batch processing),
        # build the full image dict and save the original dataframe
        if not image_dict:
            image_dict, df = self._full_image_dict_finder(image_path, csv_path,
                                                          image_column_headers,
                                                          new_csv_name)
            self.df_original = df
            self.full_dataframe = df
            self.image_column_headers = image_column_headers
            self.image_dict = image_dict

        scaled_size, full_image_data, csv_path = \
            self._load_data_helper(self.model_name, image_column_headers,
                                   image_path, image_dict, csv_path,
                                   new_csv_name, grayscale)

        # Save all of the necessary data to the featurizer
        if save_data:
            self.data = full_image_data
        self.csv_path = csv_path
        self.image_path = image_path
        self.scaled_size = scaled_size
        return full_image_data

    @t.guard(batch_data=t.Type(np.ndarray),
             image_column_headers=t.List(t.String(allow_blank=True)) | t.String(allow_blank=True),
             batch_processing=t.Bool,
             save_features=t.Bool,
             save_csv=t.Bool,
             omit_model=t.Bool,
             omit_depth=t.Bool,
             omit_output=t.Bool,
             omit_time=t.Bool,
             )
    def featurize_preloaded_data(self, batch_data=np.zeros((1)), image_column_headers='',
                                 batch_processing=False, save_features=False, save_csv=False,
                                 omit_model=False, omit_depth=False, omit_output=False,
                                 omit_time=False):
        """
        Featurize the loaded data, returning the dataframe and writing the features
        and the full combined data to csv

        Parameters
        ----------
        None, just operates on the loaded data

        Returns
        -------
            full_dataframe : pandas.DataFrame
                Dataframe containing the features appended to the original csv.
                Also writes csvs containing the features only and the full dataframe
                to the same path as the csv containing the list of names

        """

        # If the batch data isn't passed in, then load the full data from the attributes
        if np.array_equal(batch_data, np.zeros((1))):
            batch_data = self.data
        if image_column_headers == '':
            image_column_headers = self.image_column_headers
        if isinstance(image_column_headers, str):
            image_column_headers = [image_column_headers]

        # Check data has been loaded, and that the data was vectorized correctly
        if np.array_equal(batch_data, np.zeros((1))):
            raise IOError('Must load data into the model first. Call load_data.')

        # If batch processing, make sure we're only doing a single column at a time.
        # Otherwise, make sure the number of columns matches the first dimension of the data
        if batch_processing:
            assert len(image_column_headers) == 1 or isinstance(image_column_headers, str)
        else:
            assert len(image_column_headers) == batch_data.shape[0]
        logging.info("Trying to featurize data.")

        # Initialize featurized data vector with appropriate size
        features = np.zeros((batch_data.shape[1],
                             self.num_features * len(image_column_headers)))
        # Save csv
        full_dataframe, df_features = self._featurize_helper(
            features, image_column_headers, save_features, batch_data)

        if save_csv:
            self.save_csv(omit_model, omit_depth, omit_output, omit_time)

        self.full_dataframe = full_dataframe

        if save_features:
            self.features = df_features
            return full_dataframe, df_features

        return full_dataframe

    def featurize(self,
                  image_column_headers,
                  image_path='',
                  csv_path='',
                  new_csv_name='~/Downloads/featurized_images.csv',
                  batch_processing=True,
                  batch_size=1000,
                  grayscale=False,
                  save_data=False,
                  save_features=False,
                  save_csv=False,
                  omit_time=False,
                  omit_model=False,
                  omit_depth=False,
                  omit_output=False,
                  verbose=True
                  # crop_size = (299, 299),
                  # number_crops = 0,
                  # random_crop = False,
                  # isotropic_scaling = True
                  ):
        """
        Load image directory and/or csv, and vectorize the images for input into the featurizer.
        Then, featurize the data.

        Parameters:
        ----------
            image_column_headers : list of str
                list of the names of the column holding the image data, if a csv exists,
                or what the name of the column will be, if generating the csv
                from a directory

            image_path : str
                the path to the folder containing the images. If using URLs, leave blank

            csv_path : str
                the path to the csv. If just using a directory, leave blank, and
                specify the path for the generated csv in new_csv_name.
                If csv exists, this is the path where the featurized csv will be
                generated.

            new_csv_name : str
                the path to the new csv, if one is being generated from a directory.
                If no csv exists, this is the path where the featurized csv will
                be generated

            grayscale : bool
                Decides if image is grayscale or not. May get deprecated. Don't
                think it works on the InceptionV3 model due to input size.

            # These features haven't been implemented yet.
            # isotropic_scaling : bool
            #     if True, images are scaled keeping proportions and then cropped
            #
            # crop_size: tuple
            #     if the image gets cropped, decides the size of the crop
            #
            # random_crop: bool
            #    If False, only take the center crop. If True, take random crop
            #

        Returns:
        --------
            full_dataframe :
                Dataframe containing the features appended to the original csv.
                Also writes csvs containing the features only and the full dataframe
                to the same path as the csv containing the list of names

        """
        # Fix column headers and image path if necessary
        image_column_headers, image_path = self._input_fixer(image_column_headers, image_path)

        # Find the full image dict and save the original dataframe. This is required early to know
        # how many images exist in total, to control batch processing.
        full_image_dict, df_original = self._full_image_dict_finder(image_path, csv_path,
                                                                    image_column_headers,
                                                                    new_csv_name)
        # Save the fixed inputs and full image dict
        self.df_original = df_original
        self.image_column_headers = image_column_headers
        self.image_dict = full_image_dict

        # Users can turn off batch processing by either setting batch_processing to false, or
        # setting batch_size to 0
        if batch_processing and batch_size:
            # Perform batch processing, and save the full dataframe and the full features dataframe
            full_df, features_df = self._batch_processing(full_image_dict, image_column_headers,
                                                          df_original, image_path, csv_path,
                                                          new_csv_name, batch_size, grayscale,
                                                          save_features, verbose)

            # Save the full dataframe with the features
            self.full_dataframe = full_df

        # If batch processing is turned off, load the images in one big batch and features them all
        else:
            if verbose:
                logging.warning("Loading full data tensor without batch processing. If you "
                                "experience a memory error, make sure batch processing is enabled.")
            full_data = self.load_data(image_column_headers, image_path, full_image_dict, csv_path,
                                       new_csv_name, grayscale, save_data)

            full_df, features_df = \
                self.featurize_preloaded_data(full_data, image_column_headers=image_column_headers,
                                              save_features=save_features, save_csv=save_csv,
                                              omit_time=omit_time, omit_model=omit_model,
                                              omit_depth=omit_depth, omit_output=omit_output)

            # Save the full dataframe with the features
            self.full_dataframe = full_df

        # Save features and csv if flags are enabled
        if save_features:
            self.features = features_df
        if save_csv:
            self.save_csv(csv_path=csv_path, omit_model=omit_model, omit_depth=omit_depth,
                          omit_output=omit_output, omit_time=omit_time, save_features=save_features)

        # Return the results
        if save_features:
            return full_df, features_df
        return full_df

    def save_csv(self, csv_path="", omit_model=False, omit_depth=False,
                 omit_output=False, omit_time=False, save_features=False):
        # Save the name and extension separately, for robust naming
        if not csv_path:
            csv_path = self.csv_path
        csv_name, ext = os.path.splitext(csv_path)

        name_path = _named_path_finder(csv_name, self.model_name, self.depth, self.num_features,
                                       omit_model, omit_depth, omit_output, omit_time)

        _create_csv_path(csv_path)
        logger.warning("Saving full dataframe to csv as {}_full{}".format(name_path, ext))
        self.full_dataframe.to_csv("{}_full{}".format(name_path, ext), index=False)
        if save_features:
            logger.warning("Saving features to csv as {}_features_only{}".format(name_path, ext))
            self.df_features.to_csv("{}_features_only{}".format(name_path, ext), index=False)

    @t.guard(confirm=t.Bool)
    def clear_input(self, confirm=False):
        """
        Clear all input for the model. Requires the user to confirm with an additional "confirm"
        argument in order to run.
        """
        if not confirm:
            raise ValueError('If you\'re sure you would like to clear the inputs of this model, '
                             'rerun the function with the following argument: '
                             'clear_input(confirm=True). This operation cannot be reversed.')

        self.data = np.zeros((1))
        self.features = pd.DataFrame()
        self.full_dataframe = pd.DataFrame()
        self.csv_path = ''
        self.image_list = ''
        self.image_column_headers = ''
        self.image_path = ''

    # ###################
    # Helper Functions! #
    # ###################

    def _load_data_helper(self,
                          model_name,
                          image_column_headers,
                          image_path,
                          image_dict,
                          csv_path,
                          new_csv_name,
                          grayscale):
        # Save size that model scales to
        scaled_size = SIZE_DICT[model_name]

        # Save the full image tensor, the path to the csv, and the list of image paths
        (image_data, csv_path, list_of_image_paths) = \
            preprocess_data(image_column_headers[0], model_name,
                            image_dict[image_column_headers[0]],
                            image_path, csv_path, new_csv_name, scaled_size, grayscale)

        full_image_data = np.expand_dims(image_data, axis=0)

        if len(image_column_headers) > 1:
            for column in image_column_headers[1:]:
                (image_data, csv_path, list_of_image_paths) = \
                    preprocess_data(column, model_name, image_dict[column], image_path,
                                    csv_path, new_csv_name, scaled_size, grayscale)
                full_image_data = np.concatenate((full_image_data,
                                                  np.expand_dims(image_data, axis=0)))
        return scaled_size, full_image_data, csv_path

    def _featurize_helper(self, features, image_column_headers,
                          save_features, batch_data):

        # Save the initial features list
        features_list = []

        # For each image column, perform the full featurization and add the features to the df
        for column in range(batch_data.shape[0]):
            # Featurize the data, and save it to the appropriate columns
            partial_features = featurize_data(self.featurizer, batch_data[column])

            features[:, self.num_features * column:self.num_features * column + self.num_features]\
                = partial_features

            # Save the full dataframe
            df_features = \
                create_features(batch_data[column],
                                partial_features,
                                image_column_headers[column],
                                save_features=save_features)

            features_list.append(df_features)

        df_features = pd.concat(features_list, axis=1)
        full_dataframe = pd.concat([self.df_original, df_features], axis=1)

        return full_dataframe, df_features

    def _batch_processing(self,
                          full_image_dict,
                          image_column_headers,
                          df_original,
                          image_path='',
                          csv_path='',
                          new_csv_name='~/Downloads/featurized_images.csv',
                          batch_size=1000,
                          grayscale=False,
                          save_features=False,
                          verbose=True):

        full_features_df = pd.DataFrame()
        full_df = df_original
        full_df_columns_list = []
        # Iterate through each image column
        for column_index in range(len(image_column_headers)):
            # Initialize the batch index and save the column name
            index = 0
            batch_number = 0
            column = image_column_headers[column_index]
            batch_features_df = pd.DataFrame()

            # Get the list of image paths and the number of images in this column
            list_of_image_paths = full_image_dict[column]
            num_images = len(list_of_image_paths)

            batch_features_list = []
            # Loop through the images, featurizing each batch
            if verbose and len(image_column_headers) > 1:
                print("Featurizing column #{}".format(column_index + 1))
            while index < num_images:
                if verbose:
                    tic = time.clock()
                # Cap the batch size against the total number of images left to prevent overflow
                if index + batch_size > num_images:
                    batch_size = num_images - index

                # Create a dictionary for just the batch of images
                batch_image_dict = {column: full_image_dict[column][index:index + batch_size]}

                # Load the images
                if verbose:
                    print("Loading image batch.")
                batch_data = self.load_data(column, image_path,
                                            batch_image_dict, csv_path, new_csv_name,
                                            grayscale, save_data=False)
                if verbose:
                    print("Featurizing image batch.")
                # If this is the first batch, the batch features will be saved alone.
                # Otherwise, they are concatenated to the last batch
                batch_features_list.append(self.featurize_preloaded_data(batch_data, column,
                                                                         save_features=True,
                                                                         batch_processing=True)[1])

                # Increment index by batch size
                index += batch_size
                batch_number += 1

                # Give update on time and number of images left in column
                if verbose:
                    remaining_batches = int(math.ceil(num_images - index) / batch_size)
                    print("Featurized batch #{}. Number of images left: {}\nEstimated total time "
                          "left: {} seconds".format(batch_number, num_images - index,
                                                    int((time.clock() - tic) * remaining_batches)))

            # After the full column's features are calculated, concatenate them all and append them
            # to the full DataFrame list
            batch_features_df = pd.concat(batch_features_list, ignore_index=True)
            full_df_columns_list.append(batch_features_df)

        # Once all the features are created for each column, concatenate them together for both
        # the features dataframe and the full dataframe
        full_features_df = pd.concat(full_df_columns_list, axis=1)
        if save_features:
            self.features = full_features_df
        else:
            self.features = pd.DataFrame()
        full_df = pd.concat([full_df, full_features_df], axis=1)

        # Return the full dataframe and features dataframe
        return full_df, full_features_df

    def _full_image_dict_finder(self, image_path, csv_path, image_column_headers, new_csv_name):
        full_image_dict = {}
        for column in image_column_headers:
            list_of_image_paths, df = _image_paths_finder(image_path, csv_path,
                                                          column, new_csv_name)

            full_image_dict[column] = list_of_image_paths
        return full_image_dict, df

    def _input_fixer(self, image_column_headers, image_path):
            # Convert column header to list if it's passed a single string
        if isinstance(image_column_headers, str):
            image_column_headers = [image_column_headers]

        # Add backslash to end of image path if it is not there
        if image_path != '' and image_path[-1] != "/":
            image_path = '{}/'.format(image_path)

        return image_column_headers, image_path
