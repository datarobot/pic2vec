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

        autosample : bool
            a boolean flag signalling automatic downsampling

        downsample_size : int
            desired number of features to downsample the final layer to. Must be an
            integer divisor of the number of features in the layer.

    2. Load the data. The self.load_data() function takes as input:
            image_columns : str
                the name of the column holding the image data, if a csv exists,
                or what the name of the column will be, if generating the csv
                from a directory

            image_path : str
                the path to the folder containing the images. If using URLs, leave blank

            csv_path : str
                the path to the csv. If just using a directory, leave blank.
                If csv exists, this is the path where the featurized csv will be
                generated.

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
from .feature_preprocessing import preprocess_data, _image_paths_finder
from .data_featurizing import featurize_data, create_features


logger = logging.getLogger(__name__)

SIZE_DICT = {'squeezenet': (227, 227), 'vgg16': (224, 224), 'vgg19': (224, 224),
             'resnet50': (224, 224), 'inceptionv3': (299, 299), 'xception': (299, 299)}

DEFAULT_NEW_CSV_PATH = '{}{}'.format(os.path.expanduser('~'), '/Downloads/images.csv')


class ImageFeaturizer:
    """
    This object can load images, rescale, crop, and vectorize them into a
    uniform batch, and then featurize the images for use with custom classifiers.

          Methods
    ------------------
        __init__(depth, autosample,
                 downsample_size):
            --------------------------------
            Initialize the ImageFeaturizer. Build the featurizer model with the
            depth and feature downsampling specified by the inputs.



        featurize_data(image_columns, image_path,
                       csv_path, new_csv_path, scaled_size, grayscale):
            --------------------------------
            Loads image directory and/or csv into the model, and
            featurizes the images



        load_data(image_columns, image_path, csv_path,
                  scaled_size, grayscale):
            --------------------------------
            Loads image directory and/or csv into the model, and vectorize the
            images for input into the featurizer



        featurize_preloaded_data():
            --------------------------------
            Featurize the loaded data, append the features to the csv, and
            return the full dataframe


    """

    @t.guard(depth=t.Int(gte=1, lte=4),
             autosample=t.Bool,
             downsample_size=t.Int(gte=0),
             model=t.Enum(*supported_model_types.keys()))
    def __init__(self,
                 depth=1,
                 autosample=False,
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

            autosample : bool
                If True, feature layer is automatically downsampled to the right size.

            downsample_size: int
                The number of features to downsample the featurizer to

        Returns:
        --------
        None. Initializes and saves the featurizer object attributes.

        """
        # BUILDING THE MODEL #
        logging.info("Building the featurizer.")

        featurizer = build_featurizer(depth, autosample,
                                      downsample_size, model_str=model.lower())

        # Saving initializations of model
        self.depth = depth
        self.autosample = autosample
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
        self.image_columns = ''
        self.image_path = ''

        # Image scaling and cropping
        self.scaled_size = (0, 0)
        self.crop_size = (0, 0)
        self.number_crops = 0
        self.isotropic_scaling = False

    def load_data(self,
                  image_columns,
                  image_path='',
                  image_dict='',
                  csv_path='',
                  grayscale=False,
                  save_data=True,
                  # crop_size = (299, 299),
                  # number_crops = 0,
                  # random_crop = False,
                  # isotropic_scaling = True
                  ):
        """
        Load image directory and/or csv, and vectorize the images for input into the featurizer.

        Parameters:
        ----------
            image_columns : str
                the name of the column holding the image data, if a csv exists,
                or what the name of the column will be, if generating the csv
                from a directory

            image_path : str
                the path to the folder containing the images. If using URLs, leave blank

            csv_path : str
                the path to the csv. If just using a directory, leave blank.
                If csv exists, this is the path where the featurized csv will be
                generated.

            # These features haven't been implemented yet.
            # grayscale : bool
            #     Flags the image as grayscale
            #
            # isotropic_scaling : bool
            #     If True, images are scaled keeping proportions and then cropped
            #
            # crop_size: tuple
            #     If the image gets cropped, decides the size of the crop
            #
            # random_crop: bool
            #     If False, only take the center crop. If True, take random crop
            #

        """
        # Fix column headers and image path if they haven't been done, build path for new csv
        image_columns, image_path = _input_fixer(image_columns, image_path)

        # If there's no dataframe, build it!
        if csv_path == '':
            if len(image_columns) > 1:
                raise ValueError('If building the dataframe from an image directory, the featurizer'
                                 'can only create a single image column. If two image columns are '
                                 'needed, please create a csv to pass in.')

        # If the image_dict hasn't been passed in (which only happens in batch processing),
        # build the full image dict and save the original dataframe
        if not image_dict:
            image_dict, df = _build_image_dict(image_path, csv_path,
                                               image_columns)
            self.df_original = df
            self.full_dataframe = df
            self.image_columns = image_columns
            self.image_dict = image_dict

        scaled_size, full_image_data = \
            self._load_data_helper(self.model_name, image_columns,
                                   image_path, image_dict, csv_path, grayscale)

        # Save all of the necessary data to the featurizer
        if save_data:
            self.data = full_image_data

        self.csv_path = csv_path
        self.image_path = image_path
        self.scaled_size = scaled_size
        return full_image_data

    @t.guard(batch_data=t.Type(np.ndarray),
             image_columns=t.List(t.String(allow_blank=True)) | t.String(allow_blank=True),
             batch_processing=t.Bool,
             features_only=t.Bool,
             save_features=t.Bool,
             save_csv=t.Bool,
             new_csv_path=t.String(allow_blank=True),
             omit_model=t.Bool,
             omit_depth=t.Bool,
             omit_output=t.Bool,
             omit_time=t.Bool,
             )
    def featurize_preloaded_data(self, batch_data=np.zeros((1)), image_columns='',
                                 batch_processing=False, features_only=False,
                                 save_features=False, save_csv=False, new_csv_path='',
                                 omit_model=False, omit_depth=False, omit_output=False,
                                 omit_time=False):
        """
        Featurize the loaded data, returning the dataframe and writing the features
        and the full combined data to csv

        Parameters
        ----------


        Returns
        -------
            full_dataframe or df_features: pandas.DataFrame
                If features_only, this returns a Dataframe containing the features.
                Otherwise, it returns a DataFrame containing the features appended to the
                original csv. If save_csv is set to True, it also writes csv's
                to the same path as the csv containing the list of names.

        """

        # If the batch data isn't passed in, then load the full data from the attributes
        if np.array_equal(batch_data, np.zeros((1))):
            batch_data = self.data
        if image_columns == '':
            image_columns = self.image_columns
        if isinstance(image_columns, str):
            image_columns = [image_columns]

        # Check data has been loaded, and that the data was vectorized correctly
        if np.array_equal(batch_data, np.zeros((1))):
            raise IOError('Must load data into the model first. Call load_data.')

        # If batch processing, make sure we're only doing a single column at a time.
        # Otherwise, make sure the number of columns matches the first dimension of the data
        if batch_processing:
            assert len(image_columns) == 1 or isinstance(image_columns, str)
        else:
            assert len(image_columns) == batch_data.shape[0]
        logging.info("Trying to featurize data.")

        # Initialize featurized data vector with appropriate size
        features = np.zeros((batch_data.shape[1],
                             self.num_features * len(image_columns)))

        # Get the image features
        df_features = self._featurize_helper(
            features, image_columns, batch_data)

        # Save features if boolean set to True
        if save_features:
            self.features = df_features

        # If called with features_only, returns only the features
        if features_only:
            return df_features

        # Save the image features with the original dataframe
        full_dataframe = pd.concat([self.df_original, df_features], axis=1)

        # If batch processing, this is only the batch dataframe. Otherwise, this is the actual
        # full dataframe.
        if not batch_processing:
            self.full_dataframe = full_dataframe

        # Save csv if called
        if save_csv:
            self.save_csv(new_csv_path=new_csv_path, omit_model=omit_model, omit_depth=omit_depth,
                          omit_output=omit_output, omit_time=omit_time, save_features=save_features)

        return full_dataframe

    @t.guard(image_columns=t.List(t.String(allow_blank=True)) | t.String(allow_blank=True),
             image_path=t.String(allow_blank=True),
             csv_path=t.String(allow_blank=True),
             new_csv_path=t.String(allow_blank=True),
             batch_processing=t.Bool,
             batch_size=t.Int,
             save_data=t.Bool,
             save_features=t.Bool,
             save_csv=t.Bool,
             omit_time=t.Bool,
             omit_model=t.Bool,
             omit_depth=t.Bool,
             omit_output=t.Bool,
             verbose=t.Bool,
             grayscale=t.Bool
             )
    def featurize(self,
                  image_columns,
                  image_path='',
                  csv_path='',
                  new_csv_path='',
                  batch_processing=True,
                  batch_size=1000,
                  save_data=False,
                  save_features=False,
                  save_csv=False,
                  omit_time=False,
                  omit_model=False,
                  omit_depth=False,
                  omit_output=False,
                  verbose=True,
                  grayscale=False
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
            image_columns : list of str
                list of the names of the column holding the image data, if a csv exists,
                or what the name of the column will be, if generating the csv
                from a directory

            image_path : str
                the path to the folder containing the images. If using URLs, leave blank

            csv_path : str
                the path to the csv. If just using a directory, leave blank, and
                specify the path for the generated csv in new_csv_path.
                If csv exists, this is the path where the featurized csv will be
                generated.

            new_csv_path : str
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
        if not image_path and not csv_path:
            raise ValueError("Must specify either image_path or csv_path as input.")

        # Set logging level
        if verbose:
            logger.setLevel(logging.INFO)

        # Fix column headers and image path if necessary
        image_columns, image_path = _input_fixer(image_columns, image_path)

        # Find the full image dict and save the original dataframe. This is required early to know
        # how many images exist in total, to control batch processing.
        full_image_dict, df_original = _build_image_dict(image_path, csv_path,
                                                         image_columns)
        # Save the fixed inputs and full image dict
        self.df_original = df_original
        self.image_columns = image_columns
        self.image_dict = full_image_dict

        # Users can turn off batch processing by either setting batch_processing to false, or
        # setting batch_size to 0
        if batch_processing and batch_size:
            # Perform batch processing, and save the full dataframe and the full features dataframe
            features_df = self._batch_processing(full_image_dict, image_columns,
                                                 image_path, csv_path,
                                                 batch_size, grayscale)

        # If batch processing is turned off, load the images in one big batch and features them all
        else:
            logger.info("Loading full data tensor without batch processing. If you "
                        "experience a memory error, make sure batch processing is enabled.")

            full_data = self.load_data(image_columns, image_path, full_image_dict, csv_path,
                                       grayscale, save_data)

            features_df = \
                self.featurize_preloaded_data(full_data, image_columns=image_columns,
                                              features_only=True)

        # Save the full dataframe with the features
        full_df = pd.concat([df_original, features_df], axis=1)
        self.full_dataframe = full_df

        # Save features and csv if flags are enabled
        if save_features:
            self.features = features_df
        if save_csv:
            self.save_csv(new_csv_path=new_csv_path, omit_model=omit_model, omit_depth=omit_depth,
                          omit_output=omit_output, omit_time=omit_time, save_features=save_features)

        # Return the full featurized dataframe
        return full_df

    def save_csv(self, new_csv_path='', omit_model=False, omit_depth=False,
                 omit_output=False, omit_time=False, save_features=False):
        """
        """
        if self.full_dataframe.empty:
            raise AttributeError('No dataframe has been featurized.')

        # Save the name and extension separately, for robust naming
        if not new_csv_path:
            new_csv_path = self.csv_path or DEFAULT_NEW_CSV_PATH

            csv_name, ext = os.path.splitext(new_csv_path)
            name_path = _named_path_finder("{}_featurized".format(csv_name), self.model_name,
                                           self.depth, self.num_features, omit_model, omit_depth,
                                           omit_output, omit_time)
        else:
            name_path, ext = os.path.splitext(new_csv_path)

        _create_csv_path(name_path)
        logger.warning("Saving full dataframe to csv as {}{}".format(name_path, ext))
        self.full_dataframe.to_csv("{}{}".format(name_path, ext), index=False)

        if save_features:
            logger.warning("Saving features to csv as {}_features_only{}".format(name_path, ext))
            self.df_features.to_csv("{}_features_only{}".format(name_path, ext),
                                    index=False)

    @t.guard(confirm=t.Bool)
    def clear_input(self, confirm=False):
        """
        Clear all input for the model. Requires the user to confirm with an additional "confirm"
        argument in order to run.

        Parameters:
        ----------
        confirm : bool
            Users are required to modify this to true in order to clear all attributes
            from the featurizer
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
        self.image_columns = ''
        self.image_path = ''

    # ###################
    # Helper Functions! #
    # ###################

    def _load_data_helper(self,
                          model_name,
                          image_columns,
                          image_path,
                          image_dict,
                          csv_path,
                          grayscale):
        """
        This function helps load the image data from the image directory and/or csv.
        It can be called by either batch processing, where each column is handled separately in the
        parent function and the data is loaded in batches, or it can be called without batch
        processing, where the columns must each be loaded and concatenated here.

        Parameters:
        ----------
        model_name : str
            The name of the model type, which determines scaling size

        image_columns : list
            A list of the image column headers

        image_path : str
            Path to the image directory

        image_dict : dict
            This is a dictionary containing the names of each image column as a key, along with
            all of the image paths for that column.

        csv_path : str
            Path to the csv

        grayscale : bool
            Whether the images are grayscale or not
        """

        # Save size that model scales to
        scaled_size = SIZE_DICT[model_name]

        # Save the full image tensor, the path to the csv, and the list of image paths
        image_data, list_of_image_paths = \
            preprocess_data(image_columns[0], model_name,
                            image_dict[image_columns[0]],
                            image_path, csv_path, scaled_size, grayscale)

        image_data_list = [np.expand_dims(image_data, axis=0)]

        # If there is more than one image column, repeat this process for each
        if len(image_columns) > 1:
            for column in image_columns[1:]:
                image_data, list_of_image_paths = \
                    preprocess_data(column, model_name, image_dict[column], image_path,
                                    csv_path, scaled_size, grayscale)

                image_data_list.append(np.expand_dims(image_data, axis=0))

        full_image_data = np.concatenate(image_data_list)

        return scaled_size, full_image_data

    def _featurize_helper(self, features, image_columns, batch_data):
        """
        This function featurizes the data for each image column, and creates the features array
        from all of the featurized columns

        Parameters:
        ----------
        features : array
            Array of features already computed

        image_columns : list
            A list of the image column headers

        batch_data : array
            The batch loaded image data (which may be the full array if not running with batches)
        """
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
                                image_columns[column])

            features_list.append(df_features)

        df_features = pd.concat(features_list, axis=1)

        return df_features

    def _batch_processing(self,
                          full_image_dict,
                          image_columns,
                          image_path='',
                          csv_path='',
                          batch_size=1000,
                          grayscale=False):
        """
        This function handles batch processing. It takes the full list of images that need
        to be processed and loads/featurizes the images in batches.

        Parameters:
        ----------
        full_image_dict : dict
            This is a dictionary containing the names of each image column as a key, along with
            all of the image paths for that column.

        image_columns : list
            A list of the image column headers

        df_original : pandas.DataFrame
            The original dataframe (not containing the image features)

        image_path : str
            Path to the image directory

        csv_path : str
            Path to the csv

        batch_size : int
            The number of images processed per batch

        grayscale : bool
            Whether the images are grayscale or not

        """

        features_df = pd.DataFrame()
        features_df_columns_list = []
        # Iterate through each image column
        for column_index in range(len(image_columns)):
            # Initialize the batch index and save the column name
            index = 0
            batch_number = 0
            column = image_columns[column_index]
            batch_features_df = pd.DataFrame()

            # Get the list of image paths and the number of images in this column
            list_of_image_paths = full_image_dict[column]
            num_images = len(list_of_image_paths)

            batch_features_list = []
            # Loop through the images, featurizing each batch
            if len(image_columns) > 1:
                logger.info("Featurizing column #{}".format(column_index + 1))

            while index < num_images:
                tic = time.clock()

                # Cap the batch size against the total number of images left to prevent overflow
                if index + batch_size > num_images:
                    batch_size = num_images - index

                # Create a dictionary for just the batch of images
                batch_image_dict = {column: full_image_dict[column][index:index + batch_size]}

                # Load the images
                logger.info("Loading image batch.")

                batch_data = self.load_data(column, image_path,
                                            batch_image_dict, csv_path,
                                            grayscale, save_data=False)
                logger.info("\nFeaturizing image batch.")

                # If this is the first batch, the batch features will be saved alone.
                # Otherwise, they are concatenated to the last batch
                batch_features_list.append(self.featurize_preloaded_data(batch_data, column,
                                                                         features_only=True,
                                                                         batch_processing=True))

                # Increment index by batch size
                index += batch_size
                batch_number += 1

                # Give update on time and number of images left in column
                remaining_batches = int(math.ceil(num_images - index) / batch_size)

                logger.info("Featurized batch #{}. Number of images left: {}\n"
                            "Estimated total time left: {} seconds\n".format(
                                batch_number, num_images - index,
                                int((time.clock() - tic) * remaining_batches))
                            )

            # After the full column's features are calculated, concatenate them all and append them
            # to the full DataFrame list
            batch_features_df = pd.concat(batch_features_list, ignore_index=True)
            features_df_columns_list.append(batch_features_df)

        # Once all the features are created for each column, concatenate them together for both
        # the features dataframe and the full dataframe
        features_df = pd.concat(features_df_columns_list, axis=1)

        # Return the full dataframe and features dataframe
        return features_df


def _build_image_dict(image_path, csv_path, image_columns):
    """
    This function creates the image dictionary that maps each image column to the images
    in that column

    Parameters
    ----------
    image_path : str
        Path to the image directory

    csv_path : str
        Path to the csv

    image_columns : list
        A list of the image column headers
    """
    full_image_dict = {}
    for column in image_columns:
        list_of_image_paths, df = _image_paths_finder(image_path, csv_path,
                                                      column)

        full_image_dict[column] = list_of_image_paths
    return full_image_dict, df


def _input_fixer(image_columns, image_path):
    """
    This function turns image_columns into a list of a single element if there is only
    one image column. It also fixes the image path to contain a trailing `/` if the path to the
    directory is missing one.

    Parameters
    ----------
    image_columns : list
        A list of the image column headers

    image_path : str
        Path to the image directory
    """
    # Convert column header to list if it's passed a single string
    if isinstance(image_columns, str):
        image_columns = [image_columns]

    # Add backslash to end of image path if it is not there
    if image_path != '' and image_path[-1] != "/":
        image_path = '{}/'.format(image_path)

    return image_columns, image_path


def _create_csv_path(new_csv_path):
    """
    Create the necessary csv along with the appropriate directories
    """
    # Create the filepath to the new csv
    path_to_new_csv = os.path.dirname(new_csv_path)
    if not os.path.isdir(path_to_new_csv) and path_to_new_csv != '':
        os.makedirs(path_to_new_csv)


def _named_path_finder(csv_name, model_str, model_depth, model_output,
                       omit_model, omit_depth, omit_output, omit_time):
    """
    Create the named path from the robust naming configuration available.

    Parameters:
    -----------
        omit_model : Bool
            Boolean to omit the model name from the CSV name

        omit_depth : Bool
            Boolean to omit the model depth from the CSV name

        omit_output : Bool
            Boolean to omit the model output size from the CSV name

        omit_time : Bool
            Boolean to omit the time of creation from the CSV name

        model_str : Str
            The model name

        model_depth : Str
            The model depth

        model_output : Str
            The model output size

    Returns:
    --------
        named_path : Str
            The full name of the CSV file
    """
    # Naming switches! Can turn on or off to remove time, model, depth, or output size
    # from output filename
    if not omit_time:
        saved_time = "_({})".format(time.strftime("%d-%b-%Y-%H.%M.%S", time.gmtime()))
    else:
        saved_time = ""
    if not omit_model:
        saved_model = "_{}".format(model_str)
    else:
        saved_model = ""
    if not omit_depth:
        saved_depth = "_depth-{}".format(model_depth)
    else:
        saved_depth = ""
    if not omit_output:
        saved_output = "_output-{}".format(model_output)
    else:
        saved_output = ""

    named_path = "{}{}{}{}{}".format(csv_name, saved_model, saved_depth, saved_output, saved_time)
    return named_path
