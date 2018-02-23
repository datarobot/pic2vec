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

    3. Featurize the data. The self.featurize() function takes no input, and featurizes
       the loaded data, writing the new csvs to the same path as the loaded csv
       Also adds a binary "image_missing" column automatically, for any images that are missing
       from the image list.

    3a. Users can also load and featurize the data in one pass, with the
        self.load_and_featurize_data function, which takes the same input as the
        load_data function and performs the featurization automatically.

"""

import logging
import os

import numpy as np
import trafaret as t
import pandas as pd

from .build_featurizer import build_featurizer, supported_model_types
from .feature_preprocessing import preprocess_data, _image_paths_finder
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



        load_and_featurize_data(image_column_headers, image_path,
                                csv_path, new_csv_name, scaled_size, grayscale):
            --------------------------------
            Loads image directory and/or csv into the model, and
            featurizes the images



        load_data(image_column_headers, image_path, csv_path,
                  new_csv_name, scaled_size, grayscale):
            --------------------------------
            Loads image directory and/or csv into the model, and vectorize the
            images for input into the featurizer



        featurize():
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
        self.features = np.zeros((1))
        self.df_original = pd.DataFrame()
        self.full_dataframe = pd.DataFrame()
        self.csv_path = ''
        self.image_dict = {}
        self.image_column_headers = ''
        self.image_path = ''

        # Image scaling and cropping
        self.scaled_size = (0, 0)
        self.crop_size = (0, 0)
        self.number_crops = 0
        self.isotropic_scaling = False

    def load_and_featurize_data(self,
                                image_column_headers,
                                image_path='',
                                csv_path='',
                                new_csv_name='featurizer_csv/generated_images.csv',
                                batch_size=1000,
                                grayscale=False,
                                save_features=False,
                                omit_time=False,
                                omit_model=False,
                                omit_depth=False,
                                omit_output=False
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

        Returns:
        --------
            full_dataframe :
                Dataframe containing the features appended to the original csv.
                Also writes csvs containing the features only and the full dataframe
                to the same path as the csv containing the list of names

        """
        # Fix column headers and image path if necessary
        image_column_headers, image_path = self._input_fixer(image_column_headers, image_path)

        full_image_dict, self.df_original = self._full_image_dict_finder(image_path, csv_path,
                                                                         image_column_headers,
                                                                         new_csv_name)

        csv = self.batch_processing(full_image_dict, image_column_headers, image_path, csv_path,
                                    new_csv_name, batch_size, grayscale, save_features)

        self.save_csv(csv, omit_time=omit_time, omit_model=omit_model, omit_depth=omit_depth,
                      omit_output=omit_output)

    def batch_processing(self,
                         full_image_dict,
                         image_column_headers,
                         image_path='',
                         csv_path='',
                         new_csv_name='featurizer_csv/generated_images.csv',
                         batch_size=1000,
                         grayscale=False,
                         save_features=False):
        tot_num_images = sum(len(full_image_dict[image_list]) for image_list in full_image_dict)

        for column in full_image_dict:
            index = 0
            list_of_image_paths = full_image_dict[column]
            num_images = len(list_of_image_paths)

            while index < num_images:
                if index + batch_size > num_images:
                    batch_size = num_images - index

                # Create the directory to load the image, and
                batch_data = self.load_data(image_column_headers, image_path, full_image_dict,
                                            csv_path, new_csv_name, batch_size, grayscale,
                                            save_array=False)

                batch_features = self.featurize(batch_data, column, batch=True, save_features=False)

    # TODO: Batch processing has to first fully featurize one column, then the next, then the next
    # (if there are multiple columns). Otherwise the ordering gets fucked up.
    # Also need to figure out dict vs list stuff
    def load_data(self,
                  image_column_headers,
                  image_path='',
                  full_image_dict='',
                  csv_path='',
                  new_csv_name='featurizer_csv/generated_images_csv',
                  batch_size=1000,
                  grayscale=False,
                  save_array=True
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
        self._creating_csv_path(csv_path, image_column_headers, new_csv_name)

        # Save size that model scales to
        scaled_size = SIZE_DICT[self.model_name]

        # If the full_image_dict hasn't been passed in, build it
        if not full_image_dict:
            full_image_dict, df = self._full_image_dict_finder(image_path, csv_path,
                                                               image_column_headers, new_csv_name)

        # Save the full image tensor, the path to the csv, and the list of image paths
        (image_data, csv_path, list_of_image_paths) = \
            preprocess_data(image_column_headers[0], self.model_name,
                            full_image_dict[image_column_headers[0]],
                            image_path, csv_path, new_csv_name, scaled_size, grayscale)

        full_image_data = np.expand_dims(image_data, axis=0)

        if len(image_column_headers) > 1:
            for column in image_column_headers[1:]:
                (image_data, csv_path, list_of_image_paths) = \
                    preprocess_data(column, self.model_name, full_image_dict[column], image_path,
                                    csv_path, new_csv_name, scaled_size, grayscale)
                full_image_data = np.concatenate((full_image_data,
                                                  np.expand_dims(image_data, axis=0)))

        # Save all of the necessary data to the featurizer
        if save_array:
            self.data = full_image_data
        self.csv_path = csv_path
        self.image_dict = full_image_dict
        self.image_column_headers = image_column_headers
        self.scaled_size = scaled_size
        self.image_path = image_path
        self.df_original = df

    @t.guard(batch_data=t.Type(np.ndarray),
             image_column_headers=t.String(allow_blank=True),
             batch=t.Bool,
             save_features=t.Bool,
             save_csv=t.Bool,
             omit_model=t.Bool,
             omit_depth=t.Bool,
             omit_output=t.Bool,
             omit_time=t.Bool,
             )
    def featurize(self, batch_data=np.zeros((1)), image_column_headers='',
                  batch=False, save_features=False, save_csv=False, omit_model=False,
                  omit_depth=False, omit_output=False, omit_time=False):
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
        if batch_data == np.zeros((1)):
            batch_data = self.data
        if image_column_headers == '':
            image_column_headers = self.image_column_headers

        # Check data has been loaded, and that the data was vectorized correctly
        if np.array_equal(batch_data, np.zeros((1))):
            raise IOError('Must load data into the model first. Call load_data.')
        if batch:
            assert len(image_column_headers) == 1
        if not batch:
            assert len(image_column_headers) == self.data.shape[0]

        logging.info("Trying to featurize data.")

        # Initialize featurized data vector with appropriate size
        features = np.zeros((batch_data.shape[1],
                             self.num_features * len(image_column_headers)))

        # Save csv
        full_dataframe = self._featurize_helper(features, image_column_headers, save_features)

        if save_csv:
            save_csv(omit_model, omit_depth, omit_output, omit_time)

        return full_dataframe

    def _featurize_helper(self, features, image_column_headers, save_features):
        # For each image column, perform the full featurization and add the features to the csv
        for column in range(self.data.shape[0]):
            if not column:
                df_prev = pd.read_csv(self.csv_path)
            else:
                df_prev = self.full_dataframe

            # Featurize the data, and save it to the appropriate columns
            features[:,
                     self.num_features * column:self.num_features * column +
                     self.num_features] \
                = partial_features = featurize_data(self.featurizer, self.data[column])

            # Save the full dataframe
            self.full_dataframe, df_features = \
                create_features(self.data[column],
                                partial_features,
                                df_prev,
                                self.image_column_headers[column],
                                self.image_dict[image_column_headers[column]],
                                df_prev,
                                continued_column=bool(column),
                                save_features=save_features)

        return self.full_dataframe

    def save_csv(self, omit_model=False, omit_depth=False, omit_output=False, omit_time=False):
        # Save the name and extension separately, for robust naming
        csv_name, ext = os.path.splitext(self.csv_path)

        name_path = _named_path_finder(csv_name, self.model_name, self.depth, self.num_features,
                                       omit_model, omit_depth, omit_output, omit_time)

        self.full_dataframe.to_csv("{}{}".format(name_path, ext), index=False)

    @t.guard(confirm=t.Bool)
    def clear_input(self, confirm=False):
        """
        Clear all input for the model. Requires the user to confirm with an additional "confirm"
        argument in order to run.
        """
        if not confirm:
            logger.warning("If you're sure you would like to clear the inputs of this model, rerun"
                           " the function with the following argument: clear_input(confirm=True). "
                           "This operation cannot be reversed.")
            return

        self.data = np.zeros((1))
        self.features = np.zeros((1))
        self.full_dataframe = pd.DataFrame()
        self.csv_path = ''
        self.image_list = ''
        self.image_column_headers = ''
        self.image_path = ''

    # ###################
    # Helper Functions! #
    # ###################
    def _full_image_dict_finder(self, image_path, csv_path, image_column_headers, new_csv_name):
        full_image_dict = {}

        for column in image_column_headers:
            list_of_image_paths, df = _image_paths_finder(image_path, csv_path,
                                                          column, new_csv_name)

            full_image_dict[column] = list_of_image_paths
        print("This is the full dictionary: {}".format(full_image_dict))
        return full_image_dict, df

    def _input_fixer(self, image_column_headers, image_path):
            # Convert column header to list if it's passed a single string
        if isinstance(image_column_headers, str):
            image_column_headers = [image_column_headers]

        # Add backslash to end of image path if it is not there
        if image_path != '' and image_path[-1] != "/":
            image_path = '{}/'.format(image_path)

        return image_column_headers, image_path

    def _creating_csv_path(self, csv_path, image_column_headers, new_csv_name):
        """
        Create the necessary csv along with the appropriate directories
        """

        # If new csv_path is being generated, make sure the folder exists.
        if (csv_path == ''):
            # Raise error if multiple image columns are passed in without a csv
            if len(image_column_headers) > 1:
                raise ValueError('If building the csv from a directory, featurizer can only '
                                 'create a single image column. If two image columns are needed, '
                                 'please create a csv to pass in.')

            # Create the filepath to the new csv
            path_to_new_csv = os.path.dirname(new_csv_name)
            if not os.path.isdir(path_to_new_csv) and path_to_new_csv != '':
                os.makedirs(path_to_new_csv)
            csv_path = new_csv_name

        return csv_path
