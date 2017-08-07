"""
This file deals with featurizing the data once the featurizer has been built and the data has been
loaded and vectorized.

It allows users to featurize the data with model.predict. It also lets the featurizer write the
featurized data to the csv containing the images, appending the features to additional columns
in-line with each image row. Also adds "image_missing" columns automatically for each image_column
which contains binary values of whether the image in that row is missing.
"""

import logging
import os
import time

import trafaret as t
import numpy as np
import pandas as pd

from keras.models import Model

@t.guard(model=t.Type(Model), array=t.Type(np.ndarray))
def featurize_data(model, array):
    """
    Given a model and an array, perform error checking and return the prediction
    of the full feature array.

    Parameters:
    ----------
        model : keras.models.Model
            The featurizer model performing predictions

        array : np.ndarray
            The vectorized array of images being converted into features

    Returns:
    --------
        full_feature_array : np.ndarray
            A numpy array containing the featurized images

    """
    # Raise error if the array has the wrong shape
    if len(array.shape) != 4:
        raise ValueError('Image array must be a 4D tensor, with dimensions: '
                         '[batch, height, width, channel]')

    # Perform predictions
    logging.info('Creating feature array.')
    full_feature_array = model.predict(array, verbose=1)

    # Return features
    logging.info('Feature array created successfully.')
    return full_feature_array

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


def _features_to_csv(data_array, full_feature_array, csv_path, image_column_header, image_list,
                     model_str, model_depth, model_output, omit_model=False, omit_depth=False,
                     omit_output=False, omit_time=False, continued_column=False,
                     save_features=False):
    """
    Write the feature array to a new csv, and append the features to the appropriate
    rows of the given csv.

    Parameters:
    -----------
        full_feature_array : np.ndarray
            The featurized images contained in a single 2D array

        csv_path : str
            The path to the given (or generated) csv with the image names

        image_column_header : str
            The name of the column holding the image list

        image_list : list
            List of strings containing either URLs, or pointers to images in a directory

    Returns:
    --------
        df_full : pandas.DataFrame
            The full dataframe containing the features appended to the csv of the images

        Method also writes new csvs at the path of the original csv, one containing
        just the features, and another containing the full combined csv and features

    """
    # Read the original csv
    df = pd.read_csv(csv_path)

    # -------------- #
    # ERROR CHECKING #

    # Raise error if the image_column_header is not in the csv
    if image_column_header not in df.columns:
        raise ValueError('Must pass the name of the column where the images are '
                         'stored in the csv. The column passed was not in the csv.')

    # Raise error if the data array has the wrong shape
    if len(data_array.shape) != 4:
        raise ValueError('Data array must be 4D array, with shape: [batch, height, width, channel].'
                         ' Gave feature array of shape: {}'.format(data_array.shape))

    # Raise error if the feature array has the wrong shape
    if len(full_feature_array.shape) != 2:
        raise ValueError('Feature array must be 2D array, with shape: [batch, num_features]. '
                         'Gave feature array of shape: {}'.format(full_feature_array.shape))
    # --------------------------------------- #

    # Save number of features
    num_features = full_feature_array.shape[1]

    logging.info('Adding image features to csv.')

    # Checking how many photos are missing or blank:
    zeros_index = (data_array == np.zeros((data_array.shape[1],
                                           data_array.shape[2],
                                           data_array.shape[3])))[:, 0, 0, 0]
    logging.info('Number of missing photos: {}'.format(len(zeros_index)))

    # Create column headers for features, and the features dataframe
    array_column_headers = ['{}_feat_{}'.format(image_column_header, feature) for feature in
                            range(num_features)]
    df_features = pd.DataFrame(data=full_feature_array, columns=array_column_headers)

    missing_column_header = ['{}_missing'.format(image_column_header)]
    df_missing = pd.DataFrame(data=zeros_index, columns=missing_column_header)

    # Create the full combined csv+features dataframe
    df_full = pd.concat([df, df_missing, df_features], axis=1)

    # Save the name and extension separately, for robust naming
    csv_name, ext = os.path.splitext(csv_path)

    # Find the CSV prefix with user naming configuration
    named_path = _named_path_finder(csv_name, model_str, model_depth, model_output,
                                    omit_model, omit_depth, omit_output, omit_time)

    if not continued_column:
        if save_features:
            # Save the features dataframe to a csv without index or headers, for easy modeling
            df_features.to_csv('{}_features_only{}'.format(named_path, ext),
                               index=False, header=False)

        # Save the combined csv+features to a csv with no index, but with column headers
        # for DR platform
        df_full.to_csv('{}_full{}'.format(named_path, ext), index=False)
    else:
        csv_name_orig = named_path.split('_full')[0]

        if save_features:
            features_name = '{}_features_only{}'.format(csv_name_orig, ext)

            df_features = pd.concat([pd.read_csv(features_name), df_features])
            df_features.to_csv('{}_features_only{}'.format(csv_name_orig, ext),
                               index=False, header=False)

        df_full.to_csv(csv_path, index=False)

    # Return the full combined dataframe
    return df_full
