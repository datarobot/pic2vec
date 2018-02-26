"""
This file deals with featurizing the data once the featurizer has been built and the data has been
loaded and vectorized.

It allows users to featurize the data with model.predict. It also lets the featurizer write the
featurized data to the csv containing the images, appending the features to additional columns
in-line with each image row. Also adds "image_missing" columns automatically for each image_column
which contains binary values of whether the image in that row is missing.
"""

import logging
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

    # NOTE: No clue why this is here, it's to make the models note break due to
    # Keras update: https://github.com/keras-team/keras/issues/9394
    model.compile('sgd', 'mse')
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


def _create_features_df_helper(data_array, full_feature_array, image_column_header, df):
    # Log how many photos are missing or blank:
    zeros_index = [np.count_nonzero(array_slice) == 0 for array_slice in data_array[:]]
    logging.info('Number of missing photos: {}'.format(len(zeros_index)))

    # Create column headers for features, and the features dataframe
    array_column_headers = ['{}_feat_{}'.format(image_column_header, feature) for feature in
                            range(full_feature_array.shape[1])]

    df_features = pd.DataFrame(data=full_feature_array, columns=array_column_headers)

    # Create the missing column
    missing_column_header = ['{}_missing'.format(image_column_header)]
    df_missing = pd.DataFrame(data=zeros_index, columns=missing_column_header)

    # Create the full combined csv+features dataframe
    df_full = pd.concat([df, df_missing, df_features], axis=1)

    return df_full, df_features


def create_features(data_array, new_feature_array, df_prev, image_column_header,
                    image_list, continued_column=False, df_features_prev=pd.DataFrame(),
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

    # -------------- #
    # ERROR CHECKING #

    # Raise error if the image_column_header is not in the csv
    if image_column_header not in df_prev.columns:
        raise ValueError('Must pass the name of the column where the images are '
                         'stored in the csv. The column passed was not in the csv.')

    # Raise error if the data array has the wrong shape
    if len(data_array.shape) != 4:
        raise ValueError('Data array must be 4D array, with shape: [batch, height, width, channel].'
                         ' Gave feature array of shape: {}'.format(data_array.shape))

    # Raise error if the feature array has the wrong shape
    if len(new_feature_array.shape) != 2:
        raise ValueError('Feature array must be 2D array, with shape: [batch, num_features]. '
                         'Gave feature array of shape: {}'.format(new_feature_array.shape))
    # --------------------------------------- #

    logging.info('Adding image features to csv.')

    df_full, df_features = _create_features_df_helper(data_array, new_feature_array,
                                                      image_column_header, df_prev)

    if continued_column and save_features:
        df_features = pd.concat([df_features_prev, df_features], axis=1)

    # Return the full combined dataframe
    return df_full, df_features
