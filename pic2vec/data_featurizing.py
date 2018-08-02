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


def _create_features_df_helper(data_array, full_feature_array, image_column_header):
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
    df_features_full = pd.concat([df_missing, df_features], axis=1)

    return df_features_full


def create_features(data_array, new_feature_array, image_column_header):
    """
    Create features dataframe, and append the features to the appropriate
    rows of the original dataframe.

    Parameters:
    -----------
        data_array : np.ndarray
            The images contained in a single 2D array. Used to track missing images.

        new_feature_array : np.ndarray
            The array of generated features

        image_column_header : str
            String containing the name of the image column

    Returns:
    --------
        df_features : pandas.DataFrame
            The full dataframe containing the features appended to the dataframe of the images
    """

    # -------------- #
    # ERROR CHECKING #
    # Raise error if the data array has the wrong shape
    if len(data_array.shape) != 4:
        raise ValueError('Data array must be 4D array, with shape: [batch, height, width, channel].'
                         ' Gave feature array of shape: {}'.format(data_array.shape))

    # Raise error if the feature array has the wrong shape
    if len(new_feature_array.shape) != 2:
        raise ValueError('Feature array must be 2D array, with shape: [batch, num_features]. '
                         'Gave feature array of shape: {}'.format(new_feature_array.shape))
    # --------------------------------------- #

    logging.info('Combining image features with original dataframe.')

    df_features = _create_features_df_helper(data_array, new_feature_array,
                                             image_column_header)

    # Return the full combined dataframe
    return df_features
