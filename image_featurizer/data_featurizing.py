import os

import pandas as pd
from keras.models import Model


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
    # ------------------------------------#
    # Error Checking

    # Raise error if it is not a numpy array
    if 'numpy' not in str(type(array)):
        raise TypeError('Must pass in a numpy array!')
    # Raise error if the array has the wrong shape
    if len(array.shape) != 4:
        raise ValueError('Image array must be a 4D tensor, with dimensions: '
                         '[batch, height, width, channel]')

    # Raise error if not passed a model
    if not isinstance(model, Model):
        raise TypeError('model must be a keras Model!')
    # ------------------------------------#

    # Perform predictions
    print('Creating feature array!')
    full_feature_array = model.predict(array, verbose=1)

    # Return features
    print('Feature array created successfully.')
    return full_feature_array


def features_to_csv(full_feature_array, csv_path, image_column_header, image_list):
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

    # ----------------------------------------#
    # Error Checking

    # Raise error if the image_column_header is not in the csv
    if image_column_header not in df.columns:
        raise ValueError('Must pass the name of the column where the images are '
                         'stored in the csv! The column passed was not in the csv.')

    # Raise error if the feature array has the wrong shape
    if len(full_feature_array.shape) != 2:
        raise ValueError('Feature array must be 2D array, with shape: [batch, num_features]. '
                         'Gave feature array of shape: {}'.format(full_feature_array.shape))
    # ----------------------------------------#

    # Save number of features
    num_features = full_feature_array.shape[1]

    print('Adding image features to csv!')

    # Create column headers for features, and the features dataframe
    array_column_headers = ['image_feature_{}'.format(str(feature)) for feature in
                            xrange(num_features)]
    df_features = pd.DataFrame(data=full_feature_array, columns=array_column_headers)

    # Create the full combined csv+features dataframe
    df_full = pd.concat([df, df_features], axis=1)

    # Save the name and extension separately, for robust naming
    csv_name, ext = os.path.splitext(csv_path)

    # Save the features dataframe to a csv without index or headers, for easy modeling
    df_features.to_csv('{}_features_only{}'.format(csv_name, ext), index=False, header=False)

    # Save the combined csv+features to a csv with no index, but with column headers
    # for DR platform
    df_full.to_csv('{}_full{}'.format(csv_name, ext), index=False)

    # Return the full combined dataframe
    return df_full
