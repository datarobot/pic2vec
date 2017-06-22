import pandas as pd
import numpy as np

from keras.models import Model

def featurize_data(model, array):

    #------------------------------------#
    if 'numpy' not in str(type(array)):
        raise TypeError('Must pass in a numpy array!')

    if len(array.shape) != 4:
        raise ValueError('Image array must be a 4D tensor, with dimensions: ' \
                         '[batch, height, width, channel]')

    print('Creating feature array!')
    full_feature_array = model.predict(array, verbose=1)

    print('Feature array created successfully.')
    return full_feature_array


def features_to_csv(full_feature_array, csv_path, image_column_header, image_list):
    df = pd.read_csv(csv_path)

    if image_column_header not in df.columns:
        raise ValueError('Must pass the name of the column where the images are ' \
                         'stored in the csv! The column passed was not in the csv.')

    if len(full_feature_array.shape) != 2:
        raise ValueError('Feature array must be 2D array, with shape: [batch, num_features]. ' \
                         'Gave feature array of shape: {}'.format(full_feature_array.shape))

    num_features = full_feature_array.shape[1]

    print('Adding image features to csv!')

    print num_features
    array_column_headers = ['image_feature_{}'.format(str(feature)) for feature in xrange(num_features)]
    df_features = pd.DataFrame(data=full_feature_array, columns=array_column_headers)

    df_full = pd.concat([df, df_features], axis=1)

    df_features.to_csv('{}_features_only'.format(csv_path), index=False)

    df_full.to_csv('{}_full'.format(csv_path), index=False)

    return df_full
