from image_featurizer.image_featurizer import ImageFeaturizer
import numpy as np
import shutil
import os
import pytest

def test_ImageFeaturizer():
    '''
    Test the featurizer raises the necessary errors and performs its functions correctly
    '''

    def test_featurizer_class(featurizer,
                              downsample_size,
                              image_column_header,
                              automatic_downsample,
                              csv_path,
                              image_list,
                              scaled_size,
                              depth,
                              featurized_data,
                              data):

        '''
        This internal method simple checks the necessary assertions for
        a featurizer image
        '''
        assert featurizer.downsample_size == downsample_size
        assert featurizer.image_column_header == image_column_header
        assert featurizer.automatic_downsample == automatic_downsample
        assert featurizer.csv_path == csv_path
        assert featurizer.image_list == image_list
        assert featurizer.scaled_size == scaled_size
        assert featurizer.depth == depth
        assert np.array_equal(featurizer.featurized_data, featurized_data)
        assert np.array_equal(featurizer.data, data)

    check_features = np.load('tests/ImageFeaturizer_testing/check_prediction_array_1_2048.npy')
    check_data_array = np.load('tests/ImageFeaturizer_testing/check_data_array.npy')
    test_csv_name = 'tests/ImageFeaturizer_testing/csv_tests/generated_images_csv_test'

    # Remove path to the generated csv
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')

    # Raise error if depth is not an integer
    with pytest.raises(TypeError):
        f = ImageFeaturizer(depth=1.)
    # Raise error if depth not 1, 2, 3, or 4
    with pytest.raises(ValueError):
        f = ImageFeaturizer(depth=5)

    # Raise error if downsample isn't a boolean
    with pytest.raises(TypeError):
        f = ImageFeaturizer(automatic_downsample='True')
    with pytest.raises(TypeError):
        f = ImageFeaturizer(automatic_downsample=4)
    with pytest.raises(TypeError):
        f = ImageFeaturizer(automatic_downsample=(True,True))

    # Raise error if downsample_size isn't an integer
    with pytest.raises(TypeError):
        f = ImageFeaturizer(downsample_size = 1.)



    # Check initialization
    f = ImageFeaturizer()
    test_featurizer_class(f, 0, '',False, '', '', (0,0),1,np.zeros((1)),np.zeros((1)))

    # Raise error if attempting to featurize before loading data
    with pytest.raises(IOError):
        f.featurize()


    # Check loading the data
    f.load_data('images',image_directory_path='tests/feature_preprocessing_testing/test_images/',\
                new_csv_name=test_csv_name)
    test_featurizer_class(f, 0, 'images',False, test_csv_name,\
        ['arendt.bmp','borges.jpg','sappho.png'], (299,299),1,np.zeros((1)),check_data_array)

    # Check featurization
    f.featurize()

    test_featurizer_class(f, 0, 'images',False, test_csv_name,\
        ['arendt.bmp','borges.jpg','sappho.png'], (299,299),1,check_features,check_data_array)

    # Check load and featurize at once
    f = ImageFeaturizer()
    f.load_and_featurize_data('images',image_directory_path='tests/feature_preprocessing_testing/test_images/',\
                new_csv_name=test_csv_name)
    test_featurizer_class(f, 0, 'images',False, test_csv_name,\
        ['arendt.bmp','borges.jpg','sappho.png'], (299,299),1,check_features,check_data_array)

    # Remove the created csv after test finished
    shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')
