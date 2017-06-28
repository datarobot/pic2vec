import os
import shutil

import numpy as np
import pytest

from image_featurizer.image_featurizer import ImageFeaturizer

TEST_CSV_NAME = 'tests/ImageFeaturizer_testing/csv_tests/generated_images_csv_test'

def compare_featurizer_class(featurizer,
                          downsample_size,
                          image_column_header,
                          automatic_downsample,
                          csv_path,
                          image_list,
                          scaled_size,
                          depth,
                          featurized_data):
    """
    This method simply checks the necessary assertions for
    a featurizer image
    """
    assert featurizer.downsample_size == downsample_size
    assert featurizer.image_column_header == image_column_header
    assert featurizer.automatic_downsample == automatic_downsample
    assert featurizer.csv_path == csv_path
    assert featurizer.image_list == image_list
    assert featurizer.scaled_size == scaled_size
    assert featurizer.depth == depth
    assert np.array_equal(featurizer.featurized_data, featurized_data)

def test_squeezenet_ImageFeaturizer():
    """
    Test the featurizer raises the necessary errors and performs its functions correctly
    """

    check_features = np.load('tests/ImageFeaturizer_testing/check_prediction_array_squeezenet.npy')

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
        f = ImageFeaturizer(automatic_downsample=(True, True))

    # Raise error if downsample_size isn't an integer
    with pytest.raises(TypeError):
        f = ImageFeaturizer(downsample_size=1.)

    # Check initialization
    f = ImageFeaturizer()
    compare_featurizer_class(f, 0, '', False, '', '', (0, 0), 1, np.zeros((1)))

    # Raise error if attempting to featurize before loading data
    with pytest.raises(IOError):
        f.featurize()

    # Check loading the data and test that the directory path works without a '/' at the end
    f.load_data('images', image_path='tests/feature_preprocessing_testing/test_images',
                new_csv_name=TEST_CSV_NAME)
    compare_featurizer_class(f, 0, 'images', False, TEST_CSV_NAME,
                          ['arendt.bmp', 'borges.jpg', 'sappho.png'], (227, 227), 1, np.zeros((1)))

    # Check featurization
    f.featurize()

    compare_featurizer_class(f, 0, 'images', False, TEST_CSV_NAME,
                          ['arendt.bmp', 'borges.jpg', 'sappho.png'], (227, 227), 1, check_features)

    # Check load and featurize at once
    f = ImageFeaturizer()
    f.load_and_featurize_data('images',
                              image_path='tests/feature_preprocessing_testing/test_images/',
                              new_csv_name=TEST_CSV_NAME)
    compare_featurizer_class(f, 0, 'images', False, TEST_CSV_NAME,
                          ['arendt.bmp', 'borges.jpg', 'sappho.png'], (227, 227), 1, check_features)

    # Remove the created csv after test finished
    shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')


def test_vgg16_ImageFeaturizer():
    """
    Test the featurizer raises the necessary errors and performs its functions correctly
    """

    check_features = np.load('tests/ImageFeaturizer_testing/check_prediction_array_vgg16.npy')

    # Remove path to the generated csv
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')

    # Check load and featurize at once
    f = ImageFeaturizer(model='vgg16')
    f.load_and_featurize_data('images',
                              image_path='tests/feature_preprocessing_testing/test_images/',
                              new_csv_name=TEST_CSV_NAME)
    compare_featurizer_class(f, 0, 'images', False, TEST_CSV_NAME,
                          ['arendt.bmp', 'borges.jpg', 'sappho.png'], (224, 224), 1, check_features)

    # Remove the created csv after test finished
    shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')


def test_vgg19_ImageFeaturizer():
    """
    Test the featurizer raises the necessary errors and performs its functions correctly
    """

    check_features = np.load('tests/ImageFeaturizer_testing/check_prediction_array_vgg19.npy')

    # Remove path to the generated csv
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')

    # Check load and featurize at once
    f = ImageFeaturizer(model='vgg19')
    f.load_and_featurize_data('images',
                              image_path='tests/feature_preprocessing_testing/test_images/',
                              new_csv_name=TEST_CSV_NAME)
    compare_featurizer_class(f, 0, 'images', False, TEST_CSV_NAME,
                          ['arendt.bmp', 'borges.jpg', 'sappho.png'], (224, 224), 1, check_features)

    # Remove the created csv after test finished
    shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')


def test_resnet50_ImageFeaturizer():
    """
    Test the featurizer raises the necessary errors and performs its functions correctly
    """

    check_features = np.load('tests/ImageFeaturizer_testing/check_prediction_array_resnet50.npy')

    # Remove path to the generated csv
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')

    # Check load and featurize at once
    f = ImageFeaturizer(model='resnet50')
    f.load_and_featurize_data('images',
                              image_path='tests/feature_preprocessing_testing/test_images/',
                              new_csv_name=TEST_CSV_NAME)
    compare_featurizer_class(f, 0, 'images', False, TEST_CSV_NAME,
                          ['arendt.bmp', 'borges.jpg', 'sappho.png'], (224, 224), 1, check_features)

    # Remove the created csv after test finished
    shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')

def test_inceptionv3_ImageFeaturizer():
    """
    Test the featurizer raises the necessary errors and performs its functions correctly
    """

    check_features = np.load('tests/ImageFeaturizer_testing/check_prediction_array_inceptionv3.npy')

    # Remove path to the generated csv
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')

    # Check load and featurize at once
    f = ImageFeaturizer(model='inceptionv3')
    f.load_and_featurize_data('images',
                              image_path='tests/feature_preprocessing_testing/test_images/',
                              new_csv_name=TEST_CSV_NAME)
    compare_featurizer_class(f, 0, 'images', False, TEST_CSV_NAME,
                          ['arendt.bmp', 'borges.jpg', 'sappho.png'], (299, 299), 1, check_features)

    # Remove the created csv after test finished
    shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')

def test_xception_ImageFeaturizer():
    """
    Test the xception featurizer performs its functions correctly
    """

    check_features = np.load('tests/ImageFeaturizer_testing/check_prediction_array_xception.npy')

    # Remove path to the generated csv
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/'):
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')

    # Check load and featurize at once
    f = ImageFeaturizer(model='xception')
    f.load_and_featurize_data('images',
                              image_path='tests/feature_preprocessing_testing/test_images/',
                              new_csv_name=TEST_CSV_NAME)
    compare_featurizer_class(f, 0, 'images', False, TEST_CSV_NAME,
                          ['arendt.bmp', 'borges.jpg', 'sappho.png'], (299, 299), 1, check_features)

    # Remove the created csv after test finished
    shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')
