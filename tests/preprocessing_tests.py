import csv
import os
import imghdr
import urllib
import filecmp
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
from keras.applications.inception_v3 import preprocess_input

random.seed(10)

from image_featurizer.feature_preprocessing import convert_single_image, _load_csv_data, \
    _find_valid_image_paths, preprocess_data, _create_csv_with_image_names


def test_convert_single_image():
    '''
    This tests that the convert_single_image method correctly loads images from
    url or from a local file, and generates the correct numpy arrays to be
    processed by the featurizer.
    '''
    image = 'tests/preprocessing_testing/test_images/borges.jpg'

    # Loading the hand-saved image tests
    test_image_1 = np.load('tests/preprocessing_testing/test_image_arrays/image_test_default.npy')
    test_image_2 = np.load('tests/preprocessing_testing/test_image_arrays/image_test_grayscale.npy')
    test_image_3 = np.load('tests/preprocessing_testing/test_image_arrays/image_test_isotropic.npy')
    test_image_4 = np.load('tests/preprocessing_testing/test_image_arrays/image_test_isotropic_grayscale.npy')

    # Converting the image from URL
    x1 = convert_single_image('from_url','http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg')
    x2 = convert_single_image('from_url','http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg', grayscale=True)
    x3 = convert_single_image('from_url','http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg', target_size=(299,467))
    x4 = convert_single_image('from_url','http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg', grayscale=True, target_size=(299,467))

    # Checking that it produces the same array
    assert np.array_equal(test_image_1, x1)
    assert np.array_equal(test_image_2, x2)
    assert np.array_equal(test_image_3, x3)
    assert np.array_equal(test_image_4, x4)

    # Creating the images from locally saved file
    x1 = convert_single_image('from_directory', image)
    x2 = convert_single_image('from_directory', image, grayscale=True)
    x3 = convert_single_image('from_directory', image, target_size=(299,467))
    x4 = convert_single_image('from_directory', image, grayscale=True, target_size=(299,467))

    # Checking that it produces the same array
    assert np.array_equal(test_image_1, x1)
    assert np.array_equal(test_image_2, x2)
    assert np.array_equal(test_image_3, x3)
    assert np.array_equal(test_image_4, x4)

def test_find_valid_image_paths():
    '''
    This takes a directory, and returns a sorted list of valid image files
    to be fed into the featurizer. Supports jpeg, bmp, or png files.
    '''

    valid_images = ['arendt.bmp', 'borges.jpg', 'sappho.png']

    image_check = _find_valid_image_paths('tests/preprocessing_testing/test_images/')

    assert image_check == valid_images

def test_create_csv_with_image_names():
    list_of_images = ['arendt.bmp', 'borges.jpg', 'sappho.png']
    new_csv_path = 'tests/preprocessing_testing/test_csv/csv_test'
    image_column_header = 'images'
    csv = _create_csv_with_image_names(list_of_images, new_csv_path, image_column_header)
    assert filecmp.cmp(new_csv_path, 'tests/preprocessing_testing/test_csv/csv_check')

if __name__ == "__main__":
    test_convert_single_image()
