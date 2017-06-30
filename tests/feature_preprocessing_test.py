"""Test feature_preprocessing module"""
import filecmp
import os
import random

import numpy as np
import pytest

from image_featurizer.feature_preprocessing import (_create_csv_with_image_paths,
                                                    _find_directory_image_paths,
                                                    _find_csv_image_paths,
                                                    _find_combined_image_paths,
                                                    _image_paths_finder, convert_single_image,
                                                    preprocess_data)

# Initialize seed to cut out any randomness (such as in image interpolation, etc)
random.seed(5102020)

# Initializing shared paths
IMAGE_PATH = 'tests/feature_preprocessing_testing/test_images/'
CSV_PATH = 'tests/feature_preprocessing_testing/csv_testing/'
IMG_COL_HEAD = 'images'
NEW_IMG_COL_HEAD = 'new_images'
IMAGE_ARRAY_PATH = 'tests/feature_preprocessing_testing/test_image_arrays/'


def test_create_csv_with_image_paths():
    """Test method creates csv correctly from list of images"""
    list_of_images = ['arendt.bmp', 'borges.jpg', 'sappho.png']
    new_csv_path = 'tests/feature_preprocessing_testing/csv_testing/generated_create_csv_test'

    assert not os.path.isfile(new_csv_path)

    _create_csv_with_image_paths(list_of_images, new_csv_path, IMG_COL_HEAD)

    assert filecmp.cmp(new_csv_path, '{}create_csv_check'.format(CSV_PATH))

    os.remove(new_csv_path)


def test_find_directory_image_paths():
    """
    Test method returns a sorted list of valid image files
    to be fed into the featurizer from a directory.
    """
    check_image_paths = ['arendt.bmp', 'borges.jpg', 'sappho.png']

    test_image_paths = _find_directory_image_paths(IMAGE_PATH)

    assert set(test_image_paths) == set(check_image_paths)


def test_find_csv_image_paths():
    """Test method correctly finds image paths in the csv, and in the right order"""
    check_image_paths = ['borges.jpg', 'arendt.bmp', 'sappho.png']
    test_image_paths = _find_csv_image_paths('{}csv_image_path_check'.format(CSV_PATH),
                                             IMG_COL_HEAD)

    with pytest.raises(ValueError):
        _find_csv_image_paths('{}csv_image_path_check'.format(CSV_PATH), 'Error Column')

    assert test_image_paths == check_image_paths


def test_find_combined_image_paths():
    """Test that method only returns images that overlap between directory and csv"""
    check_image_paths = ['', 'arendt.bmp', 'sappho.png']

    invalid_csv_image_path = 'heidegger.png'
    invalid_directory_image_path = 'borges.jpg'

    test_image_path = _find_combined_image_paths(IMAGE_PATH,
                                                 '{}directory_combined_image_path_test'.format(
                                                     CSV_PATH), IMG_COL_HEAD)

    with pytest.raises(ValueError):
        _find_combined_image_paths(IMAGE_PATH,
                                   '{}error_directory_combined_test'.format(CSV_PATH),
                                   IMG_COL_HEAD)

    assert invalid_csv_image_path not in test_image_path
    assert invalid_directory_image_path not in test_image_path

    assert check_image_paths == test_image_path


def test_convert_single_image():
    """
    Test that the convert_single_image method correctly loads images from
    url or from a local file, and generates the correct numpy arrays to be
    processed by the featurizer.
    """
    image = '{}borges.jpg'.format(IMAGE_PATH)
    image_url = 'http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg'
    # Loading the hand-saved image tests
    test_image_1 = np.load('{}image_test_default.npy'.format(IMAGE_ARRAY_PATH))
    test_image_2 = np.load('{}image_test_isotropic.npy'.format(IMAGE_ARRAY_PATH))
    test_image_3 = np.load('{}image_test_grayscale.npy'.format(IMAGE_ARRAY_PATH))
    test_image_4 = np.load('{}image_test_isotropic_grayscale.npy'.format(IMAGE_ARRAY_PATH))

    # Converting the image from URL
    converted_image_url_basic = convert_single_image('url', image_url)
    converted_image_url_isotropic = convert_single_image('url', image_url, target_size=(299, 467))
    converted_image_url_grayscale = convert_single_image('url', image_url, grayscale=True)
    converted_image_url_isotropic_grayscale = convert_single_image('url', image_url,
                                                                   target_size=(299, 467),
                                                                   grayscale=True)

    # Checking that it produces the same array
    assert np.array_equal(test_image_1, converted_image_url_basic)
    assert np.array_equal(test_image_2, converted_image_url_isotropic)
    assert np.array_equal(test_image_3, converted_image_url_grayscale)
    assert np.array_equal(test_image_4, converted_image_url_isotropic_grayscale)

    # Creating the images from locally saved file
    converted_image_directory_basic = convert_single_image('directory', image)
    converted_image_directory_isotropic = convert_single_image('directory', image,
                                                               target_size=(299, 467))
    converted_image_directory_grayscale = convert_single_image('directory', image, grayscale=True)
    converted_image_directory_isotropic_grayscale = convert_single_image('directory', image,
                                                                         target_size=(299, 467),
                                                                         grayscale=True)

    # Checking that it produces the same array
    assert np.array_equal(test_image_1, converted_image_directory_basic)
    assert np.array_equal(test_image_2, converted_image_directory_isotropic)
    assert np.array_equal(test_image_3, converted_image_directory_grayscale)
    assert np.array_equal(test_image_4, converted_image_directory_isotropic_grayscale)


def test_image_paths_finder():
    """
    Test the correct image paths returns for all three cases: directory only,
    csv only, and combined csv + directory
    """
    url_csv_path = '{}url_combined_image_path_test'.format(CSV_PATH)
    directory_csv_path = '{}directory_combined_image_path_test'.format(CSV_PATH)
    new_csv_name = '{}paths_finder_integration_test'.format(CSV_PATH)

    # check the new csv doesn't already exist
    assert not os.path.isfile(new_csv_name)

    # test image lists
    case1_images = ['arendt.bmp', 'borges.jpg', 'sappho.png']
    case2_images = ['http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg',
                    'http://queerbio.com/wiki/images/thumb/8/8d/Sappho.png/200px-Sappho.png',
                    'http://sisareport.com/wp-content/uploads/2016/09/%E2%96%B2-%ED'
                    '%95%9C%EB%82%98-%EC%95%84%EB%A0%8C%ED%8A%B8Hannah-Arendt-1906-1975.bmp']
    case3_images = ['', 'arendt.bmp', 'sappho.png']

    # generated image lists
    case1 = _image_paths_finder(IMAGE_PATH, '', NEW_IMG_COL_HEAD, new_csv_name)

    assert os.path.isfile(new_csv_name)
    # remove the generated csv
    os.remove(new_csv_name)

    case2 = _image_paths_finder('', url_csv_path, IMG_COL_HEAD, '')
    case3 = _image_paths_finder(IMAGE_PATH, directory_csv_path, IMG_COL_HEAD, '')

    # check the image lists match
    assert case1 == case1_images
    assert case2 == case2_images
    assert case3 == case3_images


def test_preprocess_data():
    """
    Full integration test: check for Type and Value errors for badly passed variables,
    and make sure that the network preprocesses data correctly for all three cases!
    """
    # Saving paths
    new_csv_name = '{}generated_preprocess_system_test'.format(CSV_PATH)
    url_csv_path = '{}url_preprocess_system_test'.format(CSV_PATH)
    directory_csv_path = '{}directory_preprocess_system_test'.format(CSV_PATH)
    error_new_csv_name = '{}generated_error_preprocess_system_test'.format(CSV_PATH)

    # saving image lists
    url_list = ['http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg',
                'http://sisareport.com/wp-content/uploads/2016/09/%E2%96%B2-%ED%95%9C%'
                'EB%82%98-%EC%95%84%EB%A0%8C%ED%8A%B8Hannah-Arendt-1906-1975.bmp',
                'http://queerbio.com/wiki/images/thumb/8/8d/Sappho.png/200px-Sappho.png'
                ]
    directory_list = ['arendt.bmp', 'borges.jpg', 'sappho.png']
    combined_list = ['', 'arendt.bmp', 'sappho.png', 'arendt.bmp']

    # Loading the pre-tested arrays
    arendt_test_array = np.load(
        'tests/feature_preprocessing_testing/test_preprocessing_arrays/arendt.npy')
    borges_test_array = np.load(
        'tests/feature_preprocessing_testing/test_preprocessing_arrays/borges.npy')
    sappho_test_array = np.load(
        'tests/feature_preprocessing_testing/test_preprocessing_arrays/sappho.npy')
    arendt_grayscale_test_array = np.load(
        'tests/feature_preprocessing_testing/test_preprocessing_arrays/arendt_grayscale.npy')
    sappho_grayscale_test_array = np.load(
        'tests/feature_preprocessing_testing/test_preprocessing_arrays/sappho_grayscale.npy')

    # -------------- #
    # ERROR CHECKING #
    # -------------- #

    # Raise error if no csv or directory is passed
    with pytest.raises(ValueError):
        preprocess_data(IMG_COL_HEAD)

    # Raise an error if image_column_header is not a string
    with pytest.raises(TypeError):
        preprocess_data(4, image_path=IMAGE_PATH, new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(3.0, image_path=IMAGE_PATH, new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(None, image_path=IMAGE_PATH, new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(True, image_path=IMAGE_PATH, new_csv_name=error_new_csv_name)

    # Raise an error if image_path is not a string
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=4, new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=3., new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=None, new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=True, new_csv_name=error_new_csv_name)

    # Raise an error if the image_path doesn't point to a real directory
    error_dir = 'egaugnalymgnidnatsrednufoerusuoyera/emdaerohwuoy/'

    try:
        assert not os.path.isdir(error_dir)
    except AssertionError:
        print('Whoops, that labyrinth exists. '
              'Change to error_dir to a directory path that does not exist.')
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=error_dir, new_csv_name=error_new_csv_name)

    # Raise an error if csv_path is not a string
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, csv_path=3, new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, csv_path=3., new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, csv_path=None, new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, csv_path=True, new_csv_name=error_new_csv_name)

    # Raise an error if the csv_path doesn't point to a file
    error_file = 'rehtonaybtmaerdecnaraeppaeremasawootehtahtdootsrednueh'

    try:
        assert not os.path.isfile(error_file)
    except AssertionError:
        print(
            'Whoops, that dreamer exists. change to error_file to a file path that does not exist.')

    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, csv_path=error_file, new_csv_name=error_new_csv_name)

    # Raise an error if new_csv_name is not a string
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH, new_csv_name=3)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH, new_csv_name=3.)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH, new_csv_name=None)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH, new_csv_name=True)

    # Raise an error if target_size is not a tuple of integers

    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH,
                        target_size=(299, 299.), new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH,
                        target_size=(299, True), new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH,
                        target_size=(None, 299), new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH,
                        target_size=299, new_csv_name=error_new_csv_name)

    # Raise error if grayscale is not a boolean
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH,
                        grayscale=4, new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH,
                        grayscale=None, new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH,
                        grayscale='True', new_csv_name=error_new_csv_name)

    # Ensure the new csv doesn't already exist, and an error csv wasn't created!
    assert not os.path.isfile(new_csv_name)
    assert not os.path.isfile(error_new_csv_name)

    # Create the full (data, csv_path, image_list) for each of the three cases
    full_preprocessed_case_1 = preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH,
                                               new_csv_name=new_csv_name)
    full_preprocessed_case_2 = preprocess_data(IMG_COL_HEAD, csv_path=url_csv_path,
                                               new_csv_name='{}breaking_test'.format(CSV_PATH))
    full_preprocessed_case_3 = preprocess_data(IMG_COL_HEAD, image_path=IMAGE_PATH,
                                               csv_path=directory_csv_path,
                                               new_csv_name='{}breaking_test'.format(CSV_PATH))

    grayscale_test = preprocess_data(IMG_COL_HEAD, grayscale=True, image_path=IMAGE_PATH,
                                     csv_path=directory_csv_path,
                                     new_csv_name='{}breaking_test'.format(CSV_PATH))

    # Ensure a new csv wasn't created when they weren't needed, and that a new csv
    # WAS created when it was needed. Then, remove the new csv.
    assert not os.path.isfile('breaking_test')
    assert os.path.isfile(new_csv_name)
    os.remove(new_csv_name)

    # CHECK ALL THE CORRECT OUTPUTS FOR CASE 1:
    # Correct number of images vectorized
    assert len(full_preprocessed_case_1[0]) == 3

    # All data vectors correctly generated
    assert np.array_equal(full_preprocessed_case_1[0][0], arendt_test_array)
    assert np.array_equal(full_preprocessed_case_1[0][1], borges_test_array)
    assert np.array_equal(full_preprocessed_case_1[0][2], sappho_test_array)

    # csv path correctly returned as non-existent, and correct image list returned
    assert full_preprocessed_case_1[1] == new_csv_name
    assert full_preprocessed_case_1[2] == directory_list

    # CHECK ALL THE CORRECT OUTPUTS FOR CASE 2:
    # Correct number of images vectorized
    assert len(full_preprocessed_case_2[0]) == 3

    # All data vectors correctly generated
    assert np.array_equal(full_preprocessed_case_2[0][1], arendt_test_array)
    assert np.array_equal(full_preprocessed_case_2[0][0], borges_test_array)
    assert np.array_equal(full_preprocessed_case_2[0][2], sappho_test_array)

    # csv path and image list correctly returned
    assert full_preprocessed_case_2[1] == url_csv_path
    assert full_preprocessed_case_2[2] == url_list

    # CHECK ALL THE CORRECT OUTPUTS FOR CASE 3:
    # Correct number of images vectorized
    assert len(full_preprocessed_case_3[0]) == 4
    assert full_preprocessed_case_3[2] == combined_list

    # All data vectors correctly generated
    assert np.array_equal(full_preprocessed_case_3[0][0],
                          np.zeros(full_preprocessed_case_3[0][0].shape))
    assert np.array_equal(full_preprocessed_case_3[0][1], arendt_test_array)
    assert np.array_equal(full_preprocessed_case_3[0][2], sappho_test_array)
    assert np.array_equal(full_preprocessed_case_3[0][3], arendt_test_array)

    # csv path and image list correctly returned
    assert full_preprocessed_case_3[1] == directory_csv_path
    assert full_preprocessed_case_3[2] == combined_list

    # CHECK ALL THE CORRECT OUTPUTS FOR GRAYSCALE TEST:
    # Correct number of images vectorized
    assert len(grayscale_test[0]) == 4

    # All data vectors correctly generated
    assert np.array_equal(grayscale_test[0][0], np.zeros(grayscale_test[0][0].shape))
    assert np.array_equal(grayscale_test[0][1], arendt_grayscale_test_array)
    assert np.array_equal(grayscale_test[0][2], sappho_grayscale_test_array)
    assert np.array_equal(grayscale_test[0][3], arendt_grayscale_test_array)

    # csv path and image list correctly returned
    assert grayscale_test[1] == directory_csv_path
    assert grayscale_test[2] == combined_list


if __name__ == "__main__":
    test_create_csv_with_image_paths()
    test_find_directory_image_paths()
    test_find_csv_image_paths()
    test_find_combined_image_paths()
    test_convert_single_image()
    test_image_paths_finder()
    test_preprocess_data()
