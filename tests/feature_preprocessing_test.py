"""Test feature_preprocessing module"""
import filecmp
import logging
import os
import random

import numpy as np
import pytest

from pic2vec.feature_preprocessing import (_create_csv_with_image_paths,
                                           _find_directory_image_paths,
                                           _find_csv_image_paths,
                                           _find_combined_image_paths,
                                           _image_paths_finder, convert_single_image,
                                           preprocess_data,
                                           natural_key)

# Initialize seed to cut out any randomness (such as in image interpolation, etc)
random.seed(5102020)

# Shared paths
IMAGE_PATH = 'tests/feature_preprocessing_testing/test_images/'
CSV_PATH = 'tests/feature_preprocessing_testing/csv_testing/'
IMAGE_ARRAY_PATH = 'tests/feature_preprocessing_testing/test_image_arrays/'
URL_PATH = '{}url_test'.format(CSV_PATH)
TEST_ARRAY = 'tests/feature_preprocessing_testing/test_preprocessing_arrays/{}.npy'

# Column headers
IMG_COL_HEAD = 'images'
NEW_IMG_COL_HEAD = 'new_images'

# Image lists for directory and url
IMAGE_LIST = ['arendt.bmp', 'borges.jpg', 'sappho.png']
URL_LIST = ['http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg',
            'http://sisareport.com/wp-content/uploads/2016/09/%E2%96%B2-%ED'
            '%95%9C%EB%82%98-%EC%95%84%EB%A0%8C%ED%8A%B8Hannah-Arendt-1906-1975.bmp',
            'http://queerbio.com/wiki/images/thumb/8/8d/Sappho.png/200px-Sappho.png'
            ]

# Preprocessing paths
DIRECTORY_CSV_PATH_PREPROCESS = '{}directory_preprocess_system_test'.format(CSV_PATH)
ERROR_NEW_CSV_NAME_PREPROCESS = '{}generated_error_preprocess_system_test'.format(CSV_PATH)
NEW_CSV_NAME_PREPROCESS = '{}generated_preprocess_system_test'.format(CSV_PATH)
COMBINED_LIST_PREPROCESS = ['', 'arendt.bmp', 'sappho.png', 'arendt.bmp']

# Loading image arrays
arendt_array = np.load(TEST_ARRAY.format('arendt'))
borges_array = np.load(TEST_ARRAY.format('borges'))
sappho_array = np.load(TEST_ARRAY.format('sappho'))
arendt_grayscale_array = np.load(TEST_ARRAY.format('arendt_grayscale'))
sappho_grayscale_array = np.load(TEST_ARRAY.format('sappho_grayscale'))

# Test arrays for build_featurizer
DIRECTORY_ARRAYS = [arendt_array, borges_array, sappho_array]
CSV_ARRAYS = [borges_array, arendt_array, sappho_array]
COMBINED_ARRAYS = [np.zeros((borges_array.shape)), arendt_array, sappho_array, arendt_array]
GRAYSCALE_ARRAYS = [np.zeros((arendt_grayscale_array.shape)), arendt_grayscale_array,
                    sappho_grayscale_array, arendt_grayscale_array]


# ---- TESTING ---- #
def test_create_csv_with_image_paths():
    """Test method creates csv correctly from list of images"""
    new_csv_path = 'tests/feature_preprocessing_testing/csv_testing/generated_create_csv_test'

    if os.path.isfile(new_csv_path):
        os.remove(new_csv_path)

    _create_csv_with_image_paths(IMAGE_LIST, new_csv_path, IMG_COL_HEAD)

    assert filecmp.cmp(new_csv_path, '{}create_csv_check'.format(CSV_PATH))

    if os.path.isfile(new_csv_path):
        os.remove(new_csv_path)


def test_natural_sort():
    """Test the natural sort function"""
    unsorted_alphanumeric = ['1.jpg', '10.jpg', '2.jpg', '15.jpg', '20.jpg', '5.jpg']
    natural_sort = ['1.jpg', '2.jpg', '5.jpg', '10.jpg', '15.jpg', '20.jpg']
    assert natural_sort == sorted(unsorted_alphanumeric, key=natural_key)


def test_find_directory_image_paths():
    """
    Test method returns a sorted list of valid image files
    to be fed into the featurizer from a directory.
    """
    test_image_paths = _find_directory_image_paths(IMAGE_PATH)

    assert test_image_paths == IMAGE_LIST


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

    test_path = _find_combined_image_paths(IMAGE_PATH,
                                           '{}directory_combined_image_path_test'
                                           .format(CSV_PATH), IMG_COL_HEAD)

    with pytest.raises(ValueError):
        _find_combined_image_paths(IMAGE_PATH,
                                   '{}error_directory_combined_test'.format(CSV_PATH),
                                   IMG_COL_HEAD)

    assert invalid_csv_image_path not in test_path
    assert invalid_directory_image_path not in test_path

    assert check_image_paths == test_path


CONVERT_IMAGE_CASES = [
                       ('url', URL_LIST[0]),
                       ('directory', '{}borges.jpg'.format(IMAGE_PATH))
                      ]
@pytest.mark.parametrize('grayscale', [None, True], ids=['RGB', 'grayscale'])
@pytest.mark.parametrize('size', [(299, 299), (299, 467)], ids=['scaled', 'isotropic'])
@pytest.mark.parametrize('image_source,image_path', CONVERT_IMAGE_CASES, ids=['url', 'directory'])
def test_convert_single_image(image_source, image_path, size, grayscale):
    """Test converting images from url and directory with options for size and grayscale"""
    iso = ''
    gscale = ''

    if size != (299, 299):
        iso = '_isotropic'
    if grayscale is not None:
        gscale = '_grayscale'

    check_array = np.load('{path}image_test{isotropic}{grayscale}.npy'
                          .format(path=IMAGE_ARRAY_PATH,
                                  isotropic=iso,
                                  grayscale=gscale))

    converted_image = convert_single_image(image_source, image_path, size, grayscale)

    assert np.array_equal(check_array, converted_image)


PATHS_FINDER_CASES = [
                      (IMAGE_PATH, '', NEW_IMG_COL_HEAD,
                       '{}paths_finder_integration_test'.format(CSV_PATH), IMAGE_LIST),

                      ('', URL_PATH, IMG_COL_HEAD, '', URL_LIST),

                      (IMAGE_PATH, '{}directory_combined_image_path_test'.format(CSV_PATH),
                       IMG_COL_HEAD, '', ['', 'arendt.bmp', 'sappho.png'])
                     ]
@pytest.mark.parametrize('image_path, csv_path, image_column_header, new_csv, check_images',
                         PATHS_FINDER_CASES, ids=['directory_only', 'csv_only', 'combined'])
def test_image_paths_finder(image_path, csv_path, image_column_header, new_csv, check_images):
    """
    Test the correct image paths returns for all three cases: directory only,
    csv only, and combined csv + directory
    """
    # check the new csv doesn't already exist
    if os.path.isfile(new_csv) and new_csv != '':
        os.remove(new_csv)

    # generated image lists
    case = _image_paths_finder(image_path, csv_path, image_column_header, new_csv)

    if new_csv != '':
        assert os.path.isfile(new_csv)
        # remove the generated csv
        os.remove(new_csv)

    # Check the image lists match
    assert case == check_images


def test_preprocess_data_no_input():
    """Raise error if no csv or directory is passed"""
    with pytest.raises(ValueError):
        preprocess_data(IMG_COL_HEAD)


def test_preprocess_data_fake_dir():
    """Raise an error if the image_path doesn't point to a real directory"""
    error_dir = 'egaugnalymgnidnatsrednufoerusuoyera/emdaerohwuoy/'
    try:
        assert not os.path.isdir(error_dir)
    except AssertionError:
        logging.error('Whoops, that labyrinth exists. '
                      'Change error_dir to a directory path that does not exist.')
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, image_path=error_dir,
                        new_csv_name=ERROR_NEW_CSV_NAME_PREPROCESS)

    assert not os.path.isfile(ERROR_NEW_CSV_NAME_PREPROCESS)


def test_preprocess_data_fake_csv():
    """Raise an error if the csv_path doesn't point to a file"""
    error_file = 'rehtonaybtmaerdecnaraeppaeremasawootehtahtdootsrednueh'
    try:
        assert not os.path.isfile(error_file)
    except AssertionError:
        logging.error(
            'Whoops, that dreamer exists. change to error_file to a file path that does not exist.')
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, csv_path=error_file,
                        new_csv_name=ERROR_NEW_CSV_NAME_PREPROCESS)

    assert not os.path.isfile(ERROR_NEW_CSV_NAME_PREPROCESS)


def compare_preprocessing(case, csv_name, check_arrays, image_list):
    """Compare a case from a full preprocessing step with the expected values of that case"""
    # Check correct number of images vectorized
    if image_list != COMBINED_LIST_PREPROCESS:
        assert len(case[0]) == 3
    else:
        assert len(case[0]) == 4

    # Check all data vectors correctly generated
    assert np.array_equal(case[0][0], check_arrays[0])
    assert np.array_equal(case[0][1], check_arrays[1])
    assert np.array_equal(case[0][2], check_arrays[2])

    # csv path correctly returned as non-existent, and correct image list returned
    assert case[1] == csv_name
    assert case[2] == image_list


PREPROCESS_DATA_CASES = [
                         (False, IMAGE_PATH, '', NEW_CSV_NAME_PREPROCESS,
                          DIRECTORY_ARRAYS, IMAGE_LIST),

                         (False, '', URL_PATH, ERROR_NEW_CSV_NAME_PREPROCESS,
                          CSV_ARRAYS, URL_LIST),

                         (False, IMAGE_PATH, DIRECTORY_CSV_PATH_PREPROCESS,
                          ERROR_NEW_CSV_NAME_PREPROCESS, COMBINED_ARRAYS,
                          COMBINED_LIST_PREPROCESS),

                         (True, IMAGE_PATH, DIRECTORY_CSV_PATH_PREPROCESS,
                          ERROR_NEW_CSV_NAME_PREPROCESS, GRAYSCALE_ARRAYS,
                          COMBINED_LIST_PREPROCESS)
                        ]
@pytest.mark.parametrize('grayscale, image_path, csv_path, new_csv_name, check_arrays, image_list',
                         PREPROCESS_DATA_CASES, ids=['dir_only', 'csv_only', 'combined', 'gray'])
def test_preprocess_data(grayscale, image_path, csv_path, new_csv_name, check_arrays, image_list):
    """
    Full integration test: check for Type and Value errors for badly passed variables,
    and make sure that the network preprocesses data correctly for all three cases.
    """
    # Ensure the new csv doesn't already exist
    if os.path.isfile(new_csv_name):
        os.remove(new_csv_name)

    # Create the full (data, csv_path, image_list) for each of the three cases
    preprocessed_case = preprocess_data(IMG_COL_HEAD, grayscale=grayscale, image_path=image_path,
                                        csv_path=csv_path, new_csv_name=new_csv_name)

    # Ensure a new csv wasn't created when they weren't needed, and that a new csv
    # WAS created when it was needed. Then, remove the new csv.
    assert not os.path.isfile(ERROR_NEW_CSV_NAME_PREPROCESS)

    if new_csv_name == NEW_CSV_NAME_PREPROCESS:
        csv_path = new_csv_name

    compare_preprocessing(preprocessed_case, csv_path, check_arrays, image_list)

    if new_csv_name == NEW_CSV_NAME_PREPROCESS:
        assert os.path.isfile(new_csv_name)
        os.remove(new_csv_name)


if __name__ == "__main__":
    test_create_csv_with_image_paths()
    test_find_directory_image_paths()
    test_find_csv_image_paths()
    test_find_combined_image_paths()
    test_convert_single_image()
    test_image_paths_finder()
    test_preprocess_data()
