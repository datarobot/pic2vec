"""Test feature_preprocessing module"""
import logging
import os
import random
import pandas as pd
import numpy as np
import pytest

from tests.test_build_featurizer import ATOL
from pic2vec.feature_preprocessing import (_create_df_with_image_paths,
                                           _find_directory_image_paths,
                                           _find_csv_image_paths,
                                           _find_combined_image_paths,
                                           _image_paths_finder, _convert_single_image,
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
URL_LIST = ['https://s3.amazonaws.com/datarobot_public_datasets/images/pic2vec/borges.jpg',
            'https://s3.amazonaws.com/datarobot_public_datasets/images/pic2vec/arendt.bmp',
            'https://s3.amazonaws.com/datarobot_public_datasets/images/pic2vec/sappho.png'
            ]

# Preprocessing paths
DIRECTORY_CSV_PATH_PREPROCESS = '{}directory_preprocess_system_test'.format(CSV_PATH)
ERROR_NEW_CSV_PATH_PREPROCESS = '{}generated_error_preprocess_system_test'.format(CSV_PATH)
NEW_CSV_PATH_PREPROCESS = '{}generated_preprocess_system_test'.format(CSV_PATH)
COMBINED_LIST_PREPROCESS = ['', 'arendt.bmp', 'sappho.png', 'arendt.bmp']
ERROR_ROW_CSV = '{}error_row'.format(CSV_PATH)

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
BATCH_ARRAYS_DIR = [arendt_array, borges_array]

# ---- TESTING ---- #


def test_create_df_with_image_paths():
    """Test method creates csv correctly from list of images"""
    df = _create_df_with_image_paths(IMAGE_LIST, IMG_COL_HEAD)

    assert pd.read_csv('{}create_csv_check'.format(CSV_PATH)).equals(df)


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
    test_image_paths, df = _find_csv_image_paths('{}csv_image_path_check'.format(CSV_PATH),
                                                 IMG_COL_HEAD)

    with pytest.raises(ValueError):
        _find_csv_image_paths('{}csv_image_path_check'.format(CSV_PATH), 'Error Column')

    assert test_image_paths == check_image_paths
    assert pd.read_csv('{}csv_image_path_check'.format(CSV_PATH)).equals(df)


def test_find_combined_image_paths():
    """Test that method only returns images that overlap between directory and csv"""
    check_image_paths = ['', 'arendt.bmp', 'sappho.png']

    invalid_csv_image_path = 'heidegger.png'
    invalid_directory_image_path = 'borges.jpg'

    test_path, df = _find_combined_image_paths(IMAGE_PATH,
                                               '{}directory_combined_image_path_test'
                                               .format(CSV_PATH), IMG_COL_HEAD)

    with pytest.raises(ValueError):
        _find_combined_image_paths(IMAGE_PATH,
                                   '{}error_directory_combined_test'.format(CSV_PATH),
                                   IMG_COL_HEAD)

    assert invalid_csv_image_path not in test_path
    assert invalid_directory_image_path not in test_path

    assert check_image_paths == test_path
    assert pd.read_csv('{}directory_combined_image_path_test'.format(CSV_PATH)).equals(df)


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

    converted_image = _convert_single_image(image_source, 'xception', image_path, size, grayscale)

    assert np.allclose(check_array, converted_image, atol=ATOL)


PATHS_FINDER_CASES = [
    (IMAGE_PATH, '', NEW_IMG_COL_HEAD, IMAGE_LIST),

    ('', URL_PATH, IMG_COL_HEAD, URL_LIST),

    (IMAGE_PATH, '{}directory_combined_image_path_test'.format(CSV_PATH),
     IMG_COL_HEAD, ['', 'arendt.bmp', 'sappho.png'])
]


@pytest.mark.parametrize('image_path, csv_path, image_column_header, check_images',
                         PATHS_FINDER_CASES, ids=['directory_only', 'csv_only', 'combined'])
def test_image_paths_finder(image_path, csv_path, image_column_header, check_images):
    """
    Test the correct image paths returns for all three cases: directory only,
    csv only, and combined csv + directory
    """
    # check the new csv doesn't already exist
    # generated image lists
    case, df = _image_paths_finder(image_path, csv_path, image_column_header)

    # Check the image lists match
    assert case == check_images


def test_preprocess_data_no_input():
    """Raise error if no csv or directory is passed"""
    with pytest.raises(ValueError):
        preprocess_data(IMG_COL_HEAD, 'xception', [''])


def test_preprocess_data_fake_dir():
    """Raise an error if the image_path doesn't point to a real directory"""
    error_dir = 'egaugnalymgnidnatsrednufoerusuoyera/emdaerohwuoy/'
    try:
        assert not os.path.isdir(error_dir)
    except AssertionError:
        logging.error('Whoops, that labyrinth exists. '
                      'Change error_dir to a directory path that does not exist.')
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, 'xception', list_of_images=IMAGE_LIST, image_path=error_dir)

    assert not os.path.isfile(ERROR_NEW_CSV_PATH_PREPROCESS)


@pytest.mark.xfail
def test_preprocess_data_fake_csv():
    """Raise an error if the csv_path doesn't point to a file"""
    error_file = 'rehtonaybtmaerdecnaraeppaeremasawootehtahtdootsrednueh'
    try:
        assert not os.path.isfile(error_file)
    except AssertionError:
        logging.error(
            'Whoops, that dreamer exists. change to error_file to a file path that does not exist.')
    with pytest.raises(TypeError):
        preprocess_data(IMG_COL_HEAD, 'xception', csv_path=error_file, list_of_images=IMAGE_LIST)

    assert not os.path.isfile(ERROR_NEW_CSV_PATH_PREPROCESS)


def test_preprocess_data_invalid_url_or_dir():
    """Raise an error if the image in the column is an invalid path"""
    preprocess_data(IMG_COL_HEAD, 'xception', list_of_images=IMAGE_LIST, csv_path=ERROR_ROW_CSV)


def test_preprocess_data_invalid_model_str():
    """Raise an error if the model_str is not a valid model"""
    with pytest.raises(ValueError):
        preprocess_data(IMG_COL_HEAD, 'derp', [''], csv_path=DIRECTORY_CSV_PATH_PREPROCESS)


def compare_preprocessing(case, csv_path, check_arrays, image_list):
    """Compare a case from a full preprocessing step with the expected values of that case"""
    # Check correct number of images vectorized
    assert len(case[0]) == len(check_arrays)

    for image in range(len(check_arrays)):
        # Check all data vectors correctly generated
        assert np.allclose(case[0][image], check_arrays[image], atol=ATOL)

    # csv path correctly returned as non-existent, and correct image list returned
    assert case[1] == image_list


@pytest.mark.xfail
def test_preprocess_data_grayscale():
    # Ensure the new csv doesn't already exist
    if os.path.isfile(ERROR_NEW_CSV_PATH_PREPROCESS):
        os.remove(ERROR_NEW_CSV_PATH_PREPROCESS)

    # Create the full (data, csv_path, image_list) for each of the three cases
    preprocessed_case = preprocess_data(IMG_COL_HEAD, 'xception', grayscale=True,
                                        image_path=IMAGE_PATH,
                                        csv_path=DIRECTORY_CSV_PATH_PREPROCESS)

    # Ensure a new csv wasn't created when they weren't needed
    assert not os.path.isfile(ERROR_NEW_CSV_PATH_PREPROCESS)

    compare_preprocessing(preprocessed_case, DIRECTORY_CSV_PATH_PREPROCESS,
                          GRAYSCALE_ARRAYS, COMBINED_LIST_PREPROCESS)


PREPROCESS_DATA_CASES = [
    # Tests an image directory-only preprocessing step
    (IMAGE_PATH, '',
     DIRECTORY_ARRAYS, IMAGE_LIST),

    # Tests a CSV-only URL-based preprocessing step
    ('', URL_PATH,
     CSV_ARRAYS, URL_LIST),

    # Tests a combined directory+csv preprocessing step
    (IMAGE_PATH, DIRECTORY_CSV_PATH_PREPROCESS,
     COMBINED_ARRAYS, COMBINED_LIST_PREPROCESS),
]


@pytest.mark.parametrize('image_path, csv_path, check_arrays, image_list',
                         PREPROCESS_DATA_CASES, ids=['dir_only', 'csv_only', 'combined'])
def test_preprocess_data(image_path, csv_path, check_arrays, image_list):
    """
    Full integration test: check for Type and Value errors for badly passed variables,
    and make sure that the network preprocesses data correctly for all three cases.
    """

    # Create the full (data, csv_path, image_list) for each of the three cases
    preprocessed_case = preprocess_data(IMG_COL_HEAD, 'xception', list_of_images=image_list,
                                        grayscale=False,
                                        image_path=image_path, csv_path=csv_path)

    compare_preprocessing(preprocessed_case, csv_path, check_arrays, image_list)
