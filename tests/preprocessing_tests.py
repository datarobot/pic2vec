from keras.preprocessing.image import load_img, img_to_array

import numpy as np

import os
import random
import filecmp
import pytest

from image_featurizer.feature_preprocessing import _create_csv_with_image_paths,\
    _find_directory_image_paths, _find_csv_image_paths, _find_combined_image_paths,\
    _image_paths_finder,convert_single_image, preprocess_data

random.seed(5102020)

IMAGE_DIRECTORY_PATH = 'tests/preprocessing_testing/test_images/'
CSV_PATH = 'tests/preprocessing_testing/csv_testing/'
IMG_COL_HEAD = 'images'
NEW_IMG_COL_HEAD = 'new_images'
IMAGE_ARRAY_PATH = 'tests/preprocessing_testing/test_image_arrays/'

def test_create_csv_with_image_paths():
    list_of_images = ['arendt.bmp', 'borges.jpg', 'sappho.png']
    new_csv_path = 'tests/preprocessing_testing/csv_testing/generated_create_csv_test'

    assert not os.path.isfile(new_csv_path)

    _create_csv_with_image_paths(list_of_images, new_csv_path, IMG_COL_HEAD)

    assert filecmp.cmp(new_csv_path, CSV_PATH + 'create_csv_check')

    os.remove(new_csv_path)


def test_find_directory_image_paths():
    '''
    This takes a directory, and returns a sorted list of valid image files
    to be fed into the featurizer. Supports jpeg, bmp, or png files.
    '''

    valid_images = ['arendt.bmp','borges.jpg','sappho.png']
    invalid_images_order = ['borges.jpg','arendt.bmp','sappho.png']

    image_test = _find_directory_image_paths(IMAGE_DIRECTORY_PATH)

    assert image_test != invalid_images_order
    assert image_test == valid_images

def test_find_csv_image_paths():
    check_image_paths = ['arendt.bmp','borges.jpg','sappho.png']
    invalid_image_order = ['borges.jpg','arendt.bmp','sappho.png']
    test_image_paths = _find_csv_image_paths(CSV_PATH + 'csv_image_path_test', IMG_COL_HEAD)

    assert test_image_paths != invalid_image_order
    assert test_image_paths == check_image_paths

def test_find_combined_image_paths():
    check_image_paths = ['arendt.bmp','sappho.png']

    invalid_csv_image_path = 'heidegger.png'
    invalid_directory_image_path = 'borges.jpg'

    invalid_image_order = ['sappho.png','arendt.bmp']

    test_image_path = _find_combined_image_paths(IMAGE_DIRECTORY_PATH,\
                                CSV_PATH + 'directory_combined_image_path_test',IMG_COL_HEAD)

    assert invalid_csv_image_path not in test_image_path
    assert invalid_directory_image_path not in test_image_path

    assert test_image_path != invalid_image_order

    assert check_image_paths == ['arendt.bmp','sappho.png']

def test_convert_single_image():
    '''
    This tests that the convert_single_image method correctly loads images from
    url or from a local file, and generates the correct numpy arrays to be
    processed by the featurizer.
    '''
    image = IMAGE_DIRECTORY_PATH + 'borges.jpg'
    image_url = 'http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg'
    # Loading the hand-saved image tests
    test_image_1 = np.load(IMAGE_ARRAY_PATH + 'image_test_default.npy')
    test_image_2 = np.load(IMAGE_ARRAY_PATH + 'image_test_grayscale.npy')
    test_image_3 = np.load(IMAGE_ARRAY_PATH + 'image_test_isotropic.npy')
    test_image_4 = np.load(IMAGE_ARRAY_PATH + 'image_test_isotropic_grayscale.npy')

    # Converting the image from URL
    x1 = convert_single_image('url',image_url)
    x2 = convert_single_image('url',image_url, grayscale=True)
    x3 = convert_single_image('url',image_url, target_size=(299,467))
    x4 = convert_single_image('url',image_url, grayscale=True, target_size=(299,467))

    # Checking that it produces the same array
    assert np.array_equal(test_image_1, x1)
    assert np.array_equal(test_image_2, x2)
    assert np.array_equal(test_image_3, x3)
    assert np.array_equal(test_image_4, x4)

    # Creating the images from locally saved file
    x1 = convert_single_image('directory', image)
    x2 = convert_single_image('directory', image, grayscale=True)
    x3 = convert_single_image('directory', image, target_size=(299,467))
    x4 = convert_single_image('directory', image, grayscale=True, target_size=(299,467))

    # Checking that it produces the same array
    assert np.array_equal(test_image_1, x1)
    assert np.array_equal(test_image_2, x2)
    assert np.array_equal(test_image_3, x3)
    assert np.array_equal(test_image_4, x4)

def test_image_paths_finder():
    url_csv_path = CSV_PATH + 'url_combined_image_path_test'
    directory_csv_path = CSV_PATH + 'directory_combined_image_path_test'
    new_csv_name = CSV_PATH + 'paths_finder_integration_test'

    assert not os.path.isfile(new_csv_name)
    case1_images = ['arendt.bmp','borges.jpg','sappho.png']
    case2_images = ['http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg',
                    'http://queerbio.com/wiki/images/thumb/8/8d/Sappho.png/200px-Sappho.png',
                    'http://sisareport.com/wp-content/uploads/2016/09/%E2%96%B2-%ED%95%9C%EB%82%98-%EC%95%84%EB%A0%8C%ED%8A%B8Hannah-Arendt-1906-1975.bmp'
                    ]
    case3_images = ['arendt.bmp','sappho.png']

    case1 = _image_paths_finder(IMAGE_DIRECTORY_PATH,'',NEW_IMG_COL_HEAD,new_csv_name)
    case2 = _image_paths_finder('',url_csv_path,IMG_COL_HEAD, '')
    case3 = _image_paths_finder(IMAGE_DIRECTORY_PATH,directory_csv_path,IMG_COL_HEAD, '')

    assert case1 == case1_images
    assert case2 == case2_images
    assert case3 == case3_images

    os.remove(new_csv_name)
def test_preprocess_data():


    new_csv_name = CSV_PATH + 'generated_preprocess_system_test'
    url_csv_path = CSV_PATH + 'url_preprocess_system_test'
    directory_csv_path = CSV_PATH + 'directory_preprocess_system_test'
    error_new_csv_name = CSV_PATH + 'error_preprocess_system_test'
    url_list = ['http://i2.wp.com/roadsandkingdoms.com/uploads/2013/11/Jorge_Luis_Borges.jpg',
                    'http://queerbio.com/wiki/images/thumb/8/8d/Sappho.png/200px-Sappho.png',
                    'http://sisareport.com/wp-content/uploads/2016/09/%E2%96%B2-%ED%95%9C%EB%82%98-%EC%95%84%EB%A0%8C%ED%8A%B8Hannah-Arendt-1906-1975.bmp'
                    ]

    directory_list = ['arendt.bmp','borges.jpg','sappho.png']
    combined_list = ['arendt.bmp', 'sappho.png']

    arendt_test_array = np.load('tests/preprocessing_testing/test_preprocessing_arrays/arendt.npy')
    borges_test_array = np.load('tests/preprocessing_testing/test_preprocessing_arrays/borges.npy')
    sappho_test_array = np.load('tests/preprocessing_testing/test_preprocessing_arrays/sappho.npy')

    #------------------------------------------------#
                    # Error Checking

    # Raise error if no csv or directory is passed
    with pytest.raises(ValueError):
        preprocess_data(IMG_COL_HEAD)

    # Raise an error if image_column_header is not a string
    with pytest.raises(TypeError):
        error = preprocess_data(4, image_directory_path=IMAGE_DIRECTORY_PATH,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(3.0, image_directory_path=IMAGE_DIRECTORY_PATH,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(None, image_directory_path=IMAGE_DIRECTORY_PATH,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(True, image_directory_path=IMAGE_DIRECTORY_PATH,new_csv_name=error_new_csv_name)

    # Raise an error if image_directory_path is not a string
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=4,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=3.,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=None,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=True,new_csv_name=error_new_csv_name)


    # Raise an error if the image_directory_path doesn't point to a real directory
    error_dir = 'egaugnalymgnidnatsrednufoerusuoyera/emdaerohwuoy/'

    try:
        assert not os.path.isdir(error_dir)
    except AssertionError:
        print('Whoops, that labyrinth exists. Change to error_dir to a directory path that does not exist.')
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=error_dir,new_csv_name=error_new_csv_name)


    # Raise an error if csv_path is not a string
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, csv_path= 3,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, csv_path= 3.,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, csv_path= None,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, csv_path= True,new_csv_name=error_new_csv_name)

    # Raise an error if the csv_path doesn't point to a file
    error_file = 'rehtonaybtmaerdecnaraeppaeremasawootehtahtdootsrednueh'

    try:
        assert not os.path.isfile(error_file)
    except AssertionError:
        print('Whoops, that dreamer exists. change to error_file to a file path that does not exist.')
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, CSV_PATH=error_file,new_csv_name=error_new_csv_name)

    # Raise an error if new_csv_name is not a string
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH,new_csv_name=3)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH,new_csv_name=3.)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH,new_csv_name=None)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH,new_csv_name=True)


    # Raise an error if scaled_size is not a tuple of integers
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH, \
                    scaled_size=(299,299.),new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH, \
                    scaled_size=(299,True),new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH, \
                    scaled_size=(None,299),new_csv_name=error_new_csv_name)

    # Raise error if grayscale is not a boolean
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH, \
                    grayscale=4,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH, \
                    grayscale=None,new_csv_name=error_new_csv_name)
    with pytest.raises(TypeError):
        error = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH, \
                    grayscale='True',new_csv_name=error_new_csv_name)



    #------------------------------------------------#

    assert not os.path.isfile(new_csv_name)
    assert not os.path.isfile(error_new_csv_name)

    x1 = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH,new_csv_name=new_csv_name)
    x2 = preprocess_data(IMG_COL_HEAD, csv_path = url_csv_path,new_csv_name='breaking_test')
    x3 = preprocess_data(IMG_COL_HEAD, image_directory_path=IMAGE_DIRECTORY_PATH,csv_path = directory_csv_path,new_csv_name='breaking_test')


    assert not os.path.isfile('breaking_test')
    assert os.path.isfile(new_csv_name)

    assert len(x1[0])==3
    assert np.array_equal(x1[0][0], arendt_test_array)
    assert np.array_equal(x1[0][1], borges_test_array)
    assert np.array_equal(x1[0][2], sappho_test_array)
    assert x1[1] == ''
    assert x1[2] == directory_list

    assert len(x2[0])==3
    assert np.array_equal(x2[0][2], arendt_test_array)
    assert np.array_equal(x2[0][0], borges_test_array)
    assert np.array_equal(x2[0][1], sappho_test_array)
    assert x2[1] == url_csv_path
    assert x2[2] == url_list

    assert len(x3[0])==2
    assert np.array_equal(x3[0][0], arendt_test_array)
    assert np.array_equal(x3[0][1], sappho_test_array)
    assert x3[1] == directory_csv_path
    assert x3[2] == combined_list

    os.remove(new_csv_name)

if __name__ == "__main__":
    test_create_csv_with_image_paths()
    test_find_directory_image_paths()
    test_find_csv_image_paths()
    test_find_combined_image_paths()
    test_convert_single_image()
    test_image_paths_finder()
    test_preprocess_data()
