"""
This file deals with preprocessing the images for the featurizer.

It gives the user 3 options:
1. Upload a CSV with URL pointers.
2. Upload an image directory with no CSV. The featurizer will generate a CSV automatically.
3. Upload a CSV with an image directory. The CSV will contain pointers to image in the directory.

The integrated function is the preprocess_data function, which takes in the input and
generates a 4D tensor containing the vectorized representations of the image to be featurized.
"""

from PIL import Image, ImageFile
import logging
import os
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.DEBUG = 0

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import trafaret as t  # noqa: E402
import keras.applications as ka  # noqa: E402
from keras.preprocessing.image import load_img, img_to_array  # noqa: E402

##############################################
# FUNCTIONS FOR BUILDING LIST OF IMAGE PATHS #
##############################################

# Dictionary for preprocessing algorithms
# Unnecessary 'size' entry, but leaving in case of future use...
preprocessing_dict = {
    'squeezenet': {
        'preprocess': ka.imagenet_utils.preprocess_input,
        'size': (227, 227)
    },
    'vgg16': {
        'preprocess': ka.vgg16.preprocess_input,
        'size': (224, 224)
    },
    'vgg19': {
        'preprocess': ka.vgg19.preprocess_input,
        'size': (224, 224)
    },
    'resnet50': {
        'preprocess': ka.resnet50.preprocess_input,
        'size': (224, 224)
    },
    'inceptionv3': {
        'preprocess': ka.inception_v3.preprocess_input,
        'size': (299, 299)
    },

    'xception': {
        'preprocess': ka.xception.preprocess_input,
        'size': (299, 299)
    },
}


def _create_df_with_image_paths(list_of_images, image_column_header):
    """
    Take in a list of image names, and return a DataFrame where each
    image name is a new row.

    Parameters:
    ----------
        list_of_images: list of str
            Full paths to images in a directory

        image_column_header : str
            The name of the header for the column of image paths

    Returns:
    -------
        df : pandas.DataFrame
            The dataframe containing the full list of image names.

    """
    df = pd.DataFrame(list_of_images, columns=[image_column_header])
    return df


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def _find_directory_image_paths(image_directory):
    """
    Take in a directory and parse which files in it are valid images for
    loading into the featurizer.

    List ordering explanation for directory-only featurization:
        The list will be sorted in order to create a deterministic file order for
        the generated csv, regardless of filesystem ordering. The CSV will then be
        used as the canonical order for all data preprocessing, featurizing, and
        eventually writing the features back into the csv.

    Parameters:
    ----------
        image_directory : str
            The filepath to the directory containing the images

    Returns:
    -------
        list_of_images : list of str
            A sorted list of full paths to each valid image contained in the directory

    """
    image_list = os.listdir(image_directory)

    valid = ['JPEG', 'BMP', 'PNG']
    list_of_images = []

    for fichier in image_list:
        try:
            if Image.open(image_directory + fichier).format in valid:
                list_of_images.append(fichier)
                Image.close()
        except Exception:
            pass

    return sorted(list_of_images, key=natural_key)


def _find_csv_image_paths(csv_path, image_column_header):
    """
    Find the image paths in a csv without an image directory.

    List ordering explanation for csv-included featurization:
        The list does not need to be sorted, as it is already in a set order in the csv.
        The csv will be used as the canonical order for all data preprocessing,
        featurizing, and eventually writing the features back into the csv.


    Parameters:
    ----------
        csv_path : str
            Full path to the csv

        image_column_header : str
            Name of the column containing the image paths

    Returns:
    -------
        list_of_images: list of str
            Full paths to each valid image contained in the csv

    """
    # Create the dataframe from the csv
    df = pd.read_csv(csv_path, error_bad_lines=False)

    # -------------- #
    # ERROR CHECKING #
    # Raise an error if the image column header isn't in the dataframe
    if image_column_header not in df.columns:
        raise ValueError('image_column_header error: {} does not exist as a '
                         'column in the csv file.'.format(image_column_header))
    # -------------- #

    # Create the list of image paths from the column in the dataframe
    list_of_images = df[image_column_header].tolist()

    return list_of_images, df


def _find_combined_image_paths(image_path, csv_path, image_column_header):
    """
    Find the image paths of a csv combined with a directory: take only the overlap
    to avoid errors.

    List ordering explanation for csv-included featurization:
        See docstring for _find_csv_image_paths() method.

    Parameters:
    ----------
        image_path : str
            Full path to the provided image directory

        csv_path : str
            Full path to the provided csv

        image_column_header : str
            Name of the column in the csv containing image paths

    Returns:
    -------
        list_of_images: list of str
            Full paths to each valid image contained in both the csv and directory

    """
    # Find the list of image paths in the csv
    csv_list, df = _find_csv_image_paths(csv_path, image_column_header)

    # Find the list of image paths in the directory
    directory_list = _find_directory_image_paths(image_path)

    list_of_images = []

    # Create the list of image paths by finding the overlap between the two,
    # keeping the order in the csv
    for path in csv_list:
        if path in directory_list:
            list_of_images.append(path)

        # If the image is in the csv but not the directory, input an empty string
        # as a placeholder. This image will eventually get vectorized to zeros.
        else:
            list_of_images.append('')

    # -------------- #
    # ERROR CHECKING #

    # Raise error if there are no shared images between the csv and the directory
    if all(path == '' for path in list_of_images):
        raise ValueError('Something is wrong. There are no shared images in the'
                         ' csv and the image directory. Check formatting or files.')
    # -------------- #

    return list_of_images, df


def _image_paths_finder(image_path, csv_path, image_column_header):
    """
    Given an image column header, and either a csv path or an image directory,
    find the list of image paths. If just a csv, it's pulled from the column.
    If it's just a directory, it's pulled from the directory. If it's both,
    the list is checked from the overlap between the directory and the csv.

    Parameters:
    ----------
        image_path : str
            Path to the image directory, if it exists

        csv_path : str
            Path to the csv, if it exists

        image_column_header : str
            Name of column header holding image information

    Returns:
    -------
        list_of_images : list of str
            a  list of the paths to all the images being featurized

    """
    # CASE 1: They only give an image directory with no CSV
    if csv_path == '':

        # Find list of images from the image directory
        list_of_images = _find_directory_image_paths(image_path)

        # Create the new csv in a folder called 'featurizer_csv/'
        df = _create_df_with_image_paths(list_of_images,
                                         image_column_header=image_column_header)

    # CASE 2: They only give a CSV with no directory
    elif image_path == '':
        # Create the list_of_images from the csv
        list_of_images, df = _find_csv_image_paths(csv_path, image_column_header)
        logging.info('Found image paths from csv.')

    # CASE 3: They give both a CSV and a directory
    else:
        list_of_images, df = _find_combined_image_paths(image_path, csv_path, image_column_header)
        logging.info('Found image paths that overlap between both the directory and the csv.\n')

    return list_of_images, df


#####################################
# FUNCTION FOR IMAGE VECTORIZATION #
####################################

def _convert_single_image(image_source, model_str, image_path, target_size=(299, 299),
                          grayscale=False):
    """
    Take in a path to an image (either by URL or in a native directory)
    and convert the image to a preprocessed 4D numpy array, ready to be plugged
    into the featurizer.

    Parameters:
    ----------
        image_source : str
            Flag for either url or directory source for image

        model_str : str
            Name of the model converting the image

        image_path : str
            Either the URL or the full path to the image

        target size : tuple of ints
            The desired size of the image

        grayscale : bool
            Boolean indicating whether the image is grayscale or not

    Returns:
    -------
        image_array : np.ndarray
            a numpy array that represents the loaded and preprocessed image

    """
    # Retrieve the image, either from a given url or from a directory
    try:
        if image_source == 'url':
            image_file = urlretrieve(image_path)[0]
        elif image_source == 'directory':
            image_file = image_path

    # If the image can't be retrieved, return a zeros vector of the appropriate size
    except (IOError, ValueError):
        # The channel dimension for a missing image is 3 if not grayscale, or 1 if grayscale
        im_size = target_size + (3 - 2 * grayscale,)
        logging.error('ERROR: Could not load/convert image to numpy array: {}'.format(image_path))
        return np.zeros(im_size)

    # Load the image, and convert it to a numpy array with the target size
    image = load_img(image_file, target_size=target_size, grayscale=grayscale)
    image_array = img_to_array(image)

    # Expand the dimension for keras preprocessing, and preprocess the data
    # according to the InceptionV3 training that they performed.
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocessing_dict[model_str]['preprocess'](image_array)

    # Return the image array
    return image_array


################################################
#  FUNCTION FOR END-TO-END DATA PREPROCESSING  #
################################################

def _find_image_source(image_path):

    # IMAGE RETRIEVAL AND VECTORIZATION #
    # Find image source: whether from url or directory
    if image_path == '':
        image_source = 'url'

    else:
        image_source = 'directory'

    return image_source


@t.guard(image_column_header=t.String(allow_blank=False),
         model_str=t.String(allow_blank=False),
         list_of_images=t.List(t.String(allow_blank=True)),
         image_path=t.String(allow_blank=True),
         csv_path=t.String(allow_blank=True),
         target_size=t.Tuple(t.Int, t.Int),
         grayscale=t.Bool)
def preprocess_data(image_column_header,
                    model_str,
                    list_of_images,
                    image_path='',
                    csv_path='',
                    target_size=(299, 299),
                    grayscale=False):
    """
    Receive the data (some combination of image directory + csv), find
    the list of valid images, and then convert each to an array and adds
    them to the full batch.

    Parameters:
    ----------
        image_path : str
            The path to the image directory, if it is being passed

        csv_path : str
            The path to the csv, if it is being passed

        image_column_header : str
            The name of the column that contains the image paths in the csv

        target_size : tuple of ints
            The size that the images will be scaled to

        grayscale : bool
            Boolean indicating whether the images are grayscale or not

    Returns:
    -------
        image_data : np.ndarray
            a 4D numpy tensor containing the (full or batched) vectorized images,
            ready to be pushed through the featurizer

        list_of_images : list of str
            the list of image paths in the same order as the batches
            of the numpy tensor. This will allow us to add the
            features to the correct row of the csv.

    """
    # -------------- #
    # ERROR CHECKING #
    # -------------- #

    # If there is no image directory or csv, then something is wrong.
    if image_path == '' and csv_path == '':
        raise ValueError('Need to load either an image directory or a CSV with'
                         ' URLs, if no image directory included.')
    # Raise an error if the image_path doesn't point to a directory
    if image_path and not os.path.isdir(image_path):
        raise TypeError('image_path must lead to a directory if '
                        'it is initialized. It is where the images are stored.')

    if model_str not in preprocessing_dict.keys():
        raise ValueError('model_str must be one the following: {}'.format(preprocessing_dict.keys))
    # ------------------------------------------------------ #

    # BUILDING IMAGE PATH LIST #
    num_images = len(list_of_images)

    image_source = _find_image_source(image_path)

    # Set number of grayscale channels (3 if color, 1 if grayscale)
    channels = 3 - (2 * grayscale)

    # Initialize the full batch
    image_data = np.ones((num_images, target_size[0], target_size[1], channels))

    # Create the full image tensor
    logging.info('Converting images.')

    image_dict = {}

    index = 0

    # Iterate through each image in the list of image names
    for image in list_of_images:
        # If the image is in the csv, but not in the directory, set it to all zeros
        # This allows the featurizer to correctly append features when there is
        # mismatch between the csv and the directory. Otherwise it would lose rows
        if image == '':
            image_data[index, :, :, :] = 0
            index += 1
            continue

        # If the image has already been vectorized before, just copy that slice
        if image in image_dict:
            image_data[index, :, :, :] = image_data[image_dict[image], :, :, :]

        # Otherwise, vectorize the image
        else:
            # Add the index to the dictionary to check in the future
            image_dict[image] = index

            # Append the image path to the image name. If there's none, nothing will change
            image = '{}{}'.format(image_path, image)

            # Place the vectorized image into the image data
            image_data[index, :, :, :] = _convert_single_image(image_source, model_str, image,
                                                               target_size=target_size,
                                                               grayscale=grayscale)

        # Progress report at set intervals
        if num_images < 1000:
            report_step = 100
        elif num_images < 5000:
            report_step = 500
        else:
            report_step = 1000
        if not index % report_step:
            logging.info('Converted {} images in batch. Only {} images left to go.'.format(
                index, num_images - index))

        index += 1

    return image_data, list_of_images
