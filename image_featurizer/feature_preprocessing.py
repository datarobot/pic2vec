import os
import imghdr
import urllib
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd

import numpy as np


############################################################
###----- FUNCTIONS FOR BUILDING LIST OF IMAGE PATHS -----###
############################################################

def _create_csv_with_image_paths(list_of_image_paths, new_csv_name, image_column_header):
    '''
    This takes in a list of image names, and creates a new csv file where each
    image name is a new row

    ### Parameters: ###
        list_of_image_paths: a sorted list containing each of the image names

    ### Output: ###
        None. This simply builds a csv with each row holding the name of an
        image file, and saves it to the csv_name path
    '''

    df = pd.DataFrame(list_of_image_paths, columns=[image_column_header])
    df.to_csv(new_csv_name, index=False)


def _find_directory_image_paths(image_directory):
    '''
    This takes in an directory and parses which files in it are valid images for
    loading into the featurizer.

    ### Parameters: ###
        image_directory: the filepath to the directory containing the images

    ### Output: ###
        valid_image_paths: the list of full paths to each valid image
    '''

    image_list = os.listdir(image_directory)

    valid = ['jpeg','bmp','png']
    list_of_image_paths = []

    for fichier in image_list:
        if imghdr.what(image_directory + fichier) in valid:
            list_of_image_paths.append(fichier)

    return sorted(list_of_image_paths)

def _find_csv_image_paths(csv_path, image_column_header):
    '''
    Find the image paths in a csv without an image directory

    ### Parameters: ###
        csv_path: string of the path to the csv

        image_column_header: string of the column containing the image paths

    ### Output: ###
        list_of_image_paths: a list of the image paths contained in the csv
    '''


    df = pd.read_csv(csv_path)

    #------------------------------------------------#
                ### ERROR CHECKING ###

    if not image_column_header in df.columns:
        raise ValueError('image_column_header error: ' + image_column_header +
                         ' does not exist as a column in the csv file!')
    #------------------------------------------------#

    list_of_image_paths = df[image_column_header].tolist()

    return sorted(list_of_image_paths)

def _find_combined_image_paths(image_directory_path,csv_path, image_column_header):
    '''
    Find the image paths of a csv combined with a directory: take only the overlap
    to avoid errors

    ### Parameters: ###
        image_directory_path: string of the path to the provided image directory

        csv_path: string of the path to the provided csv

        image_column_header: string of the column in the csv containing image paths

    ### Output: ###
        list_of_image_paths: list of image paths contained in both the csv and directory

    '''

    csv_list = _find_csv_image_paths(csv_path, image_column_header)

    directory_list = _find_directory_image_paths(image_directory_path)

    list_of_image_paths = []

    for path in directory_list:
        if path in csv_list:
            list_of_image_paths.append(path)

    #------------------------------------------------#
                ### ERROR CHECKING ###

    # Raise error if there are no shared images between the csv and the directory
    if len(list_of_image_paths)==0:
        raise ValueError('Something is wrong! There are no shared images in the'+
                         'csv and the image directory. Check formatting or files.')
    #------------------------------------------------#

    return sorted(list_of_image_paths)

def _image_paths_finder(image_directory_path,csv_path,image_column_header,new_csv_name):
    '''
    Given an image column header, and either a csv path or an image directory,
    find the list of image paths. If just a csv, it's pulled from the column.
    If it's just a directory, it's pulled from the directory. If it's both,
    the list is checked from the overlap between the directory and the csv.

    ### Parameters: ###
        image_directory_path: string containing path to the image directory,
                              if it exists

        csv_path: string containing the path to the csv, if it exists

        image_column_header: string to find (or create) the column holding the
                             image information

        new_csv_name: the name for the csv generated if one is not provided

    ### Output: ###
        list_of_image_paths: a sorted list of the paths to all the images being
                             featurized
    '''

    # CASE 1: They only give an image directory with no CSV
    if csv_path == None:

        # Find list of images from the image directory
        list_of_image_paths = _find_directory_image_paths(image_directory_path)

        # Warn user if new csv is generated and they haven't passed in their own
        # value.
        if new_csv_name == 'featurizer_csv/generated_images_csv':
             warning.warn('Creating new csv, but new_csv_name was not passed in.' +
                          ' Initializing to path \'./featurizer_csv/generated_images_csv\'. ')

        # Create the new csv in a folder called 'featurizer_csv/'
        _create_csv_with_image_paths(list_of_image_paths, new_csv_name=new_csv_name,\
                                    image_column_header=image_column_header)


    # CASE 2: They only give a CSV with no directory
    elif image_directory_path == None:
        # Create the list_of_image_paths from the csv
        list_of_image_paths = _find_csv_image_paths(csv_path, image_column_header)

    # CASE 3: They give both a CSV and a directory
    else:
        list_of_image_paths = _find_combined_image_paths(image_directory_path,csv_path,image_column_header)


    return list_of_image_paths



##################################################
###----- FUNCTION FOR IMAGE VECTORIZATION -----###
##################################################

def convert_single_image(image_source, image_path, target_size=(299,299), grayscale=False):
    '''
    This function takes in a path to an image (either by URL or in a native directory)
    and converts the image to a preprocessed 4D numpy array, ready to be plugged
    into the featurizer.

    ### Parameters: ###
        image_header_type: either 'from_url' or 'from_directory', depending on
                           where the images are stored
        image_path: either the URL or the full path to the image

        target size: the desired size of the image

        grayscale: a boolean indicating whether the image is grayscale or not

    ### Output: ###
        image_array: a numpy array that represents the loaded and preprocessed image
    '''

    # Retrieve the image, either from a given url or from a directory
    if image_source == 'url':
        image_file = urllib.urlretrieve(image_path)[0]
    elif image_source == 'directory':
        image_file = image_path

    # Load the image, and convert it to a numpy array with the target size
    image = load_img(image_file, target_size=target_size, grayscale=grayscale)
    image_array = img_to_array(image)

    # Expand the dimension for keras preprocessing, and preprocess the data
    # according to the InceptionV3 training that they performed.
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Return the image array
    return image_array


############################################################
###----- FUNCTION FOR END-TO-END DATA PREPROCESSING -----###
############################################################

def preprocess_data(image_column_header,
                    image_directory_path=None,
                    csv_path=None,
                    new_csv_name='featurizer_csv/generated_images_csv',
                    target_size=(299,299),
                    grayscale=False):
    '''
    This receives the data (some combination of image directory + csv), finds
    the list of valid images, and then converts each to an array and adds
    them to the full batch.

    ### Parameters: ###
        image_directory_path: the path to the image directory, if it is being passed

        csv_path: the path to the csv, if it is being passed

        image_column_header: the name of the column that contains the image paths
                             in the csv

        new_csv_name: if just being passed an image directory, this is the path
                      to save the generated csv

        target_size: the size that the images will be scaled to

        grayscale: boolean describing if the images are grayscale or not

    ### Output: ###
        full_image_data: a 4D numpy tensor containing all of the vectorized images
                        to be pushed through the featurizer

        csv_path: the path to the csv that represents the image data

        list_of_image_paths: the list of image paths in the same order as the batches
                             of the numpy tensor. This will allow us to add the
                             features to the correct row of the csv.


    '''

    #------------------------------------------------#
    ### ERROR CHECKING ###

    # If there is no image directory or csv, then something is wrong.
    if image_directory_path == None and csv_path == None:
        raise ValueError('Need to load either an image directory or a CSV with \
                         URLs, if no image directory included.')
    #------------------------------------------------#


    ### BUILDING IMAGE PATH LIST ###
    list_of_image_paths = _image_paths_finder(image_directory_path, csv_path,
                                            image_column_header, new_csv_name)


    ### IMAGE RETRIEVAL AND VECTORIZATION ###

    # Find image source: whether from url or directory
    if image_directory_path == None:
        image_source = 'url'

    else:
        image_source = 'directory'


    # Initialize the full batch
    num_images = len(list_of_image_paths)

    if grayscale:
        channels = 1
    else:
        channels = 3

    full_image_data = np.empty((num_images,target_size[0], target_size[1],channels))

    # Create the full image tensor!
    i = 0
    for image in list_of_image_paths:
        if image_directory_path != None:
            image = image_directory_path+image
        full_image_data[i,:,:,:] = convert_single_image(image_source, image, target_size = target_size)
        i += 1

    return full_image_data, csv_path, list_of_image_paths
