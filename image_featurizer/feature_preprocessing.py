import pandas as pd
import os
import imghdr
import urllib
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.applications.inception_v3 import preprocess_input


def convert_single_image(image_header_type, image_path, target_size=(299,299), grayscale=False):
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
    if image_header_type == 'from_url':
        image_file = urllib.urlretrieve(image_path)[0]
    elif image_header_type == 'from_directory':
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

def _load_csv_data(csv_path, image_header_type, image_column_header):
    csv = pandas.read_csv(csv_path)

def _find_valid_image_paths(image_directory):
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
    valid_image_paths = []

    for fichier in image_list:
        if imghdr.what(image_directory + fichier) in valid:
            valid_image_paths.append(fichier)

    return sorted(valid_image_paths)

def _create_csv_with_image_names(list_of_images, csv_name, image_column_header):
    '''
    This takes in a list of image names, and creates a new csv file where each
    image name is a new row

    ### Parameters: ###
        list_of_images: a sorted list containing each of the image names

    ### Output: ###
        image_csv: a csv with each row holding the name of an image file
    '''

    df = pd.DataFrame(list_of_images, columns=[image_column_header])
    myfile = df.to_csv(csv_name, index=False)
    return myfile



def preprocess_data(image_header_type, image_column_header, image_directory=None, csv=None, new_csv_name='images_csv', target_size=(299,299), grayscale=False):

    if image_directory==None and csv == None:
        raise ValueError('Either need to load an image directory or a CSV (with URls, if no image directory included). ')


    # If they just give a directory, write a new CSV with one column of the filenames
    # and the rest of the featurization
    if image_directory != None:
        list_of_images = _find_valid_image_paths(image_directory)
        num_images = len(valid_image_paths)

        if csv == None:
            csv = _create_csv_with_image_names(list_of_images, csv_name=new_csv_name, image_column_header=image_column_header)



    # Initialize the full batch
    if grayscale:
        channels = 1
    else:
        channels = 3

    full_image_data = np.zeros(num_images,target_size[0], target_size[1],channels)

    # Create the full batch vectors
    i = 0
    for image in valid_image_paths:
        full_image_data[i,:,:,:] = convert_single_image(image_header_type, image, target_size = target_size)
        i += 1

    return full_image_data
