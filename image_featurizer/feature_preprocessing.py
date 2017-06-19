import os
import imghdr
import urllib

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

def retrieve_image(image_header_type, image_path):
    if image_header_type == 'from_url':
        image = urllib.urlretrieve(image_path)
    elif image_header_type == 'from_directory':
        image = image_path
def image_to_array(image_name):


def preprocess_featurizer_data(directory, image_column_header):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    files_in_path = next(os.walk(datapath))[2]

    count = 0
    valid = ['jpeg','bmp','png']

    for fichier in files_in_path:
        if imghdr.what(fichier) in valid:



    test_datagen.flow_from_directory(datapath,target_size=(299,299), batch_size=1,\
                                     class_mode=None, shuffle=False)
