import os
import imghdr

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input


def preprocess_featurizer_data(datapath):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    files_in_path = next(os.walk(datapath))[2]

    count = 0
    valid = ['jpeg','bmp','png']

    for fichier in files_in_path:
        if imghdr.what(fichier) in valid:



    test_datagen.flow_from_directory(datapath,target_size=(299,299), batch_size=1,\
                                     class_mode=None, shuffle=False)
