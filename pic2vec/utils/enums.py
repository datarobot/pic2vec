"""
This file contains a list of enums that are used across the entire pic2vec
package.
"""
# List of models supported in pic2vec
MODELS = ['squeezenet', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']

# Tolerance for prediction error
ATOL = 0.00001

# List of images used in testing
IMAGE_LIST_SINGLE = ['arendt.bmp', 'borges.jpg', 'sappho.png']
IMAGE_LIST_MULT = [['arendt.bmp', 'sappho.png', ''], ['borges.jpg', '', '']]
