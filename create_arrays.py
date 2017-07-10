from pic2vec import ImageFeaturizer
import numpy as np

MULT_CSV_NAME = 'tests/ImageFeaturizer_testing/csv_checking/mult_check_csv'
MODELS = ['squeezenet', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']
LOAD_DATA_ARGS_MULT = {
                       'image_column_headers': ['images_1', 'images_2'],
                       'image_path': 'tests/feature_preprocessing_testing/test_images',
                       'csv_path': MULT_CSV_NAME
                        }
LOAD_DATA_ARGS_SING = {
                       'image_column_headers': 'images_2',
                       'image_path': 'tests/feature_preprocessing_testing/test_images',
                       'csv_path': MULT_CSV_NAME
                        }
CHECK_ARRAY_MULT = 'tests/ImageFeaturizer_testing/array_tests/check_prediction_array_{}_mult.npy'

def create_numpy_arrays(model):
    """Create the prediction arrays"""
    f = ImageFeaturizer(model=model, auto_sample=True)
    f.load_and_featurize_data(**LOAD_DATA_ARGS_MULT)
    np.save(CHECK_ARRAY_MULT.format(model), f.featurized_data)
    return f


if __name__ == "__main__":
    for model in MODELS:
        f = create_numpy_arrays(model)
        print f.featurized_data.shape
        print f.image_column_headers
        print f.image_list
        assert np.array_equal(f.featurized_data, np.load(CHECK_ARRAY_MULT.format(model)))
