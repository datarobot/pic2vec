from pic2vec import ImageFeaturizer
import numpy as np

TEST_DATA_NAME = 'tests/image_featurizer_testing/csv_checking/testing_data.csv'

MODELS = ['squeezenet', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']

LOAD_DATA_ARGS_SINGLE = {
    'image_columns': 'images',
    'image_path': 'tests/feature_preprocessing_testing/test_images',
    'save_features': True
}

LOAD_DATA_ARGS_MULT = {
    'image_columns': ['images_1', 'images_2'],
    'image_path': 'tests/feature_preprocessing_testing/test_images',
    'csv_path': TEST_DATA_NAME,
    'save_features': True
}
CHECK_ARRAY_SINGLE = 'tests/image_featurizer_testing/array_tests/check_prediction_array_{}.npy'
CHECK_ARRAY_MULT = 'tests/image_featurizer_testing/array_tests/check_prediction_array_{}_mult.npy'


def create_single_column_numpy_arrays(model):
    """Create the prediction arrays"""
    f = ImageFeaturizer(model=model)
    f.featurize(**LOAD_DATA_ARGS_SINGLE)
    features = f.features.astype(float).values
    print("\n\nAll Close for {}?".format(model))
    print(np.array_equal(features, np.load(CHECK_ARRAY_SINGLE.format(model))))

    np.save(CHECK_ARRAY_SINGLE.format(model), features)
    return features


def create_mult_column_numpy_arrays(model):
    """Create the prediction arrays"""
    f = ImageFeaturizer(model=model, autosample=True)
    f.featurize(**LOAD_DATA_ARGS_MULT)
    features = f.features.astype(float).values
    print("\n\nAll Close for {}?".format(model))
    print(np.array_equal(features, np.load(CHECK_ARRAY_MULT.format(model))))
    print("\n\n")
    np.save(CHECK_ARRAY_MULT.format(model), features)
    return features


if __name__ == "__main__":
    for model in MODELS:
        print(model)
        features = create_single_column_numpy_arrays(model)
        features = create_mult_column_numpy_arrays(model)
