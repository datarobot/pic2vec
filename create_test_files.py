import numpy as np
import pandas as pd

from tests.test_build_featurizer import INITIALIZE_MODEL_CASES, INITIALIZED_MODEL_TEST_ARRAY
from pic2vec.build_featurizer import _initialize_model
from pic2vec import ImageFeaturizer

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

CHECK_CSV_SINGLE = 'tests/image_featurizer_testing/csv_checking/{}_check_csv.csv'
CHECK_CSV_MULT = 'tests/image_featurizer_testing/csv_checking/{}_check_csv_mult.csv'

# This creates a dictionary mapping from each model to the required image size for the test file
MODEL_TO_IMAGE_SIZE_DICT = {model_map[0]: model_map[2] for model_map in INITIALIZE_MODEL_CASES}


def update_test_files(model, multiple_image_columns=False):
    # Only autosample if multiple image columns
    f = ImageFeaturizer(model=model, autosample=multiple_image_columns)

    load_data = LOAD_DATA_ARGS_MULT if multiple_image_columns else LOAD_DATA_ARGS_SINGLE
    f.featurize(**load_data)

    # Updating CSVs
    features = f.features
    test_csv = CHECK_CSV_MULT if multiple_image_columns else CHECK_CSV_SINGLE

    # Have to convert to float32
    current_csv = pd.read_csv(test_csv.format(model))

    cols = current_csv.select_dtypes(include='float64').columns
    current_csv = current_csv.astype({col: 'float32' for col in cols})

    test_csv_identical = features.equals(current_csv)

    print("Test csv identical for {}?".format(model))
    print(test_csv_identical)

    if not test_csv_identical:
        features.to_csv(test_csv.format(model), index=False)

    # Updating Arrays
    features = f.features.astype(float).values

    test_array = CHECK_ARRAY_MULT if multiple_image_columns else CHECK_ARRAY_SINGLE
    test_array_identical = np.array_equal(features, np.load(test_array.format(model)))

    print("Test array identical for {}?".format(model))
    print(test_array_identical)

    if not test_array_identical:
        np.save(test_array.format(model), features)

    return f


def update_zeros_testing(model):
    # Create the test image to be predicted on
    m = _initialize_model(model)

    blank_image = np.zeros(MODEL_TO_IMAGE_SIZE_DICT[model])

    existing_test_array = np.load(INITIALIZED_MODEL_TEST_ARRAY.format(model))
    generated_test_array = m.predict_on_batch(blank_image)

    blank_prediction_identical = np.array_equal(generated_test_array, existing_test_array)

    print("Is a blank image prediction unchanged for {}?".format(model))
    print(blank_prediction_identical)

    if not blank_prediction_identical:
        np.save(INITIALIZED_MODEL_TEST_ARRAY.format(model), generated_test_array)


if __name__ == "__main__":
    for model in MODELS:
        # update_test_files(model)
        # update_test_files(model, multiple_image_columns=True)
        update_zeros_testing(model)
