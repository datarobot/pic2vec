"""
This is a script that is used to update test files with current versions of the scientific
libraries. Whenever scientific libraries are upgraded, this can be run to check whether predictions
have changed for any of the models, and update them if need be.
"""
import numpy as np
import pandas as pd
import logging

from tests.test_build_featurizer import INITIALIZE_MODEL_CASES, INITIALIZED_MODEL_TEST_ARRAY
from pic2vec.build_featurizer import _initialize_model
from pic2vec.enums import MODELS
from pic2vec import ImageFeaturizer

TEST_DATA_NAME = 'tests/image_featurizer_testing/csv_checking/testing_data.csv'

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

# Arrays used to test model predictions on single and multiple image columns
CHECK_ARRAY_SINGLE = 'tests/image_featurizer_testing/array_tests/check_prediction_array_{}.npy'
CHECK_ARRAY_MULT = 'tests/image_featurizer_testing/array_tests/check_prediction_array_{}_mult.npy'

# CSVs used to test model predictions on single and multiple image columns
CHECK_CSV_SINGLE = 'tests/image_featurizer_testing/csv_checking/{}_check_csv.csv'
CHECK_CSV_MULT = 'tests/image_featurizer_testing/csv_checking/{}_check_csv_mult.csv'

# This creates a dictionary mapping from each model to the required image size for the test file
MODEL_TO_IMAGE_SIZE_DICT = {model_map[0]: model_map[2] for model_map in INITIALIZE_MODEL_CASES}


def update_test_files(model, multiple_image_columns=False):
    """
    This function takes a model string as the main argument, initializes the appropriate
    ImageFeaturizer model, and uses it to predict on the test array and CSV. It logs
    whether the predictions have changed, and then updates the arrays and CSVs accordingly.

    Parameters
    ----------
    model : str
        The name of one of pic2vec's supported models

    multiple_image_columns : bool
        A boolean that determines whether to update the csvs and arrays for single or multiple
        image columns

    Returns
    -------
    None
    """
    # Only autosample if updating the csvs and arrays for multiple image columns
    f = ImageFeaturizer(model=model, autosample=multiple_image_columns)

    # Load and featurize the data corresponding to either the single or multiple image columns
    load_data = LOAD_DATA_ARGS_MULT if multiple_image_columns else LOAD_DATA_ARGS_SINGLE
    f.featurize(**load_data)

    # Updating test CSVs
    features = f.features
    test_csv = CHECK_CSV_MULT if multiple_image_columns else CHECK_CSV_SINGLE

    # Have to convert to float32
    current_csv = pd.read_csv(test_csv.format(model))
    cols = current_csv.select_dtypes(include='float64').columns
    current_csv = current_csv.astype({col: 'float32' for col in cols})

    # Check prediction consistency and update files for test CSVs if necessary
    test_csv_identical = features.equals(current_csv)
    logging.INFO("Test csv identical for {}?".format(model))
    logging.INFO(test_csv_identical)

    if not test_csv_identical:
        features.to_csv(test_csv.format(model), index=False)

    # Updating test arrays
    features = f.features.astype(float).values
    test_array = CHECK_ARRAY_MULT if multiple_image_columns else CHECK_ARRAY_SINGLE

    # Check prediction consistency and update files for test arrays if necessary
    test_array_identical = np.array_equal(features, np.load(test_array.format(model)))

    logging.INFO("Test array identical for {}?".format(model))
    logging.INFO(test_array_identical)

    if not test_array_identical:
        np.save(test_array.format(model), features)


def update_zeros_testing(model):
    """
    This function is used to update arrays in a lower-level part of testing (build_featurizer) than
    the final ImageFeaturizer. This test does not use decapitated models, but rather downloads the
    full Keras pretrained model and checks its baseline predictions on a single blank
    (i.e. all-zeros) image.

    This function initializes the model, and uses it to predict on a single blank image. It logs
    whether the predictions have changed, and then updates the test arrays if necessary.

    Parameters
    ----------
    model : str
        The name of one of pic2vec's supported models

    Returns
    -------
    None
    """

    # Create the test image to be predicted on
    m = _initialize_model(model)

    # Initialize a blank image of the appropriate size for the model
    blank_image = np.zeros(MODEL_TO_IMAGE_SIZE_DICT[model])

    # Compare the generated predictions against the existing test array, and update if necessary
    existing_test_array = np.load(INITIALIZED_MODEL_TEST_ARRAY.format(model))
    generated_array = m.predict_on_batch(blank_image)

    blank_prediction_identical = np.array_equal(generated_array, existing_test_array)

    logging.INFO("Is a blank image prediction unchanged for {}?".format(model))
    logging.INFO(blank_prediction_identical)

    if not blank_prediction_identical:
        np.save(INITIALIZED_MODEL_TEST_ARRAY.format(model), generated_array)


if __name__ == "__main__":
    for model in MODELS:
        update_test_files(model)
        update_test_files(model, multiple_image_columns=True)
        update_zeros_testing(model)
