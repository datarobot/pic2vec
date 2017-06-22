from image_featurizer.image_featurizer import ImageFeaturizer
import numpy as np
import shutil

def test_ImageFeaturizer():
    def test_featurizer_class(featurizer,
                              downsample_size,
                              image_column_header,
                              downsample,
                              csv_path,
                              image_list,
                              scaled_size,
                              depth,
                              featurized_data,
                              data):
        assert featurizer.downsample_size == downsample_size
        assert featurizer.image_column_header == image_column_header
        assert featurizer.downsample == downsample
        assert featurizer.csv_path == csv_path
        assert featurizer.image_list == image_list
        assert featurizer.scaled_size == scaled_size
        assert featurizer.depth == depth
        assert np.array_equal(featurizer.featurized_data, featurized_data)
        assert np.array_equal(featurizer.data, data)

    check_features_1 = np.load('tests/ImageFeaturizer_testing/check_prediction_array_1_2048.npy')
    check_data_array = np.load('tests/ImageFeaturizer_testing/check_data_array.npy')
    test_csv_name = 'tests/ImageFeaturizer_testing/csv_tests/generated_images_csv_test'

    # Remove path to the generated csv
    if os.path.isdir('tests/ImageFeaturizer_testing/csv_tests/')
        shutil.rmtree('tests/ImageFeaturizer_testing/csv_tests/')


    f = ImageFeaturizer()
    test_featurizer_class(f, 0, '',False, '', '', (0,0),1,np.zeros((1)),np.zeros((1)))

    f.load_data('images',image_directory_path='tests/preprocessing_testing/test_images/',\
                new_csv_name=test_csv_name)

    test_featurizer_class(f, 0, 'images',False, test_csv_name,\
        ['arendt.bmp','borges.jpg','sappho.png'], (299,299),1,np.zeros((1)),check_data_array)

    f.featurize()

    test_featurizer_class(f, 0, 'images',False, test_csv_name,\
        ['arendt.bmp','borges.jpg','sappho.png'], (299,299),1,check_features_1,check_data_array)
