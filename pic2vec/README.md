
# Example: Cats vs. Dogs

This notebook demonstrates the usage of ``image_featurizer`` using the Kaggle Cats vs. Dogs dataset.

We will look at the usage of the ``ImageFeaturizer()`` class, which provides a convenient pipeline to quickly tackle image problems with DataRobot's platform.

It allows users to load image data into the featurizer, and then featurizes the images into a maximum of 2048 features. It appends these features to the CSV as extra columns in line with the image rows. If no CSV was passed in with an image directory, the featurizer generates a new CSV automatically and performs the same function.



```python
import pandas as pd
import numpy as np
from sklearn import svm
from image_featurizer.image_featurizer import ImageFeaturizer
```

    Using TensorFlow backend.


## Formatting the Data

'ImageFeaturizer' accepts as input either:
1. An image directory
2. A CSV with URL pointers to image downloads, or
3. A combined image directory + CSV with pointers to the included images.

For this example, we will load in the Kaggle Cats vs. Dogs dataset of 25,000 images, along with a CSV that includes each images class label.


```python
pd.options.display.max_rows = 10

image_path = 'cat_vs_dogs_images/'
csv_path = 'cat_vs_dog_classes.csv'

pd.read_csv(csv_path)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>images</th>
      <th>animal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat.0.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat.1.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat.10.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat.100.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat.1000.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>dog.9995.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>dog.9996.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>dog.9997.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>dog.9998.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>dog.9999.jpg</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 2 columns</p>
</div>



The image directory contains 12,500 images of cats and 12,500 images of dogs. The CSV contains pointers to each image in the directory, along with a class label (0 for cats, 1 for dogs).

## Initializing the Featurizer

We will now initialize the ImageFeaturizer( ) class with a few parameters that define the model. If in doubt, we can always call the featurizer with no parameters, and it will initialize itself to a cookie-cutter build. Here, we will call the parameters explicitly to demonstrate functionality. In general, however, the model defaults to a depth of 1 and automatic downsampling, so no parameters are necessary for this build.

Because we have not downloaded the pre-trained weights, the featurizer will automatically download the weights through the Keras backend.

The depth indicates how far down we should cut the model to draw abstract features– the further down we cut, the less complex the representations will be, but they may also be less specialized to the specific classes in the ImageNet dataset that the model was trained on.

Automatic downsampling means that this model will downsample the final layer from 2048 features to 1024 features, which is a more compact representation. With large datasets, more features may run into memory problems or difficulty optimizing, so it may be worth downsampling to a smaller featurspace.


```python
featurizer = ImageFeaturizer(depth=1, automatic_downsample = False, model='squeezenet')
```


    Building the featurizer.

    Model initialized and weights loaded successfully.
    Model decapitated.
    Model downsampled.
    Full featurizer is built.
    No downsampling. Final layer feature space has size 512


This featurizer was 'decapitated' to the first layer below the prediction layer, which will produce complex representations. Because it is so close to the final prediction layer, it will create more specialized feature representations, and therefore will be better suited for image datasets that are similar to classes within the original ImageNet dataset. Cats and dogs are present within ImageNet, so a depth of 1 should perform well.

## Loading the Data

Now that the featurizer is built, we can load our data into the network. This will parse through the images in the order given by the csv, rescale them to a target size with a default of (299, 299), and build a 4D tensor containing the vectorized representations of the images. This tensor will later be fed into the network in order to be featurized.

The tensor will have the dimensions: [number of images, height, width, color channels]. In this case, the image tensor will have size [25000, 299, 299, 3].

We have to pass in the name of the column in the CSV that contains pointers to the images, as well as the path to the image directory and the path to the CSV itself, which are both saved from earlier.

If there are images in the directory that aren't in the CSV, or image names in the CSV that aren't in the directory, or even files that aren't valid image files in the directory, don't fear– the featurizer will only try to vectorize valid images that are in both the CSV and the directory. Any images present in the CSV but not the directory will be given zero vectors, and the order of the CSV is considered the canonical order for the images.


```python
featurizer.load_data('images', image_path = image_path, csv_path = csv_path)
```

    Found image paths that overlap between both the directory and the csv.
    Converting images.
    Converted 0 images. Only 25000 images left to go.
    Converted 1000 images. Only 24000 images left to go.
    Converted 2000 images. Only 23000 images left to go.
    Converted 3000 images. Only 22000 images left to go.
    Converted 4000 images. Only 21000 images left to go.
    Converted 5000 images. Only 20000 images left to go.
    Converted 6000 images. Only 19000 images left to go.
    Converted 7000 images. Only 18000 images left to go.
    Converted 8000 images. Only 17000 images left to go.
    Converted 9000 images. Only 16000 images left to go.
    Converted 10000 images. Only 15000 images left to go.
    Converted 11000 images. Only 14000 images left to go.
    Converted 12000 images. Only 13000 images left to go.
    Converted 13000 images. Only 12000 images left to go.
    Converted 14000 images. Only 11000 images left to go.
    Converted 15000 images. Only 10000 images left to go.
    Converted 16000 images. Only 9000 images left to go.
    Converted 17000 images. Only 8000 images left to go.
    Converted 18000 images. Only 7000 images left to go.
    Converted 19000 images. Only 6000 images left to go.
    Converted 20000 images. Only 5000 images left to go.
    Converted 21000 images. Only 4000 images left to go.
    Converted 22000 images. Only 3000 images left to go.
    Converted 23000 images. Only 2000 images left to go.
    Converted 24000 images. Only 1000 images left to go.


The image data has now been loaded into the featurizer and vectorized, and is ready for featurization. We can check the size, format, and other stored information about the data by calling the featurizer object attributes:


```python
print('Vectorized data shape: {}'.format(featurizer.data.shape))

print('CSV path: \'{}\''.format(featurizer.csv_path))

print('Image directory path: \'{}\''.format(featurizer.image_path))
```

    Vectorized data shape: (25000, 227, 227, 3)
    CSV path: 'cat_vs_dog_classes.csv'
    Image directory path: 'cat_vs_dogs_images/'


For a full list of attributes, call:


```python
featurizer.__dict__.keys()
```




    ['downsample_size',
     'visualize',
     'image_path',
     'image_column_header',
     'csv_path',
     'number_crops',
     'image_list',
     'featurizer',
     'scaled_size',
     'depth',
     'automatic_downsample',
     'featurized_data',
     'isotropic_scaling',
     'data',
     'model_name',
     'crop_size']



## Featurizing the Data

Now that the data is loaded, we're ready to featurize the data. This will push the vectorized images through the network and save the 2D matrix output– each row representing a single image, and each column storing a different feature.

It will then create and save a new CSV by appending these features to the end of the given CSV in line with each image's row. The features themselves will also be saved in a separate CSV file without the image names or other data. Both generated CSVs will be saved to the same path as the original CSV, with the features-only CSV appending '_features_only' and the combined CSV appending '_full' to the end of their respective filenames.

The featurize( ) method requires no parameters, as it uses the data we just loaded into the network. This requires pushing images through the deep InceptionV3 network, and so relatively large datasets will require a GPU to perform in a reasonable amount of time. Using a mid-range GPU, it can take about 30 minutes to process the full 25,000 photos in the Dogs vs. Cats.


```python
featurizer.featurize()
```

    Checking array initialized.
    Trying to featurize data.
    Creating feature array.
    25000/25000 [==============================] - 964s   
    Feature array created successfully.
    Adding image features to csv.





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>images</th>
      <th>animal</th>
      <th>image_feature_0</th>
      <th>image_feature_1</th>
      <th>image_feature_2</th>
      <th>image_feature_3</th>
      <th>image_feature_4</th>
      <th>image_feature_5</th>
      <th>image_feature_6</th>
      <th>image_feature_7</th>
      <th>...</th>
      <th>image_feature_502</th>
      <th>image_feature_503</th>
      <th>image_feature_504</th>
      <th>image_feature_505</th>
      <th>image_feature_506</th>
      <th>image_feature_507</th>
      <th>image_feature_508</th>
      <th>image_feature_509</th>
      <th>image_feature_510</th>
      <th>image_feature_511</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat.0.jpg</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.976163</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.326453</td>
      <td>0.125493</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>20.787928</td>
      <td>0.000000</td>
      <td>0.827707</td>
      <td>0.072728</td>
      <td>0.123821</td>
      <td>0.361559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat.1.jpg</td>
      <td>0</td>
      <td>0.020484</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.537819</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.626090</td>
      <td>0.132664</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>19.593006</td>
      <td>0.000914</td>
      <td>1.273858</td>
      <td>0.112284</td>
      <td>0.076074</td>
      <td>0.626272</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat.10.jpg</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002142</td>
      <td>3.562325</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.558218</td>
      <td>0.247748</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>15.065393</td>
      <td>0.004629</td>
      <td>1.506223</td>
      <td>0.357541</td>
      <td>0.225619</td>
      <td>0.313667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat.100.jpg</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.003331</td>
      <td>0.000000</td>
      <td>2.565698</td>
      <td>0.003193</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000356</td>
      <td>...</td>
      <td>1.230276</td>
      <td>0.587366</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>16.498240</td>
      <td>0.000000</td>
      <td>0.457677</td>
      <td>0.245788</td>
      <td>0.243526</td>
      <td>0.209688</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat.1000.jpg</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687325</td>
      <td>0.166234</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.270655</td>
      <td>0.075184</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>17.514622</td>
      <td>0.000000</td>
      <td>0.854873</td>
      <td>0.008352</td>
      <td>0.016658</td>
      <td>0.344797</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>dog.9995.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.193059</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.268092</td>
      <td>0.177756</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>20.336287</td>
      <td>0.000000</td>
      <td>0.896977</td>
      <td>0.149458</td>
      <td>0.123787</td>
      <td>0.384676</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>dog.9996.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.003674</td>
      <td>0.000000</td>
      <td>2.768917</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.777494</td>
      <td>0.171866</td>
      <td>0.0</td>
      <td>0.001702</td>
      <td>17.666206</td>
      <td>0.000000</td>
      <td>0.779997</td>
      <td>0.034684</td>
      <td>0.133014</td>
      <td>1.652294</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>dog.9997.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.627975</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.797773</td>
      <td>0.114264</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>20.120745</td>
      <td>0.000000</td>
      <td>0.980927</td>
      <td>0.005060</td>
      <td>0.006118</td>
      <td>0.623514</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>dog.9998.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.044554</td>
      <td>0.000000</td>
      <td>1.563408</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.939854</td>
      <td>0.059493</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>18.924351</td>
      <td>0.000000</td>
      <td>1.038141</td>
      <td>0.069819</td>
      <td>0.069094</td>
      <td>0.904590</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>dog.9999.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000018</td>
      <td>1.050343</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.311641</td>
      <td>0.145927</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>19.418907</td>
      <td>0.000000</td>
      <td>1.413191</td>
      <td>0.014721</td>
      <td>0.111561</td>
      <td>0.944223</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 514 columns</p>
</div>



## Results

The dataset has now been fully featurized. The features are saved under the featurized_data attribute:


```python
featurizer.featurized_data
```




    array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
              7.27284253e-02,   1.23820536e-01,   3.61558944e-01],
           [  2.04837546e-02,   0.00000000e+00,   0.00000000e+00, ...,
              1.12284146e-01,   7.60741457e-02,   6.26272500e-01],
           [  0.00000000e+00,   0.00000000e+00,   2.14172155e-03, ...,
              3.57541353e-01,   2.25618958e-01,   3.13666701e-01],
           ...,
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
              5.05957752e-03,   6.11788966e-03,   6.23513699e-01],
           [  0.00000000e+00,   4.45538573e-02,   0.00000000e+00, ...,
              6.98191002e-02,   6.90938309e-02,   9.04590428e-01],
           [  0.00000000e+00,   0.00000000e+00,   1.77303536e-05, ...,
              1.47210872e-02,   1.11560710e-01,   9.44222987e-01]], dtype=float32)



The full data has also been successfully saved in CSV form, which allows it to be dropped directly into the DataRobot app:


```python
pd.read_csv('cat_vs_dog_classes_full.csv')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>images</th>
      <th>Animal</th>
      <th>image_feature_0</th>
      <th>image_feature_1</th>
      <th>image_feature_2</th>
      <th>image_feature_3</th>
      <th>image_feature_4</th>
      <th>image_feature_5</th>
      <th>image_feature_6</th>
      <th>image_feature_7</th>
      <th>...</th>
      <th>image_feature_502</th>
      <th>image_feature_503</th>
      <th>image_feature_504</th>
      <th>image_feature_505</th>
      <th>image_feature_506</th>
      <th>image_feature_507</th>
      <th>image_feature_508</th>
      <th>image_feature_509</th>
      <th>image_feature_510</th>
      <th>image_feature_511</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat.0.jpg</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.976163</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.326453</td>
      <td>0.125493</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>20.787928</td>
      <td>0.000000</td>
      <td>0.827707</td>
      <td>0.072728</td>
      <td>0.123821</td>
      <td>0.361559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat.1.jpg</td>
      <td>0</td>
      <td>0.020484</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.537819</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.626090</td>
      <td>0.132664</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>19.593006</td>
      <td>0.000914</td>
      <td>1.273858</td>
      <td>0.112284</td>
      <td>0.076074</td>
      <td>0.626272</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat.10.jpg</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002142</td>
      <td>3.562325</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.558218</td>
      <td>0.247748</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>15.065393</td>
      <td>0.004629</td>
      <td>1.506223</td>
      <td>0.357541</td>
      <td>0.225619</td>
      <td>0.313667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat.100.jpg</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.003331</td>
      <td>0.000000</td>
      <td>2.565698</td>
      <td>0.003193</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000356</td>
      <td>...</td>
      <td>1.230276</td>
      <td>0.587366</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>16.498240</td>
      <td>0.000000</td>
      <td>0.457677</td>
      <td>0.245788</td>
      <td>0.243526</td>
      <td>0.209688</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat.1000.jpg</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687325</td>
      <td>0.166234</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.270655</td>
      <td>0.075184</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>17.514622</td>
      <td>0.000000</td>
      <td>0.854873</td>
      <td>0.008352</td>
      <td>0.016658</td>
      <td>0.344797</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>dog.9995.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.193059</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.268092</td>
      <td>0.177756</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>20.336287</td>
      <td>0.000000</td>
      <td>0.896977</td>
      <td>0.149458</td>
      <td>0.123787</td>
      <td>0.384676</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>dog.9996.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.003674</td>
      <td>0.000000</td>
      <td>2.768917</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.777494</td>
      <td>0.171866</td>
      <td>0.0</td>
      <td>0.001702</td>
      <td>17.666206</td>
      <td>0.000000</td>
      <td>0.779997</td>
      <td>0.034684</td>
      <td>0.133014</td>
      <td>1.652294</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>dog.9997.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.627975</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.797773</td>
      <td>0.114264</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>20.120745</td>
      <td>0.000000</td>
      <td>0.980927</td>
      <td>0.005060</td>
      <td>0.006118</td>
      <td>0.623514</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>dog.9998.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.044554</td>
      <td>0.000000</td>
      <td>1.563408</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.939854</td>
      <td>0.059493</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>18.924351</td>
      <td>0.000000</td>
      <td>1.038141</td>
      <td>0.069819</td>
      <td>0.069094</td>
      <td>0.904590</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>dog.9999.jpg</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000018</td>
      <td>1.050343</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.311641</td>
      <td>0.145927</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>19.418907</td>
      <td>0.000000</td>
      <td>1.413191</td>
      <td>0.014721</td>
      <td>0.111561</td>
      <td>0.944223</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 514 columns</p>
</div>



But, for the purposes of this demo, we can simply test the performance of a linear classifier over the featurized data. First, we'll build the training and test sets.


```python
# Creating a training set of 10,000 for each class
train_cats = featurizer.featurized_data[:10000, :]
train_dogs = featurizer.featurized_data[12500:22500, :]

# Creating a test set from the remaining 2,500 of each class
test_cats = featurizer.featurized_data[10000:12500, :]
test_dogs = featurizer.featurized_data[22500:, :]

# Combining the training data, and creating the class labels
train_combined = np.concatenate((train_cats, train_dogs))
labels_train = np.concatenate((np.zeros((10000,)), np.ones((10000,))))

# Combining the test data, and creating the class labels to check predictions
test_combined = np.concatenate((test_cats, test_dogs))
labels_test = np.concatenate((np.zeros((2500,)), np.ones((2500,))))
```

Then, we'll train the linear classifier:


```python
# Initialize the linear SVC
clf = svm.LinearSVC()

# Fit it on the training data
clf.fit(train_combined, labels_train)

# Check the performance of the linear classifier over the full Cats vs. Dogs dataset.
clf.score(test_combined, labels_test)
```




    0.94120000000000004



After featurizing the Cats vs. Dogs dataset, we find that a simple linear classifier trained over a SqueezeNet featurization achieves about 94% accuracy on distinguishing dogs vs. cats out of the box.

## Summary

That's it! We've looked at the following:

1. What data formats can be passed into the featurizer
2. How to initialize a simple featurizer
3. How to load data into the featurizer
4. How to featurize the loaded data

And as a bonus, we looked at how we might use the featurized data to perform predictions without dropping the CSV into the DataRobot app.

Unless you would like to examine the loaded data before featurizing it, steps 3 and 4 can actually be combined into a single step with the load_and_featurize_data( ) method.

## Next Steps

We have not covered using only a CSV with URL pointers, or a more complex dataset. That will be the subject of another Notebook.

To have more control over the options in the featurizer, or to understand its internal functioning more fully, check out the full package documentation.
