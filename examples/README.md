
# Example: Cats vs. Dogs With SqueezeNet

This notebook demonstrates the usage of ``image_featurizer`` using the Kaggle Cats vs. Dogs dataset.

We will look at the usage of the ``ImageFeaturizer()`` class, which provides a convenient pipeline to quickly tackle image problems with DataRobot's platform. 

It allows users to load image data into the featurizer, and then featurizes the images into a maximum of 2048 features. It appends these features to the CSV as extra columns in line with the image rows. If no CSV was passed in with an image directory, the featurizer generates a new CSV automatically and performs the same function.



```python
# Setting up stdout logging for the example case
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
```


```python
# Importing the dependencies for this example
import pandas as pd
import numpy as np
from sklearn import svm
from pic2vec import ImageFeaturizer

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

image_path = 'cats_v_dogs_train/'
csv_path = 'cats_v_dogs_train.csv'

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
      <th>label</th>
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
      <td>cat.2.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat.3.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat.4.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>dog.12495.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>dog.12496.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>dog.12497.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>dog.12498.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>dog.12499.jpg</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 2 columns</p>
</div>



The image directory contains 12,500 images of cats and 12,500 images of dogs. The CSV contains pointers to each image in the directory, along with a class label (0 for cats, 1 for dogs).

## Initializing the Featurizer

We will now initialize the ImageFeaturizer( ) class with a few parameters that define the model. If in doubt, we can always call the featurizer with no parameters, and it will initialize itself to a cookie-cutter build. Here, we will call the parameters explicitly to demonstrate functionality. However, these are generally the default weights, so for this build we could just call ```featurizer = ImageFeaturizer()```.

Because we have not specified a model, the featurizer will default to the built-in SqueezeNet model, with loaded weights prepackaged. If you initialize another model, pic2vec will automatically download the model weights through the Keras backend.

The depth indicates how far down we should cut the model to draw abstract features– the further down we cut, the less complex the representations will be, but they may also be less specialized to the specific classes in the ImageNet dataset that the model was trained on– and so they may perform better on data that is further from the classes within the dataset.

Automatic downsampling means that this model will downsample the final layer from 512 features to 256 features, which is a more compact representation. With large datasets and bigger models (such as InceptionV3, more features may run into memory problems or difficulty optimizing, so it may be worth downsampling to a smaller featurspace.


```python
featurizer = ImageFeaturizer(depth=1, autosample = False, model='squeezenet')
```

    INFO - Building the featurizer.
    INFO - Loading/downloading SqueezeNet model weights. This may take a minute first time.
    INFO - Model successfully initialized.
    INFO - Model decapitated.
    INFO - Model downsampled.
    INFO - Full featurizer is built.
    INFO - No downsampling. Final layer feature space has size 512


This featurizer was 'decapitated' to the first layer below the prediction layer, which will produce complex representations. Because it is so close to the final prediction layer, it will create more specialized feature representations, and therefore will be better suited for image datasets that are similar to classes within the original ImageNet dataset. Cats and dogs are present within ImageNet, so a depth of 1 should perform well. 

## Loading the Data

Now that the featurizer is built, we can load our data into the network. This will parse through the images in the order given by the csv, rescale them to a target size depending on the network– SqueezeNet is (227, 227)– and build a 4D tensor containing the vectorized representations of the images. This tensor will later be fed into the network in order to be featurized.

The tensor will have the dimensions: [number of image columns, number of images, height, width, color channels]. In this case, the image tensor will have size [1, 25000, 227, 227, 3].

We have to pass in the name of the column in the CSV that contains pointers to the images, as well as the path to the image directory and the path to the CSV itself, which are both saved from earlier. 

If there are images in the directory that aren't in the CSV, or image names in the CSV that aren't in the directory, or even files that aren't valid image files in the directory, don't fear– the featurizer will only try to vectorize valid images that are present in both the CSV and the directory. Any images present in the CSV but not the directory will be given zero vectors, and the order of the CSV is considered the canonical order for the images.


```python
featurizer.load_data('images', image_path = image_path, csv_path = csv_path)
```

    INFO - Found image paths that overlap between both the directory and the csv.
    INFO - Converting images.
    INFO - Converted 0 images. Only 25000 images left to go.
    INFO - Converted 1000 images. Only 24000 images left to go.
    INFO - Converted 2000 images. Only 23000 images left to go.
    INFO - Converted 3000 images. Only 22000 images left to go.
    INFO - Converted 4000 images. Only 21000 images left to go.
    INFO - Converted 5000 images. Only 20000 images left to go.
    INFO - Converted 6000 images. Only 19000 images left to go.
    INFO - Converted 7000 images. Only 18000 images left to go.
    INFO - Converted 8000 images. Only 17000 images left to go.
    INFO - Converted 9000 images. Only 16000 images left to go.
    INFO - Converted 10000 images. Only 15000 images left to go.
    INFO - Converted 11000 images. Only 14000 images left to go.
    INFO - Converted 12000 images. Only 13000 images left to go.
    INFO - Converted 13000 images. Only 12000 images left to go.
    INFO - Converted 14000 images. Only 11000 images left to go.
    INFO - Converted 15000 images. Only 10000 images left to go.
    INFO - Converted 16000 images. Only 9000 images left to go.
    INFO - Converted 17000 images. Only 8000 images left to go.
    INFO - Converted 18000 images. Only 7000 images left to go.
    INFO - Converted 19000 images. Only 6000 images left to go.
    INFO - Converted 20000 images. Only 5000 images left to go.
    INFO - Converted 21000 images. Only 4000 images left to go.
    INFO - Converted 22000 images. Only 3000 images left to go.
    INFO - Converted 23000 images. Only 2000 images left to go.
    INFO - Converted 24000 images. Only 1000 images left to go.



```python
print('Vectorized data shape: {}'.format(featurizer.data.shape))

print('CSV path: \'{}\''.format(featurizer.csv_path))

print('Image directory path: \'{}\''.format(featurizer.image_path))
```

    Vectorized data shape: (1, 25000, 227, 227, 3)
    CSV path: 'cats_v_dogs_train.csv'
    Image directory path: 'cats_v_dogs_train/'


For a full list of attributes, call:


```python
featurizer.__dict__.keys()
```




    ['downsample_size',
     'visualize',
     'autosample',
     'isotropic_scaling',
     'num_features',
     'number_crops',
     'image_list',
     'featurizer',
     'scaled_size',
     'depth',
     'csv_path',
     'crop_size',
     'featurized_data',
     'image_column_headers',
     'data',
     'model_name',
     'image_path']



## Featurizing the Data

Now that the data is loaded, we're ready to featurize the data. This will push the vectorized images through the network and save the 2D matrix output– each row representing a single image, and each column storing a different feature.

It will then create and save a new CSV by appending these features to the end of the given CSV in line with each image's row. The features themselves will also be saved in a separate CSV file without the image names or other data. Both generated CSVs will be saved to the same path as the original CSV, with the features-only CSV appending '_features_only' and the combined CSV appending '_full' to the end of their respective filenames.

The featurize( ) method requires no parameters, as it uses the data we just loaded into the network. This requires pushing images through the deep network, and so if you choose to use a slower, more powerful model like InceptionV3, relatively large datasets will require a GPU to perform in a reasonable amount of time. Using a mid-range GPU, it can take about 30 minutes to process the full 25,000 photos in the Dogs vs. Cats through InceptionV3. On the other hand, if you would like a fast model, lightweight model without top-of-the-line accuracy, SqueezeNet works well enough and can perform inference on CPUs quickly.


```python
featurizer.featurize()
```

    INFO - Trying to featurize data.
    INFO - Creating feature array.
    25000/25000 [==============================] - 825s   
    INFO - Feature array created successfully.
    INFO - Adding image features to csv.
    INFO - Number of missing photos: 25000





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
      <th>label</th>
      <th>images_missing</th>
      <th>images_feat_0</th>
      <th>images_feat_1</th>
      <th>images_feat_2</th>
      <th>images_feat_3</th>
      <th>images_feat_4</th>
      <th>images_feat_5</th>
      <th>images_feat_6</th>
      <th>...</th>
      <th>images_feat_502</th>
      <th>images_feat_503</th>
      <th>images_feat_504</th>
      <th>images_feat_505</th>
      <th>images_feat_506</th>
      <th>images_feat_507</th>
      <th>images_feat_508</th>
      <th>images_feat_509</th>
      <th>images_feat_510</th>
      <th>images_feat_511</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat.0.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>0.422239</td>
      <td>4.080936</td>
      <td>14.894337</td>
      <td>2.842498</td>
      <td>0.967452</td>
      <td>10.851055</td>
      <td>0.164090</td>
      <td>...</td>
      <td>5.716428</td>
      <td>0.227580</td>
      <td>1.512349</td>
      <td>1.838279</td>
      <td>6.923377</td>
      <td>2.754216</td>
      <td>1.599615</td>
      <td>0.942032</td>
      <td>8.596214</td>
      <td>0.195745</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat.1.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>2.235883</td>
      <td>1.766027</td>
      <td>0.489503</td>
      <td>1.077848</td>
      <td>3.744066</td>
      <td>3.900755</td>
      <td>0.678774</td>
      <td>...</td>
      <td>0.466049</td>
      <td>0.456763</td>
      <td>0.000000</td>
      <td>8.796008</td>
      <td>8.920897</td>
      <td>2.318893</td>
      <td>3.206552</td>
      <td>5.324099</td>
      <td>25.885130</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat.2.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>0.804545</td>
      <td>0.685238</td>
      <td>0.411905</td>
      <td>3.651519</td>
      <td>7.440580</td>
      <td>1.365789</td>
      <td>1.759454</td>
      <td>...</td>
      <td>0.991104</td>
      <td>0.015178</td>
      <td>0.018916</td>
      <td>7.745066</td>
      <td>0.000000</td>
      <td>0.187744</td>
      <td>0.248889</td>
      <td>7.293088</td>
      <td>8.606462</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat.3.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>0.481214</td>
      <td>0.229483</td>
      <td>5.039218</td>
      <td>0.669226</td>
      <td>3.988109</td>
      <td>2.878755</td>
      <td>0.642729</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.469958</td>
      <td>0.532943</td>
      <td>3.121966</td>
      <td>0.095707</td>
      <td>3.489891</td>
      <td>0.262518</td>
      <td>1.729952</td>
      <td>5.988695</td>
      <td>0.080222</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat.4.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>0.000000</td>
      <td>3.258759</td>
      <td>9.666997</td>
      <td>6.237058</td>
      <td>1.160069</td>
      <td>0.055264</td>
      <td>0.394765</td>
      <td>...</td>
      <td>0.670079</td>
      <td>0.755777</td>
      <td>0.076195</td>
      <td>7.925221</td>
      <td>0.149376</td>
      <td>5.640311</td>
      <td>0.217993</td>
      <td>1.215899</td>
      <td>12.723279</td>
      <td>0.856007</td>
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
      <td>dog.12495.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>0.319323</td>
      <td>10.394070</td>
      <td>2.253926</td>
      <td>17.211163</td>
      <td>10.175515</td>
      <td>0.754048</td>
      <td>0.000000</td>
      <td>...</td>
      <td>6.547691</td>
      <td>0.542316</td>
      <td>0.000000</td>
      <td>2.855716</td>
      <td>0.984909</td>
      <td>0.789532</td>
      <td>1.463116</td>
      <td>7.819314</td>
      <td>0.194761</td>
      <td>2.515571</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>dog.12496.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>4.812177</td>
      <td>2.173984</td>
      <td>3.127878</td>
      <td>7.731273</td>
      <td>2.563596</td>
      <td>0.855450</td>
      <td>1.506930</td>
      <td>...</td>
      <td>0.805109</td>
      <td>0.897770</td>
      <td>0.067206</td>
      <td>1.332061</td>
      <td>1.023679</td>
      <td>2.697655</td>
      <td>2.661853</td>
      <td>0.294248</td>
      <td>11.114500</td>
      <td>0.605132</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>dog.12497.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>0.603782</td>
      <td>10.804434</td>
      <td>5.988084</td>
      <td>11.258055</td>
      <td>1.711970</td>
      <td>0.172748</td>
      <td>0.839580</td>
      <td>...</td>
      <td>9.690125</td>
      <td>0.000000</td>
      <td>0.681894</td>
      <td>4.975673</td>
      <td>0.593460</td>
      <td>12.891261</td>
      <td>3.147193</td>
      <td>2.841281</td>
      <td>2.273726</td>
      <td>0.726203</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>dog.12498.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>0.012421</td>
      <td>0.779886</td>
      <td>3.646794</td>
      <td>1.577259</td>
      <td>0.314669</td>
      <td>1.035333</td>
      <td>0.140289</td>
      <td>...</td>
      <td>6.654152</td>
      <td>3.092728</td>
      <td>1.509475</td>
      <td>2.738165</td>
      <td>0.000000</td>
      <td>3.335813</td>
      <td>2.847281</td>
      <td>1.110609</td>
      <td>3.183074</td>
      <td>1.643685</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>dog.12499.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>2.025731</td>
      <td>2.882919</td>
      <td>7.076093</td>
      <td>3.225270</td>
      <td>0.053434</td>
      <td>0.481757</td>
      <td>0.222437</td>
      <td>...</td>
      <td>5.783323</td>
      <td>0.401531</td>
      <td>0.397468</td>
      <td>3.518530</td>
      <td>0.342225</td>
      <td>0.887652</td>
      <td>2.742420</td>
      <td>7.574071</td>
      <td>13.408817</td>
      <td>0.622023</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 515 columns</p>
</div>



## Results

The dataset has now been fully featurized! The features are saved under the featurized_data attribute:


```python
featurizer.featurized_data
```




    array([[  4.22239482e-01,   4.08093643e+00,   1.48943367e+01, ...,
              9.42031682e-01,   8.59621429e+00,   1.95744559e-01],
           [  2.23588324e+00,   1.76602709e+00,   4.89503175e-01, ...,
              5.32409906e+00,   2.58851299e+01,   0.00000000e+00],
           [  8.04544866e-01,   6.85238183e-01,   4.11904931e-01, ...,
              7.29308796e+00,   8.60646152e+00,   0.00000000e+00],
           ..., 
           [  6.03782475e-01,   1.08044338e+01,   5.98808432e+00, ...,
              2.84128141e+00,   2.27372599e+00,   7.26202726e-01],
           [  1.24208946e-02,   7.79886365e-01,   3.64679360e+00, ...,
              1.11060941e+00,   3.18307400e+00,   1.64368451e+00],
           [  2.02573061e+00,   2.88291931e+00,   7.07609320e+00, ...,
              7.57407141e+00,   1.34088173e+01,   6.22022569e-01]])



The full data has also been successfully saved in CSV form, which allows it to be dropped directly into the DataRobot app:


```python
pd.read_csv('cats_v_dogs_train_squeezenet_depth-1_output-512_(14-Aug-2017-13.30.46)_full.csv')
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
      <th>label</th>
      <th>images_missing</th>
      <th>images_feat_0</th>
      <th>images_feat_1</th>
      <th>images_feat_2</th>
      <th>images_feat_3</th>
      <th>images_feat_4</th>
      <th>images_feat_5</th>
      <th>images_feat_6</th>
      <th>...</th>
      <th>images_feat_502</th>
      <th>images_feat_503</th>
      <th>images_feat_504</th>
      <th>images_feat_505</th>
      <th>images_feat_506</th>
      <th>images_feat_507</th>
      <th>images_feat_508</th>
      <th>images_feat_509</th>
      <th>images_feat_510</th>
      <th>images_feat_511</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat.0.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>0.422239</td>
      <td>4.080936</td>
      <td>14.894337</td>
      <td>2.842498</td>
      <td>0.967452</td>
      <td>10.851055</td>
      <td>0.164090</td>
      <td>...</td>
      <td>5.716428</td>
      <td>0.227580</td>
      <td>1.512349</td>
      <td>1.838279</td>
      <td>6.923377</td>
      <td>2.754216</td>
      <td>1.599615</td>
      <td>0.942032</td>
      <td>8.596214</td>
      <td>0.195745</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat.1.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>2.235883</td>
      <td>1.766027</td>
      <td>0.489503</td>
      <td>1.077848</td>
      <td>3.744066</td>
      <td>3.900755</td>
      <td>0.678774</td>
      <td>...</td>
      <td>0.466049</td>
      <td>0.456763</td>
      <td>0.000000</td>
      <td>8.796008</td>
      <td>8.920897</td>
      <td>2.318893</td>
      <td>3.206552</td>
      <td>5.324099</td>
      <td>25.885130</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat.2.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>0.804545</td>
      <td>0.685238</td>
      <td>0.411905</td>
      <td>3.651519</td>
      <td>7.440580</td>
      <td>1.365789</td>
      <td>1.759454</td>
      <td>...</td>
      <td>0.991104</td>
      <td>0.015178</td>
      <td>0.018916</td>
      <td>7.745066</td>
      <td>0.000000</td>
      <td>0.187744</td>
      <td>0.248889</td>
      <td>7.293088</td>
      <td>8.606462</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat.3.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>0.481214</td>
      <td>0.229483</td>
      <td>5.039218</td>
      <td>0.669226</td>
      <td>3.988109</td>
      <td>2.878755</td>
      <td>0.642729</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.469958</td>
      <td>0.532943</td>
      <td>3.121966</td>
      <td>0.095707</td>
      <td>3.489891</td>
      <td>0.262518</td>
      <td>1.729952</td>
      <td>5.988695</td>
      <td>0.080222</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat.4.jpg</td>
      <td>0</td>
      <td>False</td>
      <td>0.000000</td>
      <td>3.258759</td>
      <td>9.666997</td>
      <td>6.237058</td>
      <td>1.160069</td>
      <td>0.055264</td>
      <td>0.394765</td>
      <td>...</td>
      <td>0.670079</td>
      <td>0.755777</td>
      <td>0.076195</td>
      <td>7.925221</td>
      <td>0.149376</td>
      <td>5.640311</td>
      <td>0.217993</td>
      <td>1.215899</td>
      <td>12.723279</td>
      <td>0.856007</td>
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
      <td>dog.12495.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>0.319323</td>
      <td>10.394070</td>
      <td>2.253926</td>
      <td>17.211163</td>
      <td>10.175515</td>
      <td>0.754048</td>
      <td>0.000000</td>
      <td>...</td>
      <td>6.547691</td>
      <td>0.542316</td>
      <td>0.000000</td>
      <td>2.855716</td>
      <td>0.984909</td>
      <td>0.789532</td>
      <td>1.463116</td>
      <td>7.819314</td>
      <td>0.194761</td>
      <td>2.515571</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>dog.12496.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>4.812177</td>
      <td>2.173984</td>
      <td>3.127878</td>
      <td>7.731273</td>
      <td>2.563596</td>
      <td>0.855450</td>
      <td>1.506930</td>
      <td>...</td>
      <td>0.805109</td>
      <td>0.897770</td>
      <td>0.067206</td>
      <td>1.332061</td>
      <td>1.023679</td>
      <td>2.697655</td>
      <td>2.661853</td>
      <td>0.294248</td>
      <td>11.114500</td>
      <td>0.605132</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>dog.12497.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>0.603782</td>
      <td>10.804434</td>
      <td>5.988084</td>
      <td>11.258055</td>
      <td>1.711970</td>
      <td>0.172748</td>
      <td>0.839580</td>
      <td>...</td>
      <td>9.690125</td>
      <td>0.000000</td>
      <td>0.681894</td>
      <td>4.975673</td>
      <td>0.593460</td>
      <td>12.891261</td>
      <td>3.147193</td>
      <td>2.841281</td>
      <td>2.273726</td>
      <td>0.726203</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>dog.12498.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>0.012421</td>
      <td>0.779886</td>
      <td>3.646794</td>
      <td>1.577259</td>
      <td>0.314669</td>
      <td>1.035333</td>
      <td>0.140289</td>
      <td>...</td>
      <td>6.654152</td>
      <td>3.092728</td>
      <td>1.509475</td>
      <td>2.738165</td>
      <td>0.000000</td>
      <td>3.335813</td>
      <td>2.847281</td>
      <td>1.110609</td>
      <td>3.183074</td>
      <td>1.643685</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>dog.12499.jpg</td>
      <td>1</td>
      <td>False</td>
      <td>2.025731</td>
      <td>2.882919</td>
      <td>7.076093</td>
      <td>3.225270</td>
      <td>0.053434</td>
      <td>0.481757</td>
      <td>0.222437</td>
      <td>...</td>
      <td>5.783323</td>
      <td>0.401531</td>
      <td>0.397468</td>
      <td>3.518530</td>
      <td>0.342225</td>
      <td>0.887652</td>
      <td>2.742420</td>
      <td>7.574071</td>
      <td>13.408817</td>
      <td>0.622023</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 515 columns</p>
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

# Check the performance of the linear classifier over the full Cats vs. Dogs dataset!
clf.score(test_combined, labels_test)
```




    0.96140000000000003



After running the Cats vs. Dogs dataset through the lightest-weight pic2vec model, we find that a simple linear classifier trained over the featurized data achieves over 96% accuracy on distinguishing dogs vs. cats out of the box.

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
