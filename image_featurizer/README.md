
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
featurizer = ImageFeaturizer(depth=1, automatic_downsample = True)
```


    Building the featurizer!

    Can't find weight file. Need to download weights from Keras!

    Model successfully initialized.
    Model decapitated!
    Automatic downsampling to 1024. If you would like to set custom downsampling, pass in an integer divisor of 2048 to num_pooled_features!
    Model downsampled!
    Full featurizer is built!
    Final layer feature space downsampled to 1024


This featurizer was 'decapitated' to the first layer below the prediction layer, which will produce complex representations. Because it is so close to the final prediction layer, it will create more specialized feature representations, and therefore will be better suited for image datasets that are similar to classes within the original ImageNet dataset. Cats and dogs are present within ImageNet, so a depth of 1 should perform well.

## Loading the Data

Now that the featurizer is built, we can load our data into the network. This will parse through the images in the order given by the csv, rescale them to a target size with a default of (299, 299), and build a 4D tensor containing the vectorized representations of the images. This tensor will later be fed into the network in order to be featurized.

The tensor will have the dimensions: [number of images, height, width, color channels]. In this case, the image tensor will have size [25000, 299, 299, 3].

We have to pass in the name of the column in the CSV that contains pointers to the images, as well as the path to the image directory and the path to the CSV itself, which are both saved from earlier.

If there are images in the directory that aren't in the CSV, or image names in the CSV that aren't in the directory, or even files that aren't valid image files in the directory, don't fear– the featurizer will only try to vectorize valid images that are in both the CSV and the directory. Any images present in the CSV but not the directory will be given zero vectors, and the order of the CSV is considered the canonical order for the images.


```python
featurizer.load_data('images', image_path = image_path, csv_path = csv_path)
```

    Found image paths that overlap between both the directory and the csv!
    Converting images!
    Converted 0 images! Only 25000 images left to go!
    Converted 1000 images! Only 24000 images left to go!
    Converted 2000 images! Only 23000 images left to go!
    Converted 3000 images! Only 22000 images left to go!
    Converted 4000 images! Only 21000 images left to go!
    Converted 5000 images! Only 20000 images left to go!
    Converted 6000 images! Only 19000 images left to go!
    Converted 7000 images! Only 18000 images left to go!
    Converted 8000 images! Only 17000 images left to go!
    Converted 9000 images! Only 16000 images left to go!
    Converted 10000 images! Only 15000 images left to go!
    Converted 11000 images! Only 14000 images left to go!
    Converted 12000 images! Only 13000 images left to go!
    Converted 13000 images! Only 12000 images left to go!
    Converted 14000 images! Only 11000 images left to go!
    Converted 15000 images! Only 10000 images left to go!
    Converted 16000 images! Only 9000 images left to go!
    Converted 17000 images! Only 8000 images left to go!
    Converted 18000 images! Only 7000 images left to go!
    Converted 19000 images! Only 6000 images left to go!
    Converted 20000 images! Only 5000 images left to go!
    Converted 21000 images! Only 4000 images left to go!
    Converted 22000 images! Only 3000 images left to go!
    Converted 23000 images! Only 2000 images left to go!
    Converted 24000 images! Only 1000 images left to go!


The image data has now been loaded into the featurizer and vectorized, and is ready for featurization. We can check the size, format, and other stored information about the data by calling the featurizer object attributes:


```python
print('Vectorized data shape: {}'.format(featurizer.data.shape))

print('CSV path: \'{}\''.format(featurizer.csv_path))

print('Image directory path: \'{}\''.format(featurizer.image_path))
```

    Vectorized data shape: (25000, 299, 299, 3)
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
     'scaled_size',
     'depth',
     'automatic_downsample',
     'featurized_data',
     'model',
     'isotropic_scaling',
     'data',
     'crop_size']



## Featurizing the Data

Now that the data is loaded, we're ready to featurize the data. This will push the vectorized images through the network and save the 2D matrix output– each row representing a single image, and each column storing a different feature.

It will then create and save a new CSV by appending these features to the end of the given CSV in line with each image's row. The features themselves will also be saved in a separate CSV file without the image names or other data. Both generated CSVs will be saved to the same path as the original CSV, with the features-only CSV appending '_features_only' and the combined CSV appending '_full' to the end of their respective filenames.

The featurize( ) method requires no parameters, as it uses the data we just loaded into the network. This requires pushing images through the deep InceptionV3 network, and so relatively large datasets will require a GPU to perform in a reasonable amount of time. Using a mid-range GPU, it can take about 30 minutes to process the full 25,000 photos in the Dogs vs. Cats.


```python
featurizer.featurize()
```

    Checking array initialized.
    Trying to featurize data!
    Creating feature array!
    25000/25000 [==============================] - 1848s  
    Feature array created successfully.
    Adding image features to csv!





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
      <th>image_feature_1014</th>
      <th>image_feature_1015</th>
      <th>image_feature_1016</th>
      <th>image_feature_1017</th>
      <th>image_feature_1018</th>
      <th>image_feature_1019</th>
      <th>image_feature_1020</th>
      <th>image_feature_1021</th>
      <th>image_feature_1022</th>
      <th>image_feature_1023</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat.0.jpg</td>
      <td>0</td>
      <td>0.197623</td>
      <td>0.153129</td>
      <td>0.237384</td>
      <td>0.325404</td>
      <td>0.174551</td>
      <td>0.093449</td>
      <td>0.397525</td>
      <td>0.536594</td>
      <td>...</td>
      <td>0.346164</td>
      <td>0.676448</td>
      <td>0.316065</td>
      <td>0.374114</td>
      <td>0.615561</td>
      <td>0.902570</td>
      <td>0.520825</td>
      <td>0.560737</td>
      <td>1.129250</td>
      <td>0.372229</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat.1.jpg</td>
      <td>0</td>
      <td>0.077591</td>
      <td>0.144663</td>
      <td>0.107953</td>
      <td>0.211797</td>
      <td>0.072236</td>
      <td>0.121741</td>
      <td>0.129945</td>
      <td>0.131091</td>
      <td>...</td>
      <td>0.245134</td>
      <td>1.225246</td>
      <td>0.399110</td>
      <td>0.129122</td>
      <td>0.189429</td>
      <td>0.235852</td>
      <td>0.220862</td>
      <td>0.278146</td>
      <td>0.134976</td>
      <td>0.535692</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat.10.jpg</td>
      <td>0</td>
      <td>0.247032</td>
      <td>0.181568</td>
      <td>0.097621</td>
      <td>0.154228</td>
      <td>0.247492</td>
      <td>0.178228</td>
      <td>0.175542</td>
      <td>0.306296</td>
      <td>...</td>
      <td>0.263852</td>
      <td>0.940070</td>
      <td>0.358609</td>
      <td>0.506933</td>
      <td>0.753087</td>
      <td>0.233318</td>
      <td>0.655843</td>
      <td>0.799059</td>
      <td>0.681655</td>
      <td>0.579095</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat.100.jpg</td>
      <td>0</td>
      <td>0.259990</td>
      <td>0.129949</td>
      <td>0.165196</td>
      <td>0.130249</td>
      <td>0.156296</td>
      <td>0.108147</td>
      <td>0.477028</td>
      <td>0.563983</td>
      <td>...</td>
      <td>0.541575</td>
      <td>0.772276</td>
      <td>0.498641</td>
      <td>0.867951</td>
      <td>0.543547</td>
      <td>0.722114</td>
      <td>0.631598</td>
      <td>0.749955</td>
      <td>0.552881</td>
      <td>0.813209</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat.1000.jpg</td>
      <td>0</td>
      <td>0.446480</td>
      <td>0.201302</td>
      <td>0.126474</td>
      <td>0.079862</td>
      <td>0.215835</td>
      <td>0.399131</td>
      <td>0.636886</td>
      <td>0.109699</td>
      <td>...</td>
      <td>0.308573</td>
      <td>0.947029</td>
      <td>0.174811</td>
      <td>0.453257</td>
      <td>0.792167</td>
      <td>0.285224</td>
      <td>0.642199</td>
      <td>0.629148</td>
      <td>0.613605</td>
      <td>0.903375</td>
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
      <td>0.119224</td>
      <td>0.098877</td>
      <td>0.158328</td>
      <td>0.201316</td>
      <td>0.271268</td>
      <td>0.393014</td>
      <td>0.268696</td>
      <td>0.374308</td>
      <td>...</td>
      <td>0.272386</td>
      <td>0.190257</td>
      <td>0.244063</td>
      <td>0.040812</td>
      <td>0.220867</td>
      <td>0.548893</td>
      <td>0.155369</td>
      <td>0.188319</td>
      <td>0.014220</td>
      <td>0.469891</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>dog.9996.jpg</td>
      <td>1</td>
      <td>0.218130</td>
      <td>0.122711</td>
      <td>0.497641</td>
      <td>0.232362</td>
      <td>0.282693</td>
      <td>0.115563</td>
      <td>0.222415</td>
      <td>0.178117</td>
      <td>...</td>
      <td>0.374624</td>
      <td>0.298325</td>
      <td>0.131216</td>
      <td>0.221744</td>
      <td>0.285206</td>
      <td>0.168957</td>
      <td>0.454489</td>
      <td>0.197727</td>
      <td>0.098618</td>
      <td>0.380426</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>dog.9997.jpg</td>
      <td>1</td>
      <td>0.214751</td>
      <td>0.287097</td>
      <td>0.240857</td>
      <td>0.236063</td>
      <td>0.811545</td>
      <td>0.395226</td>
      <td>0.139665</td>
      <td>0.351027</td>
      <td>...</td>
      <td>0.271042</td>
      <td>0.196889</td>
      <td>0.410896</td>
      <td>0.236834</td>
      <td>0.291153</td>
      <td>0.828539</td>
      <td>0.511775</td>
      <td>0.504615</td>
      <td>0.496241</td>
      <td>0.379007</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>dog.9998.jpg</td>
      <td>1</td>
      <td>0.296174</td>
      <td>0.281312</td>
      <td>0.426309</td>
      <td>0.146541</td>
      <td>0.766958</td>
      <td>0.184113</td>
      <td>0.174594</td>
      <td>0.280894</td>
      <td>...</td>
      <td>0.160002</td>
      <td>0.192043</td>
      <td>0.224755</td>
      <td>0.141034</td>
      <td>0.426096</td>
      <td>0.439477</td>
      <td>0.307610</td>
      <td>0.533777</td>
      <td>0.197895</td>
      <td>0.528016</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>dog.9999.jpg</td>
      <td>1</td>
      <td>0.495791</td>
      <td>0.227660</td>
      <td>0.107934</td>
      <td>0.122440</td>
      <td>0.399653</td>
      <td>0.264278</td>
      <td>0.316264</td>
      <td>0.107855</td>
      <td>...</td>
      <td>0.531086</td>
      <td>0.360293</td>
      <td>0.487009</td>
      <td>0.469563</td>
      <td>0.326417</td>
      <td>0.285195</td>
      <td>0.100361</td>
      <td>0.130836</td>
      <td>0.593886</td>
      <td>0.511735</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 1026 columns</p>
</div>



## Results

The dataset has now been fully featurized! The features are saved under the featurized_data attribute:


```python
featurizer.featurized_data
```




    array([[ 0.19762258,  0.15312853,  0.23738424, ...,  0.56073749,
             1.12924969,  0.3722288 ],
           [ 0.07759078,  0.14466347,  0.10795289, ...,  0.27814645,
             0.13497572,  0.53569233],
           [ 0.24703167,  0.18156832,  0.09762136, ...,  0.79905927,
             0.68165481,  0.57909489],
           ...,
           [ 0.21475141,  0.28709683,  0.24085702, ...,  0.50461513,
             0.49624115,  0.37900704],
           [ 0.29617372,  0.28131226,  0.42630851, ...,  0.53377748,
             0.19789484,  0.52801621],
           [ 0.49579096,  0.22766027,  0.10793425, ...,  0.13083567,
             0.59388638,  0.51173502]], dtype=float32)



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
      <th>image_feature_1014</th>
      <th>image_feature_1015</th>
      <th>image_feature_1016</th>
      <th>image_feature_1017</th>
      <th>image_feature_1018</th>
      <th>image_feature_1019</th>
      <th>image_feature_1020</th>
      <th>image_feature_1021</th>
      <th>image_feature_1022</th>
      <th>image_feature_1023</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat.0.jpg</td>
      <td>0</td>
      <td>0.197623</td>
      <td>0.153129</td>
      <td>0.237384</td>
      <td>0.325404</td>
      <td>0.174551</td>
      <td>0.093449</td>
      <td>0.397525</td>
      <td>0.536594</td>
      <td>...</td>
      <td>0.346164</td>
      <td>0.676448</td>
      <td>0.316065</td>
      <td>0.374114</td>
      <td>0.615561</td>
      <td>0.902570</td>
      <td>0.520825</td>
      <td>0.560737</td>
      <td>1.129250</td>
      <td>0.372229</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat.1.jpg</td>
      <td>0</td>
      <td>0.077591</td>
      <td>0.144663</td>
      <td>0.107953</td>
      <td>0.211797</td>
      <td>0.072236</td>
      <td>0.121741</td>
      <td>0.129945</td>
      <td>0.131091</td>
      <td>...</td>
      <td>0.245134</td>
      <td>1.225246</td>
      <td>0.399110</td>
      <td>0.129122</td>
      <td>0.189429</td>
      <td>0.235852</td>
      <td>0.220862</td>
      <td>0.278146</td>
      <td>0.134976</td>
      <td>0.535692</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat.10.jpg</td>
      <td>0</td>
      <td>0.247032</td>
      <td>0.181568</td>
      <td>0.097621</td>
      <td>0.154228</td>
      <td>0.247492</td>
      <td>0.178228</td>
      <td>0.175542</td>
      <td>0.306296</td>
      <td>...</td>
      <td>0.263852</td>
      <td>0.940070</td>
      <td>0.358609</td>
      <td>0.506933</td>
      <td>0.753087</td>
      <td>0.233318</td>
      <td>0.655843</td>
      <td>0.799059</td>
      <td>0.681655</td>
      <td>0.579095</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat.100.jpg</td>
      <td>0</td>
      <td>0.259990</td>
      <td>0.129949</td>
      <td>0.165196</td>
      <td>0.130249</td>
      <td>0.156296</td>
      <td>0.108147</td>
      <td>0.477028</td>
      <td>0.563983</td>
      <td>...</td>
      <td>0.541575</td>
      <td>0.772276</td>
      <td>0.498641</td>
      <td>0.867951</td>
      <td>0.543547</td>
      <td>0.722114</td>
      <td>0.631598</td>
      <td>0.749955</td>
      <td>0.552881</td>
      <td>0.813209</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat.1000.jpg</td>
      <td>0</td>
      <td>0.446480</td>
      <td>0.201302</td>
      <td>0.126474</td>
      <td>0.079862</td>
      <td>0.215835</td>
      <td>0.399131</td>
      <td>0.636886</td>
      <td>0.109699</td>
      <td>...</td>
      <td>0.308573</td>
      <td>0.947029</td>
      <td>0.174811</td>
      <td>0.453257</td>
      <td>0.792167</td>
      <td>0.285224</td>
      <td>0.642199</td>
      <td>0.629148</td>
      <td>0.613605</td>
      <td>0.903375</td>
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
      <td>0.119224</td>
      <td>0.098877</td>
      <td>0.158328</td>
      <td>0.201316</td>
      <td>0.271268</td>
      <td>0.393014</td>
      <td>0.268696</td>
      <td>0.374308</td>
      <td>...</td>
      <td>0.272386</td>
      <td>0.190257</td>
      <td>0.244063</td>
      <td>0.040812</td>
      <td>0.220867</td>
      <td>0.548893</td>
      <td>0.155369</td>
      <td>0.188319</td>
      <td>0.014220</td>
      <td>0.469891</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>dog.9996.jpg</td>
      <td>1</td>
      <td>0.218130</td>
      <td>0.122711</td>
      <td>0.497641</td>
      <td>0.232362</td>
      <td>0.282693</td>
      <td>0.115563</td>
      <td>0.222415</td>
      <td>0.178117</td>
      <td>...</td>
      <td>0.374624</td>
      <td>0.298325</td>
      <td>0.131216</td>
      <td>0.221744</td>
      <td>0.285206</td>
      <td>0.168957</td>
      <td>0.454489</td>
      <td>0.197727</td>
      <td>0.098618</td>
      <td>0.380426</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>dog.9997.jpg</td>
      <td>1</td>
      <td>0.214751</td>
      <td>0.287097</td>
      <td>0.240857</td>
      <td>0.236063</td>
      <td>0.811545</td>
      <td>0.395226</td>
      <td>0.139665</td>
      <td>0.351027</td>
      <td>...</td>
      <td>0.271042</td>
      <td>0.196889</td>
      <td>0.410896</td>
      <td>0.236834</td>
      <td>0.291153</td>
      <td>0.828539</td>
      <td>0.511775</td>
      <td>0.504615</td>
      <td>0.496241</td>
      <td>0.379007</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>dog.9998.jpg</td>
      <td>1</td>
      <td>0.296174</td>
      <td>0.281312</td>
      <td>0.426309</td>
      <td>0.146541</td>
      <td>0.766958</td>
      <td>0.184113</td>
      <td>0.174594</td>
      <td>0.280894</td>
      <td>...</td>
      <td>0.160002</td>
      <td>0.192043</td>
      <td>0.224755</td>
      <td>0.141034</td>
      <td>0.426096</td>
      <td>0.439477</td>
      <td>0.307610</td>
      <td>0.533777</td>
      <td>0.197895</td>
      <td>0.528016</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>dog.9999.jpg</td>
      <td>1</td>
      <td>0.495791</td>
      <td>0.227660</td>
      <td>0.107934</td>
      <td>0.122440</td>
      <td>0.399653</td>
      <td>0.264278</td>
      <td>0.316264</td>
      <td>0.107855</td>
      <td>...</td>
      <td>0.531086</td>
      <td>0.360293</td>
      <td>0.487009</td>
      <td>0.469563</td>
      <td>0.326417</td>
      <td>0.285195</td>
      <td>0.100361</td>
      <td>0.130836</td>
      <td>0.593886</td>
      <td>0.511735</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 1026 columns</p>
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




    0.99139999999999995



After featurizing the Cats vs. Dogs dataset, we find that a simple linear classifier trained over the data achieves over 99% accuracy on distinguishing dogs vs. cats.

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
