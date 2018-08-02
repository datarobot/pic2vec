
# Example: Cats vs. Dogs With SqueezeNet

This notebook demonstrates the usage of ``image_featurizer`` using the Kaggle Cats vs. Dogs dataset.

We will look at the usage of the ``ImageFeaturizer()`` class, which provides a convenient pipeline to quickly tackle image problems with DataRobot's platform.

It allows users to load image data into the featurizer, and then featurizes the images into a maximum of 2048 features. It appends these features to the CSV as extra columns in line with the image rows. If no CSV was passed in with an image directory, the featurizer generates a new CSV automatically and performs the same function.



```python
# Setting up stdout logging
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
root.addHandler(ch)

# Setting pandas display options
pd.options.display.max_rows = 10

```


```python
# Importing the dependencies for this example
import pandas as pd
import numpy as np
from sklearn import svm
from pic2vec import ImageFeaturizer

```

    Using TensorFlow backend.


### Formatting the Data

'ImageFeaturizer' accepts as input either:
1. An image directory
2. A CSV with URL pointers to image downloads, or
3. A combined image directory + CSV with pointers to the included images.

For this example, we will load in the Kaggle Cats vs. Dogs dataset of 25,000 images, along with a CSV that includes each images class label.


```python
WORKING_DIRECTORY = os.path.expanduser('~') + '/workspace/'

csv_path = WORKING_DIRECTORY + 'cats_vs_dogs.csv'
image_path = WORKING_DIRECTORY + 'cats_vs_dogs_images/'
```

Let's take a look at the csv before featurizing the images:


```python
pd.read_csv(csv_path)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
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
featurizer = ImageFeaturizer(depth=1, auto_sample = False, model='squeezenet')
```

    INFO - Building the featurizer.
    INFO - Loading/downloading SqueezeNet model weights. This may take a minute first time.
    INFO - Model successfully initialized.
    INFO - Model decapitated.
    INFO - Model downsampled.
    INFO - Full featurizer is built.
    INFO - No downsampling. Final layer feature space has size 512


This featurizer was 'decapitated' to the first layer below the prediction layer, which will produce complex representations. Because it is so close to the final prediction layer, it will create more specialized feature representations, and therefore will be better suited for image datasets that are similar to classes within the original ImageNet dataset. Cats and dogs are present within ImageNet, so a depth of 1 should perform well.

## Loading and Featurizing Images Simultaneously

Now that the featurizer is built, we can actually load our data into the network and featurize the images all at the same time, using a single method:


```python
featurized_df = featurizer.featurize(image_column_headers='images',
                                     image_path = image_path,
                                     csv_path = csv_path)
```

    INFO - Found image paths that overlap between both the directory and the csv.
    Loading image batch.
    INFO - Converting images.
    INFO - Converted 0 images in batch. Only 1000 images left to go.
    INFO - Converted 500 images in batch. Only 500 images left to go.
    Featurizing image batch.
    INFO - Trying to featurize data.
    INFO - Creating feature array.
    1000/1000 [==============================] - 16s 16ms/step
    INFO - Feature array created successfully.
    INFO - Adding image features to csv.
    INFO - Number of missing photos: 1000
    Featurized batch #1. Number of images left: 24000
    Estimated total time left: 1149 seconds
    Loading image batch.
    INFO - Converting images.
    INFO - Converted 0 images in batch. Only 1000 images left to go.
    INFO - Converted 500 images in batch. Only 500 images left to go.
    Featurizing image batch.
    INFO - Trying to featurize data.
    INFO - Creating feature array.
    1000/1000 [==============================] - 16s 16ms/step
    INFO - Feature array created successfully.
    INFO - Adding image features to csv.
    INFO - Number of missing photos: 1000
    Featurized batch #2. Number of images left: 23000
    Estimated total time left: 1138 seconds

    ...

    Featurizing image batch.
    INFO - Trying to featurize data.
    INFO - Creating feature array.
    1000/1000 [==============================] - 16s 16ms/step
    INFO - Feature array created successfully.
    INFO - Adding image features to csv.
    INFO - Number of missing photos: 1000
    Featurized batch #25. Number of images left: 0
    Estimated total time left: 0 seconds


The images have now been featurized. The featurized dataframe contains the original csv, along with the generated features appended to the appropriate row, corresponding to each image.

There is also an `images_missing` column, to track which images were missing. Missing image features are generated on a matrix of zeros.

If there are images in the directory that aren't contained in the CSV, or image names in the CSV that aren't in the directory, or even files that aren't valid image files in the directory, have no fear– the featurizer will only try to vectorize valid images that are present in both the CSV and the directory. Any images present in the CSV but not the directory will be given zero vectors, and the order of the image column from the CSV is considered the canonical order for the images.


```python
featurized_df
```


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



As you can see, the `featurize()` function loads the images as tensors, featurizes them using deep learning, and then appends these features to the dataframe in the same row as the corresponding image.

This can be used with both an image directory and a csv with a column containing the image filepaths (as it is in this case). However, it can also be used with just an image directory, in which case it will construct a brand new DataFrame with the image column header specified. Finally, it can be used with just a csv, as long as the image column header contains URLs of each image.

This is the simplest way to use pic2vec, but it is also possible to perform the function in multiple steps. There are actually two processes happening behind the scenes in the above code block:
1. The images are loaded into the network, and then
2. The images are featurized and these features are appended to the csv.



## Loading the Data

In the next sections, I will demonstrate loading and featurizing the images in separate steps, and explain in more depth what happens during each process.

First, we have to load the images into the network. This will parse through the images in the order given by the csv, rescale them to a target size depending on the network (e.g. SqueezeNet is (227, 227))– and build a 5D tensor containing the vectorized representations of the images. This tensor will later be fed into the network in order to be featurized.

The tensor has the following dimensions: `[number_of_image_columns, number_of_images_per_image_column, height, width, color_channels]`. In this case, the image tensor will have size `[1, 25000, 227, 227, 3]`.

If one were to add a second photo of each animal taken from a new angle, the new tensor might have the dimensions `[2, 25000, 227, 227, 3]`, as there would be a second image column being featurized in each row.


To load the images, we have to pass in the name of the column(s) in the CSV containing the image paths, as well as the path to the image directory and the path to the CSV.


**Be aware**:

When both steps are performed at once, the ImageFeaturizer can use batch processing prevent any memory errors. By default, it will featurize batches of 1000 images at once, but this number can be changed to whatever batch size your machine can handle when loading the images into memory.

If you intend to load and featurize your data in separate steps, make sure your machine is capable of storing every image in memory.


```python
featurizer.load_data('images', image_path=image_path, csv_path=csv_path)
```

    INFO - Found image paths that overlap between both the directory and the csv.
    INFO - Converting images.
    INFO - Converted 0 images in batch. Only 25000 images left to go.
    INFO - Converted 1000 images in batch. Only 24000 images left to go.
    INFO - Converted 2000 images in batch. Only 23000 images left to go.
    INFO - Converted 3000 images in batch. Only 22000 images left to go.
    INFO - Converted 4000 images in batch. Only 21000 images left to go.
    INFO - Converted 5000 images in batch. Only 20000 images left to go.
    INFO - Converted 6000 images in batch. Only 19000 images left to go.
    INFO - Converted 7000 images in batch. Only 18000 images left to go.
    INFO - Converted 8000 images in batch. Only 17000 images left to go.
    INFO - Converted 9000 images in batch. Only 16000 images left to go.
    INFO - Converted 10000 images in batch. Only 15000 images left to go.
    INFO - Converted 11000 images in batch. Only 14000 images left to go.
    INFO - Converted 12000 images in batch. Only 13000 images left to go.
    INFO - Converted 13000 images in batch. Only 12000 images left to go.
    INFO - Converted 14000 images in batch. Only 11000 images left to go.
    INFO - Converted 15000 images in batch. Only 10000 images left to go.
    INFO - Converted 16000 images in batch. Only 9000 images left to go.
    INFO - Converted 17000 images in batch. Only 8000 images left to go.
    INFO - Converted 18000 images in batch. Only 7000 images left to go.
    INFO - Converted 19000 images in batch. Only 6000 images left to go.
    INFO - Converted 20000 images in batch. Only 5000 images left to go.
    INFO - Converted 21000 images in batch. Only 4000 images left to go.
    INFO - Converted 22000 images in batch. Only 3000 images left to go.
    INFO - Converted 23000 images in batch. Only 2000 images left to go.
    INFO - Converted 24000 images in batch. Only 1000 images left to go.


The image data is now loaded into the featurizer in one single batch. Like before, the tensor has the following dimensions: `[number_of_image_columns, number_of_images_per_image_column, height, width, color_channels]`.

## Featurizing the Data

Now that the data is loaded, we're ready to featurize the preloaded data. Like in the `featurize()` method, this will push the vectorized images through the network and save the 2D matrix output– each row representing a single image, and each column storing a different feature.

This requires pushing images through the deep network, and so if you choose to use a slower, more powerful model like InceptionV3, large datasets will require a GPU to perform in a reasonable amount of time. Using a low-range GPU, it can take about 30 minutes to process the full 25,000 photos in the Dogs vs. Cats through InceptionV3. On the other hand, if you would like a fast, lightweight model without top-of-the-line accuracy, SqueezeNet works well enough and can perform inference on CPUs quickly.


```python
featurize_preloaded_df = featurizer.featurize_preloaded_data(save_features=True)[0]
```

    INFO - Trying to featurize data.
    INFO - Creating feature array.
    25000/25000 [==============================] - 674s 27ms/step
    INFO - Feature array created successfully.
    INFO - Adding image features to csv.
    INFO - Number of missing photos: 25000



```python
featurize_preloaded_df
```

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

The dataset has now been fully featurized! The features are saved under the featurized_data attribute if the `save_features` argument was set to True in either the `featurize()` or `featurize_preloaded_data()` functions:


```python
featurizer.features
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>images_missing</th>
      <th>images_feat_0</th>
      <th>images_feat_1</th>
      <th>images_feat_2</th>
      <th>images_feat_3</th>
      <th>images_feat_4</th>
      <th>images_feat_5</th>
      <th>images_feat_6</th>
      <th>images_feat_7</th>
      <th>images_feat_8</th>
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
      <td>False</td>
      <td>0.422239</td>
      <td>4.080936</td>
      <td>14.894337</td>
      <td>2.842498</td>
      <td>0.967452</td>
      <td>10.851055</td>
      <td>0.164090</td>
      <td>3.244710</td>
      <td>0.000000</td>
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
      <td>False</td>
      <td>2.235883</td>
      <td>1.766027</td>
      <td>0.489503</td>
      <td>1.077848</td>
      <td>3.744066</td>
      <td>3.900755</td>
      <td>0.678774</td>
      <td>0.039899</td>
      <td>2.031924</td>
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
      <td>False</td>
      <td>0.804545</td>
      <td>0.685238</td>
      <td>0.411905</td>
      <td>3.651519</td>
      <td>7.440580</td>
      <td>1.365789</td>
      <td>1.759454</td>
      <td>0.458921</td>
      <td>1.410855</td>
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
      <td>False</td>
      <td>0.481214</td>
      <td>0.229483</td>
      <td>5.039218</td>
      <td>0.669226</td>
      <td>3.988109</td>
      <td>2.878755</td>
      <td>0.642729</td>
      <td>0.366843</td>
      <td>0.000000</td>
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
      <td>False</td>
      <td>0.000000</td>
      <td>3.258759</td>
      <td>9.666997</td>
      <td>6.237058</td>
      <td>1.160069</td>
      <td>0.055264</td>
      <td>0.394765</td>
      <td>0.476731</td>
      <td>0.611324</td>
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
      <td>False</td>
      <td>0.319323</td>
      <td>10.394070</td>
      <td>2.253926</td>
      <td>17.211163</td>
      <td>10.175515</td>
      <td>0.754048</td>
      <td>0.000000</td>
      <td>1.031596</td>
      <td>1.111096</td>
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
      <td>False</td>
      <td>4.812177</td>
      <td>2.173984</td>
      <td>3.127878</td>
      <td>7.731273</td>
      <td>2.563596</td>
      <td>0.855450</td>
      <td>1.506930</td>
      <td>2.816796</td>
      <td>0.108556</td>
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
      <td>False</td>
      <td>0.603782</td>
      <td>10.804434</td>
      <td>5.988084</td>
      <td>11.258055</td>
      <td>1.711970</td>
      <td>0.172748</td>
      <td>0.839580</td>
      <td>1.017261</td>
      <td>0.428469</td>
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
      <td>False</td>
      <td>0.012421</td>
      <td>0.779886</td>
      <td>3.646794</td>
      <td>1.577259</td>
      <td>0.314669</td>
      <td>1.035333</td>
      <td>0.140289</td>
      <td>0.526032</td>
      <td>1.808931</td>
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
      <td>False</td>
      <td>2.025731</td>
      <td>2.882919</td>
      <td>7.076093</td>
      <td>3.225270</td>
      <td>0.053434</td>
      <td>0.481757</td>
      <td>0.222437</td>
      <td>0.000000</td>
      <td>2.238212</td>
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
<p>25000 rows × 513 columns</p>
</div>



The dataframe can be saved in CSV form either by calling the pandas `DataFrame.to_csv()` method, or by using the `ImageFeaturizer.save_csv()` method on the featurizer itself. This will allow the features to be used directly in the DataRobot app:


```python
featurizer.save_csv()
```

    WARNING - Saving full dataframe to csv as /Users/jett.oristaglio/workspace/cats_vs_dogs_squeezenet_depth-1_output-512_(02-Aug-2018-03.03.49)_full.csv



```python
pd.read_csv(WORKING_DIRECTORY + 'cats_vs_dogs_squeezenet_depth-1_output-512_(02-Aug-2018-03.03.49)_full.csv')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
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
      <td>3.744067</td>
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
      <td>0.991105</td>
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
      <td>0.481215</td>
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
      <td>1.215900</td>
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
      <td>7.819313</td>
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
      <td>1.023680</td>
      <td>2.697655</td>
      <td>2.661853</td>
      <td>0.294249</td>
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



The `save_csv()` function can be called with no arguments in order to create an automatic csv name, like above. It can also be called with the `new_csv_path='{insert_new_csv_path_here}'` argument.

Alternatively, you can omit certain parts of the automatic name generation with `omit_model=True`, `omit_depth=True`, `omit_output=True`, or `omit_time=True` arguments.

But, for the purposes of this demo, we can simply test the performance of a linear SVM classifier over the featurized data. First, we'll build the training and test sets.


```python
# Creating a training set of 10,000 for each class
train_cats = featurized_df.iloc[:10000, :]
train_dogs = featurized_df.iloc[12500:22500, :]

# building training set from 12,500 images of each class
train_cats, labels_cats = train_cats.drop(['label', 'images'], axis=1), train_cats['label']
train_dogs, labels_dogs = train_dogs.drop(['label', 'images'], axis=1), train_dogs['label']

# Combining the train data and the class labels to train on
train_combined = pd.concat((train_cats, train_dogs), axis=0)
labels_train = pd.concat((labels_cats, labels_dogs), axis=0)

# Creating a test set from the remaining 2,500 of each class
test_cats = featurized_df.iloc[10000:12500, :]
test_dogs = featurized_df.iloc[22500:, :]

test_cats, test_labels_cats = test_cats.drop(['label', 'images'], axis=1), test_cats['label']
test_dogs, test_labels_dogs = test_dogs.drop(['label', 'images'], axis=1), test_dogs['label']

# Combining the test data and the class labels to check predictions
labels_test = pd.concat((test_labels_cats, test_labels_dogs), axis=0)
test_combined = pd.concat((test_cats, test_dogs), axis=0)

```

Then, we'll train the linear SVM:


```python
# Initialize the linear SVC
clf = svm.LinearSVC()

# Fit it on the training data
clf.fit(train_combined, labels_train)

# Check the performance of the linear classifier over the full Cats vs. Dogs dataset!
clf.score(test_combined, labels_test)
```




    0.9632



After running the Cats vs. Dogs dataset through the lightest-weight pic2vec model, we find that a simple linear SVM trained over the featurized data achieves over 96% accuracy on distinguishing dogs vs. cats out of the box.

## Summary

That's it! We've looked at the following:

1. What data formats can be passed into the featurizer
2. How to initialize a simple featurizer
3. How to load and featurize the data simultaneously (preferred method)
3. How to load data into the featurizer independently
4. How to featurize the loaded data independently
5. How to save the featurized dataframe as a csv

And as a bonus, we looked at how we might use the featurized data to perform predictions without dropping the CSV into the DataRobot app.

Unless you would like to examine the loaded data before featurizing it, it is recommend to use the `ImageFeaturizer.featurize()` method to perform both functions at once and allow batch processing.

## Next Steps

We have not covered using only a CSV with URL pointers, or a more complex dataset. That will be the subject of another Notebook.

To have more control over the options in the featurizer, or to understand its internal functioning more fully, check out the full package documentation.
