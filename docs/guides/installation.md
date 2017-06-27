Installation
============

Installing In Virtualenv
---------------------------------
To install virtualenv, follow this guide: [virtualenv installation guide](http://sourabhbajaj.com/mac-setup/Python/virtualenv.html)

Once virtualenv is installed, create a new environment to run the featurizer:

```bash
    $ virtualenv featurizer
```
Then activate the environment:

```bash
    $ source featurizer/bin/activate
```

Once in the virtual environment, there are several ways to install the
image_featurizer package.


Installing In Conda
-------------------
To install Anaconda, follow this guide: [Anaconda installation guide](https://docs.continuum.io/anaconda/install)

Once Anaconda is installed,. create a new environment to run the featurizer:

```bash
    $ conda create --name featurizer
```

When Conda asks for confirmation, type 'y' for 'yes'.

To activate the environment on OS X or Linux:

```bash
    $ source activate featurizer
```

To activate the environment on Windows:
```bash
    $ activate featurizer
```

Once in a virtual environment, there are several ways to install the
image_featurizer package.



Install Through Pip
-------------------

To install Image Featurizer through pip on OS X or Linux, run this command in your terminal:

```bash
    $ pip install image_featurizer
```
To install through pip on Windows, run this command in terminal:

```bash
    $ python -m pip install image_featurizer
```

This is the preferred method to install Image Featurizer, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.



From sources
------------

The sources for Image Featurizer can be downloaded from the [Github repo](https://github.com/datarobot/imagefeaturizer).

You can either clone the public repository:

```bash
    $ git clone git://github.com/datarobot/imagefeaturizer
```
Or download the [tarball](https://github.com/datarobot/imagefeaturizer/tarball/master):

```bash
    $ curl  -OL https://github.com/datarobot/imagefeaturizer/tarball/master
```

Once you have a copy of the source, you can install it from inside the directory with:

```bash
    $ python setup.py install
```
