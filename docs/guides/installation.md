Installation:
============


1: Setting Up The Virtual Environment
---------------------------------

### VirtualEnv
To install virtualenv, follow this guide: [virtualenv installation guide](http://sourabhbajaj.com/mac-setup/Python/virtualenv.html)

Once virtualenv is installed, create a new environment to run pic2vec:

```bash
    $ virtualenv pic2vec
```
Then activate the environment:

```bash
    $ source pic2vec/bin/activate
```

### Conda
To install Anaconda, follow this guide: [Anaconda installation guide](https://docs.continuum.io/anaconda/install)

Once Anaconda is installed, create a new environment to run pic2vec:

```bash
    $ conda create --name pic2vec
```

When Conda asks for confirmation, type 'y' for 'yes'.

To activate the environment on OS X or Linux:

```bash
    $ source activate pic2vec
```

To activate the environment on Windows:
```bash
    $ activate pic2vec
```


Once in a virtual environment, there are several ways to install the
pic2vec package.



2: Installing The Pic2Vec Package
-------------------

### Pip Installation
To install pic2vec through pip on OS X or Linux, run this command in your terminal:

```bash
    $ pip install pic2vec
```
To install through pip on Windows, run this command in terminal:

```bash
    $ python -m pip install pic2vec
```

This is the preferred method to install pic2vec, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.


### Installing From setup.py
The sources for pic2vec can be downloaded from the [Github repo](https://github.com/datarobot/pic2vec).

You can either clone the public repository:

```bash
    $ git clone git@github.com:datarobot/pic2vec.git
```
Or download the [tarball](https://github.com/datarobot/pic2vec/tarball/master):

```bash
    $ curl  -OL https://github.com/datarobot/pic2vec/tarball/master
```

Once you have a copy of the source, you can install it from inside the directory with:
FIXME: this actually isn't working, at least in Ubuntu16.04 this fails on dependencies installation.

```bash
    $ python setup.py install
```


3: Troubleshooting
---------------

1. If you see error similar to `TypeError: find_packages() got an unexpected
keyword argument 'include'` then you need to upgrade your setuptools.

```bash
pip install -U setuptools
```

2. If you see error similar to `No local packages or working download links
found for tensorflow`  then you need to upgrade your pip.

```bash
pip install -U pip
```

3. If you have problems with tests or strange runtime exceptions - make sure
your Keras installation don't configured for Theano use. Open `~/.keras/keras.json`
and check that `backend` parameter value is `tensorflow`. If it is `theano` -
simply remove that file, on next execution Keras will find your tensorflow
and create correct configuration file.
