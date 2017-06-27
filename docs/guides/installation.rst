.. highlight:: shell

============
Installation
============

Installing In Virtualenv
---------------------------------
To install virtualenv, follow this guide: `virtualenv installation guide`_

Once virtualenv is installed, create a new environment to run the featurizer:

.. code-block:: console

    $ virtualenv featurizer

Then activate the environment:

.. code-block:: console

    $ source featurizer/bin/activate

Once in the virtual environment, there are several ways to install the
image_featurizer package.

.. _virtualenv installation guide: http://sourabhbajaj.com/mac-setup/Python/virtualenv.html

Installing In Conda
------------------------
To install Anaconda, follow this guide: `Anaconda installation guide`_

Once Anaconda is installed,. create a new environment to run the featurizer:

.. code-block:: console

    $ conda create --name featurizer

When Conda asks for confirmation, type 'y' for 'yes'.

To activate the environment on OS X or Linux:
.. code-block:: console

    $ source activate featurizer

To activate the environment on Windows:
.. code-block:: console

    $ activate featurizer

Once in a virtual environment, there are several ways to install the
image_featurizer package.

.. _Anaconda installation guide: https://docs.continuum.io/anaconda/install


Install Through Pip
-------------------

To install Image Featurizer through pip on OS X or Linux, run this command in your terminal:

.. code-block:: console

    $ pip install image_featurizer

To install through pip on Windows, run this command in terminal:

.. code-block:: console

    $ python -m pip install image_featurizer

This is the preferred method to install Image Featurizer, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Image Featurizer can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/datarobot/imagefeaturizer

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/datarobot/imagefeaturizer/tarball/master

Once you have a copy of the source, you can install it from inside the directory with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/datarobot/imagefeaturizer
.. _tarball: https://github.com/datarobot/imagefeaturizer/tarball/master
