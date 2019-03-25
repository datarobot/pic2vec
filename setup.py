#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'h5py>=2.7.0,<3',
    'scipy>=1.1,<2',
    'numpy>=1.15.4,<2',
    'tensorflow>=1.2.0,<2',
    'keras>=2.0.8,<2.1.5',
    'pandas>=0.20.2,<1',
    'Pillow>=5.4.1,<6',
    'trafaret>=1,<2'
]

setup_requirements = [
    'pytest-runner',
    # Put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'numpy',
    'pytest',
    'keras',
]

setup(
    name='pic2vec',
    version='0.100.1',
    description='Featurize images using a decapitated, pre-trained deep learning network',
    long_description=readme + '\n\n' + history,
    author='Jett Oristaglio',
    author_email='jettori88@gmail.com',
    url='https://github.com/datarobot/pic2vec',
    packages=find_packages(include=['pic2vec']),
    include_package_data=True,
    package_data={
        'pic2vec': ['saved_models/squeezenet_weights_tf_dim_ordering_tf_kernels.h5']
        },
    install_requires=requirements,
    license='BSD license',
    zip_safe=False,
    keywords=['image_featurizer', 'featurize', 'pic2vec'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
