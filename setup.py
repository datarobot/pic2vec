#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'h5py',
    'scipy',
    'scikit-learn',
    'numpy',
    'tensorflow',
    'keras',
    'pandas',
    'Pillow',
]

setup_requirements = [
    'pytest-runner',
    # Put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'numpy'
    'pytest',
    'keras',
]

setup(
    name='image_featurizer',
    version='0.3.0',
    description="Featurize images using a decapitated, pre-trained deep learning network",
    long_description=readme + '\n\n' + history,
    author="Jett Oristaglio",
    author_email='jettori88@gmail.com',
    url='https://github.com/datarobot/imagefeaturizer',
    packages=find_packages(include=['image_featurizer']),
    entry_points={
        'console_scripts': [
            'image_featurizer=image_featurizer.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords=['image_featurizer','featurize'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
