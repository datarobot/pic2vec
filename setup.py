#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'keras',
    'pandas',
    'Pillow'
    'numpy'
]

setup_requirements = [
    'pytest-runner',
    # TODO(joristaglio): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    'keras',
    'numpy'
]

setup(
    name='image_featurizer',
    version='0.1.0',
    description="Featurize images using a small, contained, pre-trained deep learning network",
    long_description=readme + '\n\n' + history,
    author="Jett Oristaglio",
    author_email='jettori88@gmail.com',
    url='https://github.com/joristaglio/image_featurizer',
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
    keywords='image_featurizer',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
