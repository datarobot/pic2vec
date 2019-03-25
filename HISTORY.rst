=======
History
=======
0.100.2 (2019-3-25)
------------------
* Updated version of Trafaret to a non-beta version

0.100.1 (2019-3-24)
------------------
* Updated version of Pillow to 5.4.1, in order to support Python 3.7
* Updated the README

0.100.0 (2018-12-10)
------------------
* Added test coverage and increased error checking
* Changed default csv name
* Changed `image_column_headers` to `image_columns` everywhere
* Updated examples
* Updated version of scipy to 1.1 and numpy to 1.15


0.99.2 (2018-08-01)
------------------
* Updated the notebook example
* Some code cleanup

0.99.1 (2018-06-20)
------------------
* Lots of code cleanup
* Changed new_csv_name argument to new_csv_path everywhere for consistency
* Removed '_full' from the saved csv_name for the full dataframe. Features-only csv still has
  '_features_only' in csv name.
* Added '_featurized_' to saved csv names
* Removed new_csv_path as argument to functions that do not actually require it

0.99.0 (2018-04-02)
------------------
* Added batch processing
* Made pic2vec more programmatic (removed automatic csv-writing, etc.)
* Bound keras to <2.1.5 to remove resnet problem

0.9.0 (2017-09-24)
------------------
* Fixed Keras backwards compatibility issues (include_top deprecated, require_flatten added)
* Fixed ResNet50 update issues (removed a zero-padding layer, updated weights)

0.8.2 (2017-08-14)
------------------
* Updated trafaret requirement for PyPi package
* Updated cats vs. dogs example

0.8.1 (2017-08-07)
------------------
* Fixed bugs with robust naming
* Added error message for failed image conversion

0.8.0 (2017-08-02)
------------------
* Added robust naming options to the generated csv files

0.7.1 (2017-08-02)
------------------
* Fixed PIL truncated image bug

0.7.0 (2017-08-02)
------------------
* Fixed bug with CSV badly formed URLs
* Fixed mistake with InceptionV3 preprocessing happening for every model

0.6.3 (2017-07-25)
------------------
* Added Travis and Coveralls for testing and coverage automation
* Repo went public
* Python 3.x compatibility

0.6.2 (2017-07-14)
------------------
* Fixed image format recognition.

0.6.1 (2017-07-12)
------------------
* Directory-only now natural sorted.

0.6.0 (2017-07-11)
------------------
* Added multi-column support
* Added missing image column to csv

0.5.0 (2017-07-06)
------------------
* Renamed to pic2vec
* Tests parametrized

0.4.3 (2017-07-03)
------------------
* Second round of code review- optimized code, better type checking with trafaret

0.4.2 (2017-06-30)
------------------
* Improved README test examples

0.4.1 (2017-06-30)
------------------
* Fixed documentation

0.4.0 (2017-06-29)
------------------
* Added ability to call multiple models, and packaged in SqueezeNet with weights.

0.3.0 (2017-06-26)
------------------
* Created installation instructions and readme files, ready for prototype distribution

0.2.9(2017-06-25)
------------------
* Fixed import problem that prevented generated csvs from saving

0.2.8(2017-06-25)
------------------
* Fixed variable name bugs

0.2.7(2017-06-25)
------------------
* Changed image_directory_path to the more manageable image_path
* Made testing module and preprocessing module slightly more robust.

0.2.6(2017-06-23)
------------------
* Added features-only csv test, and got rid of the column headers in the file
* Added Documentation to data featurization modeules

0.2.5(2017-06-23)
------------------
* 100% test coverage
* Fixed a problem where a combined directory + csv was appending to the wrong
  rows when there was a mismatch between the directory and the csv.

0.2.4(2017-06-22)
------------------
* Fixed more bugs in build_featurizer

0.2.3(2017-06-22)
------------------
* Fixed build_featurizer troubles with building new csv paths in current directory

0.2.2(2017-06-22)
------------------
* Full requirements for keras imported

0.2.1 (2017-06-22)
------------------
* Bug fixes

0.2.0 (2017-06-22)
------------------
* Second release on PyPI.
* Install keras with tensorflow backend specifically

0.1.0 (2017-06-14)
------------------
* First release on PyPI.
