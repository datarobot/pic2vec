=======
History
=======

0.1.0 (2017-06-14)
------------------

* First release on PyPI.

0.2.0 (2017-06-22)
------------------

* Second release on PyPI.
* Install keras with tensorflow backend specifically

0.2.1 (2017-06-22)
------------------

* Bug fixes

0.2.2(2017-06-22)
------------------

* Full requirements for keras imported

0.2.3(2017-06-22)
------------------

* Fixed build_featurizer troubles with building new csv paths in current directory

0.2.4(2017-06-22)
------------------

* Fixed more bugs in build_featurizer

0.2.5(2017-06-23)
------------------
* 100% test coverage
* Fixed a problem where a combined directory + csv was appending to the wrong
  rows when there was a mismatch between the directory and the csv.

 0.2.6(2017-06-23)
 ------------------
* Added features-only csv test, and got rid of the column headers in the file
* Added Documentation to data featurization modeules

0.2.7(2017-06-25)
------------------
* Changed image_directory_path to the more manageable image_path
* Made testing module and preprocessing module slightly more robust.


0.2.8(2017-06-25)
------------------
* Fixed variable name bugs


0.2.9(2017-06-25)
------------------
* Fixed import problem that prevented generated csvs from saving

0.3.0 (2017-06-26)
------------------
* Created installation instructions and readme files, ready for prototype distribution

0.4.0 (2017-06-29)
------------------
* Added ability to call multiple models, and packaged in SqueezeNet with weights.

0.4.1 (2017-06-30)
------------------
* Fixed documentation

0.4.2 (2017-06-30)
------------------
* Improved README test examples

0.4.3 (2017-07-03)
------------------
* Second round of code review- optimized code, better type checking with trafaret

0.5.0 (2017-07-06)
------------------
* Renamed to pic2vec
* Tests parametrized

0.6.0 (2017-07-11)
------------------
* Added multi-column support
* Added missing image column to csv

0.6.1 (2017-07-12)
------------------
* Directory-only now natural sorted.
