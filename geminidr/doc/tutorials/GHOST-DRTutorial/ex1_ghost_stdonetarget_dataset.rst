.. ex1_ghost_stdonetarget_dataset.rst

.. _datastdonetarget:

*********************************
Example 1 - Datasets descriptions
*********************************

One Point Source in Standard Resolution
---------------------------------------
This is a GHOST observation of the star XX Oph obtained in standard resolution
IFU-1 is pointing at the star, IFU-2 is stowed.  This observation was obtained
during the April 2023 commissioning run.

GHOST can require a lot of calibration files.  Also, the raw data bundle the
red and blue channel data, as well as the slit-viewer data.

The calibrations we use for this example are:

* Biases.  We need biases that match the binning and read mode of the science
  frames.  We also need biases that match the binning and read mode of the
  flats and the arcs.  The matches must cover both the red and blue arms.
  The slit view camera has only one configuration, so the match is guaranteed
  for the slit view images.
* Flats.  The flats must match the read mode of the science.  The binning
  must be 1x1.  They will be binned in software to match the science. The
  matches must cover both the red and blue arms.
* Arcs.  The binning for the arcs must be 1x1.  The read mode does not matter,
  but they require their own biases and flats and it is therefore simpler if
  they have the same read mode as the science data.
* Spectrophotometric standard.  The binning must match the science.
* BPMs. The bad pixel masks are found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations. Separate BPMs are required for the blue and red arms.
  There is no BPM for the slit-viewer camera.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+-----------------+-------------------------------------------------+
| Science         || S20230416S0079 (blue:2x2,slow; red:2x2,medium) |
+-----------------+-------------------------------------------------+
| Science biases  || S20230417S0011-015                             |
+-----------------+-------------------------------------------------+
| Science Flats   || S20230416S0047 (1x1; blue:slow; red:medium)    |
+-----------------+-------------------------------------------------+
| Science Arcs    || S20230416S0049-51 (1x1; blue:slow; red:medium) |
+-----------------+-------------------------------------------------+
| Flats Biases    || S20230417S0036-40 (1x1; blue:slow; red:medium) |
+-----------------+                                                 |
| Arc Biases      ||                                                |
+-----------------+-------------------------------------------------+
| Standard        || S20230416S0073 (blue:2x2,slow; red:2x2,medium) |
| (CD -32 9927)   ||                                                |
+-----------------+-------------------------------------------------+
| Standard biases || In this case, the calibrations for the         |
+-----------------+  science can be used for the standard star.     |
| Standard flats  ||                                                |
+-----------------+                                                 |
| Standard arc    ||                                                |
+-----------------+                                                 |
| Std flat biases ||                                                |
+-----------------+                                                 |
| Std arc biases  ||                                                |
+-----------------+-------------------------------------------------+
+ BPMs            || bpm_20220601_ghost_blue_11_full_4amp.fits      |
|                 || bpm_20220601_ghost_red_11_full_4amp.fits       |
|                 ||                                                |
|                 || ** Not in data package.  Need to be retrieved  |
|                 || ** from the archive.                           |
+-----------------+-------------------------------------------------+


BPMs from the archive
---------------------
.. The BPMs are not included in the data package.  If you have added the
  ``archive.gemini.edu`` in the list of databases in ``.dragonsrc``, they will be
  automatically associated and retrieved.


You can download them from
here: https://archive.gemini.edu/searchform/GHOST/notengineering/cols=CTOWBEQ/NotFail/not_site_monitoring/BPM

Put them in ``playdata/example1/``.
