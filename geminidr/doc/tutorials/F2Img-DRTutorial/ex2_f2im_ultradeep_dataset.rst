.. ex2_f2im_ultradeep_dataset.rst

.. _ultradeep_dataset:

*********************************
Example 2 - Datasets descriptions
*********************************

Deep observation
----------------

This is a Flamingos-2 imaging observation of a rather sparse field but with
the objective of going deep.   We will use this observation to show and
discuss the ``ultradeep`` near-infrared imaging recipe.

The observation is a very long (~175 images) dithered-on-target sequence, with
nine positions. DRAGONS will recognized such a sequence and will identify
frames in the sequence to use for sky subtraction.  We are using just two
dither cycles (18 images) for this tutorial to limit the amount of data needed.
For science, the whole sequence should be fed to the recipe.


The data used here is obtained with the K-red filter.  Please check the
:ref:`tips_and_tricks` section for information about the other bands.

The calibrations we use in this example are:

* Darks for the science frames.
* Darks for the K-band flats.
* Flats as a sequence of lamp-off exposures that have enough thermal flux to be
  used as "lamp-on" exposures.

The table below contains a summary of the files needed for this example:

+-----------+---------------------+---------------------------------------+
| Science   || S20200104S0075-092 | K-red, 5 s                            |
+-----------+---------------------+---------------------------------------+
| Darks     || S20200107S0035-041 | 2 s, darks for flats                  |
|           || S20200111S0257-260 | 2 s, darks for flats                  |
|           +---------------------+---------------------------------------+
|           || S20200107S0049-161 | 5 s, for science dat                  |
+-----------+---------------------+---------------------------------------+
| Flats     || S20200108S0010-019 | 2 s, Lamp-off used as lamp-on, K-red  |
+-----------+---------------------+---------------------------------------+
