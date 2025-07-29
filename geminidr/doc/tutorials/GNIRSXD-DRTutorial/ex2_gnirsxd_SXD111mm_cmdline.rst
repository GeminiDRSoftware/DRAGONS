.. ex2_gnirsxd_SXD111mm_cmdline.rst

.. include:: symbols.txt

.. _gnirsxd_SXD32mm_cmdline:

******************************************************************
Example 2 - SXD+111 Point Source - Using the "reduce" command line
******************************************************************

In this example, we will reduce the GNIRS crossed-dispersed observation of
an erupting recurrent nova using the "|reduce|" command that
is operated directly from the unix shell.  Just open a terminal and load the
DRAGONS conda environment to get started.

This cross-dispersed observation uses the 111 l/mm grating, the short-blue
camera and the 0.3 arcsec slit.  The dither pattern is the standard ABBA, one
set for each of the three central wavelength settings.  The results from the
three wavelength settings will be stitched together at the end.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`gnirsxd_SXD32mm_dataset`

Here is a copy of the table for quick reference.

+---------------------+----------------------------------------------+
| Science             || N20190928S0085-88  (1.55 |um|)              |
|                     || N20190928S0090-93  (1.68 |um|)              |
|                     || N20190928S0094-97  (1.81 |um|)              |
+---------------------+----------------------------------------------+
| Science flats       || N20190928S0117-132 (1.55 |um|)              |
|                     || N20190928S0135-150 (1.68 |um|)              |
|                     || N20190928S0153-168 (1.81 |um|)              |
+---------------------+----------------------------------------------+
| Pinholes            || None available                              |
+---------------------+----------------------------------------------+
| Science arcs        || N20190928S0115-116 (1.55 |um|)              |
|                     || N20190928S0133-134 (1.68 |um|)              |
|                     || N20190928S0151-152 (1.81 |um|)              |
+---------------------+----------------------------------------------+
| Telluric            || N20190928S0103-106 (1.55 |um|)              |
|                     || N20190928S0107-110 (1.68 |um|)              |
|                     || N20190928S0111-114 (1.81 |um|)              |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20121101_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+


Configuring the interactive interface
=====================================
In ``~/.dragons/``, add the following to the configuration file ``dragonsrc``::

    [interactive]
    browser = your_preferred_browser

The ``[interactive]`` section defines your preferred browser.  DRAGONS will open
the interactive tools using that browser.  The allowed strings are "**safari**",
"**chrome**", and "**firefox**".

Set up the Local Calibration Manager
====================================

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_cmdline`.

We recommend that you clean up your working directory (``playground``) and
start a fresh calibration database (``caldb init -w``) when you start a new
example.

Create file lists
=================

This data set contains science and calibration frames. For some programs, it
could contain different observed targets and different exposure times depending
on how you like to organize your raw data.

The DRAGONS data reduction pipeline does not organize the data for you.  You
have to do it.  However, DRAGONS provides tools to help you with that.

The first step is to create input file lists.  The tool "|dataselect|" helps.
It uses Astrodata tags and |descriptors| to select the files and
send the filenames to a text file that can then be fed to "|reduce|".  (See the
|astrodatauser| for information about Astrodata and for a list
of |descriptors|.)

First, navigate to the ``playground`` directory in the unpacked data package::

    cd <path>/gnirsxd_tutorial/playground


Three lists for the flats
-------------------------
The GNIRS XD flats are obtained using two different lamps to ensure that each
order is illuminated at a sufficient level.  The software will stack each set
and automatically assemble the orders into a new flat with all orders well
illuminated.

The particularily of this dataset is that there are three central wavelength
settings that each need to be reduced separately.

You will use "|dataselect|" to select each set of flats associated with the
configurations used for the science observations.

But first, to see which central wavelengths have been used, run |showd| on
the flats.

::

    dataselect ../playdata/example2/*.fits --tags FLAT | showd -d central_wavelength

    -------------------------------------------------------------
    filename                                   central_wavelength
    -------------------------------------------------------------
    ../playdata/example2/N20190928S0117.fits             1.55e-06
    ../playdata/example2/N20190928S0118.fits             1.55e-06
    ...
    ../playdata/example2/N20190928S0135.fits             1.68e-06
    ../playdata/example2/N20190928S0136.fits             1.68e-06
    ...
    ../playdata/example2/N20190928S0153.fits             1.81e-06
    ../playdata/example2/N20190928S0154.fits             1.81e-06

::

    dataselect ../playdata/example2/*.fits --tags FLAT --expr='central_wavelength==1.55e-6' -o flat155.lis
    dataselect ../playdata/example2/*.fits --tags FLAT --expr='central_wavelength==1.68e-6' -o flat168.lis
    dataselect ../playdata/example2/*.fits --tags FLAT --expr='central_wavelength==1.81e-6' -o flat181.lis

Note that we have downloaded only the September data from that program.  If
the October data were also in our raw data directory, we would have to add
a date constraint to the expression, like this:

    dataselect ../playdata/example2/*.fits --tags FLAT --expr='central_wavelength==1.55e-6 and ut_date=="2019-09-28"' -o flatSep155.lis


A list for the pinholes
-----------------------
This program does not use a pinholes observation.

The orders in the cross-dispersed raw data are significantly slanted and curved
on the detector.  A pinhole would trace that curvature.

However, the edges of the orders in the processed flat can be used to determine
the position of each order, the pinholes observations simply lead to a more
accurate model of the order positions and of the spatial distortion component.

We do not have pinholes, therefore all steps related to pinholes, their creation
and their usage will be skipped in this tutorial.

If you had pinholes, you would select them like for the flats above using
"PINHOLE" instead of "FLAT".


Three lists for the arcs
------------------------
The GNIRS cross-dispersed arcs were obtained between the telluric and the
science observation.  Often two are taken for each configuration.  If we
decide to use both, they will be stacked.

Here, like for the flats, we need to create a list for each of the three
configurations.

::

    dataselect ../playdata/example2/*.fits --tags ARC --expr='central_wavelength==1.55e-6' -o arc155.lis
    dataselect ../playdata/example2/*.fits --tags ARC --expr='central_wavelength==1.68e-6' -o arc168.lis
    dataselect ../playdata/example2/*.fits --tags ARC --expr='central_wavelength==1.81e-6' -o arc181.lis


Three lists for the telluric
----------------------------
DRAGONS does not recognize the telluric star as such.  This is because, at
Gemini, the observations are taken like science data and the GNIRS headers do not
explicitly state that the observation is a telluric standard.  In most cases,
the ``observation_class`` descriptor can be used to differentiate the telluric
from the science observations, along with the rejection of the ``CAL`` tag to
reject flats and arcs.  Telluric stars will be observed under the ``partnerCal``
or ``progCal`` classes, the science observation under the ``science`` class.

::

    dataselect ../playdata/example2/*.fits --xtags=CAL --expr='observation_class!="science" and central_wavelength==1.55e-6' -o tel155.lis
    dataselect ../playdata/example2/*.fits --xtags=CAL --expr='observation_class!="science" and central_wavelength==1.68e-6' -o tel168.lis
    dataselect ../playdata/example2/*.fits --xtags=CAL --expr='observation_class!="science" and central_wavelength==1.81e-6' -o tel181.lis


A list for the science observations
-----------------------------------

The science observations can be selected from the "observation class"
``science``.  This is how they are differentiated from the telluric
standards which are set to ``partnerCal`` or ``progCal``.

We already know that we have multiple central_wavelength settings and that we
will need a list of each.

If we had multiple targets, we would need to split them into separate lists. To
inspect what we have we can use |dataselect| and |showd| together.

::

    dataselect ../playdata/example2/*.fits --expr='observation_class=="science"' | showd -d object,central_wavelength

    -------------------------------------------------------------------------
    filename                                      object   central_wavelength
    -------------------------------------------------------------------------
    ../playdata/example2/N20190928S0085.fits   V3890 Sgr             1.55e-06
    ../playdata/example2/N20190928S0086.fits   V3890 Sgr             1.55e-06
    ../playdata/example2/N20190928S0087.fits   V3890 Sgr             1.55e-06
    ../playdata/example2/N20190928S0088.fits   V3890 Sgr             1.55e-06
    ../playdata/example2/N20190928S0090.fits   V3890 Sgr             1.68e-06
    ../playdata/example2/N20190928S0091.fits   V3890 Sgr             1.68e-06
    ../playdata/example2/N20190928S0092.fits   V3890 Sgr             1.68e-06
    ../playdata/example2/N20190928S0093.fits   V3890 Sgr             1.68e-06
    ../playdata/example2/N20190928S0094.fits   V3890 Sgr             1.81e-06
    ../playdata/example2/N20190928S0095.fits   V3890 Sgr             1.81e-06
    ../playdata/example2/N20190928S0096.fits   V3890 Sgr             1.81e-06
    ../playdata/example2/N20190928S0097.fits   V3890 Sgr             1.81e-06

Here we only have one object from the same sequence.  If we had multiple
objects we could add the object name in the expression.

::

    dataselect ../playdata/example2/*.fits --expr='observation_class=="science" and central_wavelength==1.55e-6 and object=="V3890 Sgr"' -o sci155.lis
    dataselect ../playdata/example2/*.fits --expr='observation_class=="science" and central_wavelength==1.68e-6 and object=="V3890 Sgr"' -o sci168.lis
    dataselect ../playdata/example2/*.fits --expr='observation_class=="science" and central_wavelength==1.81e-6 and object=="V3890 Sgr"' -o sci181.lis


Bad Pixel Mask
==============
The bad pixel masks (BPMs) are handled as calibrations.  They are downloadable
from the archive instead of being packaged with the software. They are
automatically associated like any other calibrations.  This means that the
user now must download the BPMs along with the other calibrations and add the
BPMs to the local calibration manager.

See :ref:`getBPM` in :ref:`tips_and_tricks` to learn about the various ways
to get the BPMs from the archive.

To add the static BPM included in the data package to the local calibration
database:

::

    caldb add ../playdata/example2/bpm*.fits


Master Flat Field
=================
GNIRS XD flat fields are normally obtained at night along with the
observation sequence to match the telescope and instrument flexure.  The
processed flat is constructed from two sets of stacked lamp-on flats, each illuminated
differently to ensure that all orders in the reassembled flat are well
illuminated.  You do not have to worry about the details, as long as you
pass the two sets of raw flats as input to the ``reduce`` command, the software will take
care of the assembly.

The processed flat will also contain the illumination mask that identify the location
of the illuminated areas in the array, ie, where the orders are located.

Each central wavelength settings must be reduced separately.

::

    reduce @flat155.lis
    reduce @flat168.lis
    reduce @flat181.lis

It might be useful to run the flat reduction in interactive mode.

::

    reduce @flat155.lis -p interactive=True
    reduce @flat168.lis -p interactive=True
    reduce @flat181.lis -p interactive=True

The interactive tools are introduced in section :ref:`interactive`.

.. image:: _graphics/gnirsxd_???.png
   :width: 600
   :alt: ???


Processed Pinholes - Rectification
==================================
The pinholes are used to determine the rectification of the slanted and  curved
orders.

This program does not have pinholes associated with it.  It is okay, the
edges of the orders have been traced when the flats were reduced.  This
can be used for the rectification.  The spatial axis is not as well sampled
but depending on the science using just the edges from the flat can be
sufficient.

If you had pinhole observation, just like the flats they would need to be
reduced each configuration separately (eg. `reduce @pinhole155.lis`)


Processed Arc - Wavelength Solution
===================================
Obtaining the wavelength solution for GNIRS cross-dispersed data can be a
complicated topic.  The quality of the results and what to use depend greatly
on the wavelength regime and the grating.

.. important::
     Do pay great attention to the wavelength calibration. It is critical
     to the telluric modelling.  It can be particularly challenging with the
     111 l/mm grating given the limited wavelength range each order covers.

Our configurations in this example is cross-dispersed with short-blue camera,
the SXD prism, and the 111 l/mm grating.  With this grating, each order covers
only a short waveelength range.  Some orders will not contain a sufficient
number of lines from the arc lamp.  In some cases, we will have to use the
sky emission lines, or even the telluric absorption features to get an
accurate enough wavelength solution for an order.

With the 111 l/mm grating, we are likely to end up measuring the wavelength
solution using multiple techniques and then piecing together solutions into
one master arc that contains solutions from different techniques.

This is what we will doing here.  We will try to get the most precise
wavelength solution for each order.  This can be critical to the telluric
modelling.

KL:  how is it for each cwave?  What the wave range of each order at each
   cwave?  I'd like to be able to say, range is blah -> this technique.

   Are all cwave in this dataset hyper sensitive to wavecal when it gets to
   telluric?




.. todo::  TBD whether we need a wavecal guide. (See :ref:`gnirsls_wavecal_guide`.)

The illumination mask will be obtained from the processed flat.  The
processed pinhole will provide the distortion correction.

::

    reduce @arcs.lis

The primitive ``determineWavelengthSolution``, used in the recipe, has an
interactive mode. To activate the interactive mode:

::

    reduce @arcs.lis -p interactive=True

The interactive tools are introduced in section :ref:`interactive`.

.. todo:: add interactive page.

Each order can be inspected individually by selecting the tabs above the plot.

The general shape of the fit for each order should look like this:

.. image:: _graphics/gnirsxd_SXD32mm_arc_order3.png
   :width: 600
   :alt: Arc line identifications and fit for Order 3

For this dataset, the fit for Order 7 is much improved by deleting ('d') the
blue-most identification. Below are the plots before and after the deletion
of the misidentified line.

.. image:: _graphics/gnirsxd_SXD32mm_arc_order7_before.png
   :width: 325
   :alt: Arc line identifications and fit for Order 7 before deletion of
         blue-most line.

.. image:: _graphics/gnirsxd_SXD32mm_arc_order7_after.png
   :width: 325
   :alt: Arc line identifications and fit for Order 7 after deletion of
         blue-most line.


Telluric Standard
=================
The telluric standard observed before the science observation is "hip 17030".
The spectral type of the star is A0V.

To properly calculate and fit a telluric model to the star, we need to know
its effective temperature.  To properly scale the sensitivity function (to
use the star as a spectrophotometric standard), we need to know the star's
magnitude.  Those are inputs to the ``fitTelluric`` primitive.

The default effective temperature of 9650 K is typical of an A0V star, which
is the most common spectral type used as a telluric standard. Different
sources give values between 9500 K and 9750 K and, for example,
Eric Mamajek's list "A Modern Mean Dwarf Stellar Color and Effective
Temperature Sequence"
(https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)
quotes the effective temperature of an A0V star as 9700 K. The precise
value has only a small effect on the derived sensitivity and even less
effect on the telluric correction, so the temperature from any reliable
source can be used. Using Simbad, we find that the star has a magnitude
of K=9.244.

Instead of typing the values on the command line, we will use a parameter file
to store them.  In a normal text file (here we name it "hip17030.param"), we write::

    -p
    fitTelluric:bbtemp=9700
    fitTelluric:magnitude='K=9.244'

Then we can call the ``reduce`` command with the parameter file.  The telluric
fitting primitive can be run in interactive mode.

Note that the data are recognized by Astrodata as normal GNIRS cross-dispersed
science spectra.  To calculate the telluric correction, we need to specify the
telluric recipe (``-r reduceTelluric``), otherwise the default science
reduction will be run.

::

    reduce @telluric.lis -r reduceTelluric @hip17030.param -p fitTelluric:interactive=True

The fit for Order 3 looks like this:

.. image:: _graphics/gnirsxd_SXD32mm_tellfit_order3.png
   :width: 600
   :alt: fit to the telluric standard

Order 8 needs some discussion.  You will notice many rejected data points marked
as light blue triangle.  The software by default rejects those points because
the stellar features in that part of the spectrum are notoriouly difficult to
model.

.. image:: _graphics/gnirsxd_SXD32mm_tellfit_order8.png
   :width: 325
   :alt: fit to the telluric standard

.. image:: _graphics/gnirsxd_SXD32mm_tellfit_order8_model.png
   :width: 325
   :alt: telluric absorption model fit

In our case, the model and the star do fit remarkably well, so we can
reactivate those points and give the software more points to fit.  On the top
plot, use the
box selection tool (the dotted line square) to include the blue triangles and
type "u" to unmask them and reactivate them.

.. image:: _graphics/gnirsxd_SXD32mm_tellfit_order8_after.png
   :width: 325
   :alt: fit to the telluric standard

.. image:: _graphics/gnirsxd_SXD32mm_tellfit_order8_model_after.png
   :width: 325
   :alt: telluric absorption model fit



Science Observations
====================
The science target is a supernova type II 54 days after explosion.  The sequence
is 5 ABBA dither pattern. DRAGONS will flatfield, wavelength calibrate,
subtract the sky, stack the aligned spectra, extract the source, and finally
remove telluric features and flux calibrate.

.. note::  In this observation, there is only one real source to extract.  If there
   were multiple sources in the slit, regardless of whether they are of
   interest to the program or not, DRAGONS will locate them, trace them, and extract
   them automatically. Each extracted spectrum is stored in an individual
   extension in the output multi-extension FITS file.

This is what one raw image looks like.

.. image:: _graphics/gnirsxd_SXD32mm_raw.png
   :width: 400
   :alt: raw science image

What you see are from left to right the cross-dispersed orders, from Order 3
to Order 8.  The short horizontal features are sky lines.  The "vertical lines"
are the dispersed science target in each order.  In the raw data, the red end
is at the bottom and blue at the top.  This will be reversed when the data is
resampled and the distortion corrected and wavelength calibration are applied.

With all the calibrations in the local calibration manager, one only needs
to call |reduce| on the science frames to get an extracted spectrum.

::

    reduce @sci.lis

To run the reduction with all the interactive tools activated, set the
``interactive`` parameter to ``True``.

::

    reduce @sci.lis -p interactive=True

At the ``skyCorrectFromSlit`` step, you will notice that the fit for Order 8
is not very good.  The row being sampled is in the middle of the image.  If
you look at the raw image, you will see that there is not much signal for
Order in the middle row.  Increase the row number (the data has been resampled
and flipped at this point) using the slider at the top-left of the tool and
you will see that when there is signal the fit is good.

.. image:: _graphics/gnirsxd_SXD32mm_skycor_order8_middle.png
   :width: 600
   :alt: skyCorrectFromSlit fit to the middle, no signal, row.

.. image:: _graphics/gnirsxd_SXD32mm_tellfit_order8_withsignal.png
   :width: 600
   :alt: skyCorrectFromSlit fit to a row with signal

When you get to the ``telluricCorrect`` step, you can experiment with the
shift between the telluric standard and the target.  Both need to be well
aligned in wavelength to optimize the correction.  In this case, we find
that a shift of 0.55 pixels significantly improves the correction.

.. image:: _graphics/gnirsxd_SXD32mm_2d.png
   :align: right
   :width: 200
   :alt: 2D spectrum

A section of 2D spectrum before extraction is shown on the right, with blue wavelengths at
the bottom and the red-end at the top.  Note that each order has been rectified
and is being stored in separate extensions in the MEF file.  Here they are
displayed together, side by side.  (``reduce -r display N20170113S0146_2D.fits``,
launch DS9 first.)

Each order is extracted separately and stored in separate extensions in the
MEF file.  The 1D extracted spectrum before telluric correction or flux
calibration, obtained by adding the option
``-p extractSpectra:write_outputs=True`` to the ``reduce`` call.  You can
plot all the orders on a common plot with ``dgsplot``. (The ``--thin`` option
simply plots a thinner line than the default width.)

::

    dgsplot N20170113S0146_extracted.fits 1 --thin

.. image:: _graphics/gnirsxd_SXD32mm_extracted.png
   :width: 450
   :alt: 1D extracted spectrum before telluric correction or flux calibration

The 1D extracted spectrum after telluric correction but before flux
calibration, obtained with ``-p telluricCorrect:write_outputs=True``, looks
like this.

.. image:: _graphics/gnirsxd_SXD32mm_tellcor.png
   :width: 600
   :alt: 1D extracted spectrum after telluric correction or before flux calibration

And the final spectrum, corrected for telluric features and flux calibrated.

::

    dgsplot N20170113S0146_1D.fits 1 --thin

.. image:: _graphics/gnirsxd_SXD32mm_1d.png
   :width: 600
   :alt: 1D extracted spectrum after telluric correction and flux calibration

In the final spectrum, the orders are remain separated.  Here they are simply
plotted one after the other on a common plot.  If you need to stitch the order,
and maybe stack the common wavelength ranges, you will have to do that with
your own software.





