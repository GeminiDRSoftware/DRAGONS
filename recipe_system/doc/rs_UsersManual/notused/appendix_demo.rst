.. demo:
.. include intro

*************
`reduce` demo
*************

Original demo author: Kathleen Labrie, October 2014

Setting up
----------
.. todo::
   Update here for Anaconda/astroconda.

First install Ureka, which can be obtained at http://ssb.stsci.edu/ureka/.

The second step is to install DRAGONS as described in
:ref:`Section 2 - Installation <config>`.  Please do make sure that the
command `reduce` is in your ``PATH`` and that ``PYTHONPATH`` includes the
location where the modules ``astrodata``, the ``recipe_system``, and ``gempy``
are installed.

The demo data is distributed separately.  You can find the demo data package
``gemini_python_datapkg-X1.tar.gz`` on the Gemini website where you found the
gemini_python package.  Unpack the data package somewhere convenient::

   tar xvzf gemini_python_datapkg-X1.tar.gz

In there, you will find a subdirectory named ``data_for_reduce_demo``. Those are
the data we will use here.  You will also find an empty directory called
``playground``.  This is your playground. The instructions in this demo assume
that you are running the ``reduce`` command from that directory.  There is no
requirements to run ``reduce`` from that directory, but if you want to follow
the demo to the letter, this is where you should be for all the paths to work.

Introduction to the Demo
------------------------
In this demo, we will reduce a simple dither-on-source GMOS imaging sequence.
We will first process the raw biases, and then the raw twilight flats.  We will
then use those processed files to process and stack the science observation.

Instead of the default Quality Assessment (QA) recipe that is used at the Gemini
summits, we will use another recipe that will focus on the reduction rather
than on the multiple measurements of the QA metrics used at night.  QA metrics,
here the image quality (IQ), will only be measured at the end of the reduction
rather than throughout the reduction.   Another difference between the standard
QA recipe and the demo recipe, is that the demo recipe does stack the data, while
the stacking is turned off in QA mode.

The demo recipe is essentially a Quick Look recipe.  It is NOT valid for Science
Quality.  Remember that what you are using is a QA pipeline, not a Science
pipeline.

The Recipes
-----------
To process the biases and the flats we will be using the standard recipes. The
system will be able to pick those automatically when it recognizes the input data
as GMOS biases and GMOS twilight flats.

For the science data, we will override the recipe selection to use the Demo recipe.
If we were not to override the recipe selection, the system would automatically
select the QA recipe.  The Demo recipe is more representative of a standard
Quick-Look reduction with stacking, hence probably more interesting to the reader.

The standard recipe to process GMOS biases is named ``makeProcessedBias``
and contains the instruction set::

 recipe_tags = set(['GMOS', 'CAL', 'BIAS'])
 def makeProcessedBias(p):
     """
     This recipe performs the standardization and corrections needed to convert
     the raw input bias images into a single stacked bias image. This output
     processed bias is stored on disk using storeProcessedBias and has a name
     equal to the name of the first input bias image with "_bias.fits" appended.

     Parameters
     ----------
     p : PrimitivesBASE object
         A primitive set matching the recipe_tags.

     """
     p.prepare()
     p.addDQ()
     p.addVAR(read_noise=True)
     p.overscanCorrect()
     p.addToList(purpose="forStack")
     p.getList(purpose="forStack")
     p.stackFrames()
     p.storeProcessedBias()
     return

The standard recipe to process GMOS twilight flats is named ``makeProcessedFlat``
and contains the instruction set::

 recipe_tags = set(['GMOS', 'IMAGE', 'CAL', 'FLAT'])
 def makeProcessedFlat(p):
     """
     This recipe performs the standardization and corrections needed to
     convert the raw input flat images into a single stacked and normalized
     flat image.  This output processed flat is stored on disk using
     storeProcessedFlat and has a name equal to the name of the first input
     flat image with "_flat.fits" appended.

     Parameters
     ----------
     p : PrimitivesBASE object
         A primitive set matching the recipe_tags.

     """
     p.prepare()
     p.addDQ()
     p.addVAR(read_noise=True)
     p.display()
     p.overscanCorrect()
     p.biasCorrect()
     p.ADUToElectrons()
     p.addVAR(poisson_noise=True)
     p.addToList(purpose="forStack")
     p.getList(purpose="forStack")
     p.stackFlats()
     p.normalizeFlat()
     p.storeProcessedFlat()
     return

.. todo::
   The new recipe libraries have no ``reduceDemo`` recipe.

The Demo recipe is named ``reduceDemo`` and contains the instruction set::

   # reduceDemo

   p.prepare()
   p.addDQ()
   p.addVAR(read_noise=True)
   p.overscanCorrect()
   p.biasCorrect()
   p.ADUToElectrons()
   p.addVAR(poisson_noise=True)
   p.flatCorrect()
   p.makeFringe()
   p.fringeCorrect()
   p.mosaicDetectors()
   p.detectSources()
   p.addToList(purpose=forStack)
   p.getList(purpose=forStack)
   p.alignAndStack()
   p.detectSources()
   p.measureIQ()

For the curious, the standard bias and flat recipes are found in
``???`` and the demo recipe is in
``???demos/``.  You do not really need that information
as the system will find them on its own.

The Demo
--------

Images will be displayed at times.  Therefore, start ds9::

   ds9 &


The Processed Bias
^^^^^^^^^^^^^^^^^^

The first step is to create the processed bias.  We are using the standard
recipe.  The system will recognize the inputs as GMOS biases and call the
appropriate recipe automatically.

The biases were taken on different dates around the time of the science
observations.  For convenience, we will use a file with the list of datasets
as input instead of listing all the input datasets individually.  We will use the
tool, ``typewalk``, to painlessly create the list. ::

   cd <your_path>/gemini_python_datapkg-X1/playground

   typewalk --tags GMOS BIAS --dir ../data_for_reduce_demo -o bias.list

   reduce @bias.list

This creates the processed bias, ``N20120202S0955_bias.fits``.  The output
suffix ``_bias`` is the indicator that this is a processed bias.  All processed
calibrations are cached in ``./calibrations/``.

If you wish to see what the processed bias looks like::

   reduce N20120202S0955_bias.fits -r display

.. note::
   This will issue an error about the file already existing.  Ignore it.
   The explanation of what is going on is beyond the scope of this demo.  We
   will fix this, eventually.  Remember that this is a release of software meant
   for internal use; there are still plenty of issues to be resolved.*

The Processed Flat
^^^^^^^^^^^^^^^^^^

Next we create a processed flat.  We will use the processed bias we have
just created.  The system will recognize the inputs as GMOS twilight flats and
call the appropriate recipe automatically.

The "public" RecipeSystem does not yet have a Local Calibration Server.  Therefore,
we will need to specify the processed bias we want to use on the `reduce` command
line.  For information only, internally the QA pipeline at the summit uses a
central calibration server and the most appropriate processed calibrations available
are selected and retrieved automatically.  We hope to be able to offer a "local",
end-user version of this system in the future.  For now, calibrations must be
specified on the command line.

For the flats, we do not really need a list, we can use wild cards::

   reduce ../data_for_reduce_demo/N20120123*.fits \
      --user_cal N20120202S0955_bias.fits -p clobber=True;

This creates the processed flat, ``N20120123S0123_flat.fits``.  The output suffix
``_flat`` is the indictor that this is a processed flat.  The processed flat is
also cached in ``./calibrations/``.

The ``clobber`` parameter is set to True to allow the system to overwrite the final
output.  By default, the system refuses to overwrite an output file.

If you wish to see what the processed flat looks like::

   reduce N20120123S0123_flat.fits -r display


The Science Frames
^^^^^^^^^^^^^^^^^^

We now have all the pieces required to reduce the science frames.  This time,
instead of using the standard QA recipe, we will use the Demo recipe.  Again,
we will specify the processed calibrations, bias and flat, we wish to use. ::

   reduce ../data_for_reduce_demo/N20120203S028?.fits \
      --user_cal N20120202S0955_bias.fits N20120123S0123_flat.fits \
      -r reduceDemo -p clobber=True

The demo data was obtained with the z' filter, therefore the images contain fringing.
The ``makeFringe`` and ``fringeCorrect`` primitives are filter-aware, they will do
something only when the data is from a filter that produces fringing, like the z'
filter.  The processed fringe that is created is stored with the other processed
calibrations in ``./calibrations/`` and it is named ``N20120203S0281_fringe.fits``.
The ``_fringe`` suffix indicates a processed fringe.

The last primitive in the recipe is ``measureIQ`` which is one of the QA metrics
primitives used at night by the QA pipeline.  The primitive selects stars in
the field and measures the average seeing and ellipticity.  The image it runs
on is displayed and the selected stars are circled for visual inspections.

The fully processed stacked science image is ``N20120203S0281_iqMeasured.fits``.
By default, the suffix of the final image is set by the last primitive run
on the data, in this case ``measureIQ``.

This default naming can be confusing.  If you wish to set the suffix of the
final image yourself, use ``--suffix  _myfinalsuffix``.

Clean up
^^^^^^^^

It is good practice to reset the RecipeSystem state when you are done::

   superclean --safe

Your files will stay there, only some hidden RecipeSystem directories
and files will be deleted.
