.. 04_beyond.rst

.. _reduce: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/reduce.html

.. _showpars: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#showpars

.. _show_recipes: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#show-recipes


.. _tips_and_tricks:

***************
Tips and Tricks
***************

This is a collection of tips and tricks that can be useful for reducing
different data, or to do it slightly differently from what is presented
in the example.

.. _bypassing_caldb:

Bypass automatic calibration association
========================================
We can think of two reasons why a user might want to bypass the calibration
manager and the automatic processed calibration association. The first is
to override the automatic selection, to force the use of a different processed
calibration than what the system finds. The second is if there is a problem
with the calibration manager and it is not working for some reason.

Whatever the specific situation, the following syntax can be used to bypass
the calibration manager and set the input processed calibration yourself.

.. code-block:: bash

     $ reduce @sci_images.list --user_cal processed_bias:S20001231S0001_bias.fits processed_flat:S20001231S0002_flat.fits

The list of recognized processed calibration is:

* processed_arc
* processed_bias
* processed_dark
* processed_flat
* processed_fringe


Browse Recipes and Primitives
=============================

It is also important to remember that ``reduce`` is basically a recipe with
a sequence of operations, called Primitives, and that each Primitive require
a set of parameters. When we run ``reduce`` without any extra flag, it will
run all the Primitives in our recipe using the default values. Depending on
your data/science case, you may have to try to change the parameters of one or
more Primitives.

First, you need to know what are the recipes available for a given files, then
you need to get what are Primitives living within that recipe. Finally, you need
a list of parameters that can be modified.

The show_recipes_ command line takes care of both steps. In order to list
all the recipes available for a given file, we can pass the file as an input and
the ``--all`` option. Here is an example:

..  code-block:: bash

    $ showrecipes ../playdata/N20170525S0116.fits --all

    Input file: /path_to_my_data/playdata/N20170530S0360.fits
    Input tags: {'UNPREPARED', 'GEMINI', 'GMOS', 'IMAGE', 'NORTH', 'RAW', 'SIDEREAL'}
    Recipes available for the input file:
       geminidr.gmos.recipes.sq.recipes_IMAGE::makeProcessedFringe
       geminidr.gmos.recipes.sq.recipes_IMAGE::reduce
       geminidr.gmos.recipes.qa.recipes_IMAGE::makeProcessedFringe
       geminidr.gmos.recipes.qa.recipes_IMAGE::reduce
       geminidr.gmos.recipes.qa.recipes_IMAGE::reduce_nostack
       geminidr.gmos.recipes.qa.recipes_IMAGE::stack


The output tells me that I have two recipes for the SQ (Science Quality) mode
and four recipe for the QA (Quality Assessment) mode. By default, ``reduce``
uses the SQ mode for processing the data.

The show_recipes_ command line can also display what are the Primitives that
were used within a particular Recipe. Check the example below:

.. code-block::  bash

    $ showrecipes ../playdata/N20170525S0116.fits --mode sq --recipe reduce

    Input file: /path_to_my_data/playdata/N20170530S0360.fits
    Input tags: ['RAW', 'GEMINI', 'NORTH', 'SIDEREAL', 'GMOS', 'IMAGE', 'UNPREPARED']
    Input mode: sq
    Input recipe: reduce
    Matched recipe: geminidr.gmos.recipes.sq.recipes_IMAGE::reduce
    Recipe location: /data/bquint/Repos/DRAGONS/geminidr/gmos/recipes/sq/recipes_IMAGE.py
    Recipe tags: {'IMAGE', 'GMOS'}
    Primitives used:
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
       p.adjustWCSToReference()
       p.resampleToCommonFrame()
       p.stackFrames()
       p.writeOutputs()


Now you can get the list of parameters for a given Primitive using the
showpars_ command line. Here is an example:

.. code-block:: none

    $ showpars ../playdata/N20170525S0116.fits stackFrames

    Dataset tagged as {'UNPREPARED', 'SIDEREAL', 'NORTH', 'IMAGE', 'GEMINI', 'RAW', 'GMOS'}
    Settable parameters on 'stackFrames':
    ========================================
     Name                   Current setting

    suffix               '_stack'             Filename suffix
    apply_dq             True                 Use DQ to mask bad pixels?
    statsec              None                 Section for statistics
    operation            'mean'               Averaging operation
    Allowed values:
            mean    arithmetic mean
            wtmean  variance-weighted mean
            median  median
            lmedian low-median

    reject_method        'sigclip'            Pixel rejection method
    Allowed values:
            none    no rejection
            minmax  reject highest and lowest pixels
            sigclip reject pixels based on scatter
            varclip reject pixels based on variance array

    hsigma               3.0                  High rejection threshold (sigma)
            Valid Range = [0,inf)
    lsigma               3.0                  Low rejection threshold (sigma)
            Valid Range = [0,inf)
    mclip                True                 Use median for sigma-clipping?
    max_iters            None                 Maximum number of clipping iterations
            Valid Range = [1,inf)
    nlow                 0                    Number of low pixels to reject
            Valid Range = [0,inf)
    nhigh                0                    Number of high pixels to reject
            Valid Range = [0,inf)
    memory               None                 Memory available for stacking (GB)
            Valid Range = [0.1,inf)
    separate_ext         True                 Handle extensions separately?
    scale                False                Scale images to the same intensity?
    zero                 False                Apply additive offsets to images to match intensity?


Now that we know what are is the recipe being used, what are the Primitives
it calls and what are the parameters that are set, we can finally change the
default values using the ``-p`` flag. As an example, we can change the
``reject_method`` for the stackFrames to one of its allowed values, e.g.,
``varclip``:

.. code-block:: bash

    $ reduce @list_of_science -p stackFrames:reject_method="varclip" --suffix "_stack_varclip"

The command line above changes the rejection algorithing during the stack
process. It helps with the cosmetics of the image but it might affect the
photometry if the point-spread function (seeing) changes a lot on every images
in the stack or if the images are poorly aligned. The ``--suffix`` option is
added so the final stack frame has a different name. Otherwise, reduce_
overwrites the output. Here is the product of the command line above:

.. figure:: _static/img/N20170525S0116_stack_varclip.png
   :align: center

   Sky Subtracted and Stacked Final Image. The light-gray area represents the
   masked pixels.
