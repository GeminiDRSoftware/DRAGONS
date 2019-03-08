
.. _show_recipes:

show_recipes
------------

.. todo:: write show_recipes documentation

The show_recipes_ command line displays what are the recipes available for a
given file. Here is an example::

    $ show_recipes raw/S20170505S0073.fits

     DRAGONS v2.1.x - show_recipes
     Input file: ./raw/S20170505S0073.fits
     Input tags: (AT_ZENITH) (AZEL_TARGET) (CAL) (DARK) (GEMINI) (GSAOI)
                 (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
     Recipes available for the input file:
       geminidr.gsaoi.recipes.qa.recipes_FLAT_IMAGE::makeProcessedFlat
       geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedFlat
       geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedBPM

The output tells me that I have two recipes for the SQ (Science Quality) mode
and one recipe for the QA (Quality Assesment) mode. By default, ``reduce`` uses
the SQ mode for processing the data.

As the other commands, you can use the ``--help`` or ``-h`` flags in the command
line to display the help message.
