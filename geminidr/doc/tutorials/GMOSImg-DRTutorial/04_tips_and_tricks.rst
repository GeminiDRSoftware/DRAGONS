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

The show_recipes_ command line takes care of both steps. Use the link below to
access its documentation:

    * `Recipe System - User's Manual: showrecipes <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#showrecipes>`_

Once you know the recipes and primitives available, you can pick one of them and
explore its parameters using the showpars_ command line. Again, check the link
below for mre details:

    * `Recipe System - User's Manual: showpars <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#showpars>`_
