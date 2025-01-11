.. ex2_f2im_ultradeep.rst

.. _ultradeep_example:

****************************
Example 2 - Deep observation
****************************

This is a Flamingos-2 imaging observation of a rather sparse field but with
the objective of going deep.  In most cases, the default recipe will work
fine and match the science objectives.  In some cases, often due to a crowded
field or the need to go very deep on a faint target and have a very smooth
background, there is a need to do an extra accurate sky subtraction.

The normal recipe reduces the frames that will be used for sky subtraction,
identifies sources in them and mask them before creating a stack sky frame to
use on a science image.

The ``ultradeep`` recipe will do the same, then use the stacked science image
to detect sources, mask them and then redo the sky subtraction all over again,
ensuring that even the very faint sources have been masked during the creation
of the sky frames. This recipe take a lot longer to run and uses more CPU and
memory resources.

The recipe can be run in one shot by calling the ``ultradeep`` recipe, or in
stages by calling three segments of the recipe: ``ultradeep_part1``,
``ultradeep_part2``, and ``ultradeep_part3``.  We will use the segments in
this tutorial.

.. toctree::
   :maxdepth: 1

   ex2_f2im_ultradeep_dataset
   ex2_f2im_ultradeep_cmdline
   ex2_f2im_ultradeep_api
