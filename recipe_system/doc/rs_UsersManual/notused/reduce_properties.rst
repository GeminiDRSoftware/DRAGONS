.. reduce_properties:

.. _props:

************************************************
Class Reduce: Settable properties and attributes
************************************************

The public interface on an instance of the Reduce() class provides a
number of properties and attributes that allow the user to set and reset
options as they might through the reduce command line interface. The following
table is an enumerated set of those attributes.

An instance of Reduce() provides the following attributes. (Note: defaults
are not necessarily indicative of the actual type that is expected on
the instance. Use the type specified in the type column.)::

 Attribute              Python type         Default
 -------------------------------------------------------
 adinputs               <type 'list'>        None
 drpkg                  <type 'str'>         'geminidr'
 files                  <type 'list'>        []
 mode                   <type 'str'>         'sq'
 recipename             <type 'str'>         'default'
 suffix                 <type 'str'>         None
 ucals                  <type 'dict'>        None
 uparms                 <type 'dict'>        None
 upload                 <type 'list'>        None

Examples
--------

Setting attributes on a Reduce instance::

 >>> myreduce = Reduce()
 >>> myreduce.recipename = "recipe.my_recipe"
 >>> myreduce.files = ['UVW.fits', 'XYZ.fits']

Or in other pythonic ways::

 >>> file_list = ['FOO.fits', 'BAR.fits']
 >>> myreduce.files.extend(file_list)
 >>> myreduce.files
 ['UVW.fits', 'XYZ.fits', 'FOO.fits', 'BAR.fits']

Users wishing to pass primitive parameters to the recipe_system need only set
the one attribute, ``uparms``, on the Reduce instance::

 >>> myreduce.uparms = dict([('nhigh=4')])

This is the API equivalent to the command line option::

 $ reduce -p nhigh=4 [...]

For multiple primitive parameters, the 'uparms' attribute is a `dict` that
can be built from a list of 'par:val' tuples, as in::

 >>> myreduce.uparms = dict([(par1, val), (par2, val2), ... ]

Example function
----------------

The following function shows a potential usage of class Reduce. When
(unspecified) conditions are met, the function ``reduce_conditions_met()`` is
called passing several lists of files, ``procfiles`` (a list of lists of fits
files). Here, each list of ``procfiles`` is then passed to the internal
``launch_reduce()`` function.

.. code-block:: python
    :linenos:

    from gempy.utils import logutils
    from recipe_systenm.reduction.coreReduce import Reduce

    def reduce_conditions_are_met(procfiles, control_options=None):
        reduce_object = Reduce()
        reduce_object.uparms = dict([('nhigh', 4)])

        # write logfile only, no stdout.
        logutils.config(file_name='my_reduce.log', mode='quiet')

        def launch_reduce(datasets, recipe=None, upload=None):
            reduce_object.files = datasets
	    if recipe:
	        reduce_object.recipename = recipe

            if upload:
                reduce_object.upload = upload

            reduce_object.mode = 'qa'      # request 'qa' recipes
            reduce_object.runr()
            return

        for files in procfiles:
            # Use a different recipe if FOO.fits is present
            if "FOO.fits" in files:
                launch_reduce(sorted(files), recipe="recipe.FOO")
            else:
                launch_reduce(sorted(files), upload=control_options.get('upload'))

        return

    procfiles = [ ['FOO.fits', 'BAR.fits'],
                  ['UVW.fits', 'XYZ.fits']
               ]
    if conditions_are_met:
        reduce_conditions_are_met(procfiles, control_options=['metrics'])

Calling ``reduce_conditions_are_met()`` without the ``control_options``
parameter will result in the ``mode`` attribute being set to ``'qa'``.
