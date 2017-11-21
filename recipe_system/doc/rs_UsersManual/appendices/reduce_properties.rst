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
 displayflags           <type 'bool'>        False
 files                  <type 'list'>        []
 logfile                <type 'str'>         'reduce.log'
 logmode                <type 'str'>         'standard'
 mode                   <type 'str'>         'sq'      
 recipename             <type 'str'>         None
 suffix                 <type 'str'>         None
 upload                 <type 'list'>        None
 user_cal               <type 'str'>         None
 userparam              <type 'list'>        None

Examples
--------

Setting attributes on a Reduce instance::

 >>> myreduce = Reduce()
 >>> myreduce.logfile = "my_reduction.log"
 >>> myreduce.recipe = "recipe.my_recipe"
 >>> myreduce.files = ['UVW.fits', 'XYZ.fits']

Or in other pythonic ways::

 >>> file_list = ['FOO.fits', 'BAR.fits']
 >>> myreduce.files.extend(file_list)
 >>> myreduce.files
 ['UVW.fits', 'XYZ.fits', 'FOO.fits', 'BAR.fits']

Users wishing to pass primtive parameters to the recipe_system need only set
the one attribute, ``userparam``, on the Reduce instance::

 >>> myreduce.userparam = ['clobber=True']

This is the API equivalent to the command line option::

 $ reduce -p clobber=True [...]

For muliple primitive parameters, the 'userparam' attribute is a list of 
'par=val' strings, as in::

 >>> myreduce.userparam = [ 'par1=val1', 'par2=val2', ... ]

Example function
----------------

The following function shows a potential usage of class Reduce. When 
conditions are met, the function ``reduce_conditions_met()`` is called 
passing several lists of files, ``procfiles`` (a list of lists of fits 
files). Here, each list of ``procfiles`` is then passed to the internal 
``launch_reduce()`` function.

.. code-block:: python
    :linenos:

    from gempy.utils import logutils
    from recipe_systenm.reduction.coreReduce import Reduce

    def reduce_conditions_are_met(procfiles, control_options={}):
        reduce_object = Reduce()
        reduce_object.logfile = 'my_reduce.log'
        # write logfile only, no stdout.
        reduce_object.logmode = 'quiet'
        reduce_object.userparam = ['clobber=True']
	
        logutils.config(file_name=reduce_object.logfile, mode=reduce_object.logmode)

        def launch_reduce(datasets, recipe=None, upload=None):
            reduce_object.files = datasets
            if recipe:
                reduce_object.recipename = recipe

            if upload:
                reduce_object.upload = upload

            reduce_object.mode = 'qa'  # request 'qa' recipes
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
        reduce_conditions_are_met(procfiles)

Calling ``reduce_conditions_are_met()`` without the ``control_options`` 
parameter will result in the ``mode`` attribute being set to ``'qa'``.
