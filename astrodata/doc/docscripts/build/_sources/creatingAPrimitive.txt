Creating Recipes and Primitive
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Primitives are basic transformations.  Since different dataset types will
sometimes require different concrete implementations of code to perform the
requested step, the primitive names are shared system-wide, with 
type-specific implementations. 

A "recipe" is a text file containing one primitive (or other recipe) per line.
It is thus a sequential view of a reduction or data analysis process. It
contains no branching explicitly, but since primitives can be implemented
for particular dataset types, there is implicit branching based on dataset
type.


Understanding Primitives
@@@@@@@@@@@@@@@@@@@@@@@@

Primitives are bundled together in type-specific batches. Thus, for our Sample
types of ``OBSERVED``, ``MARKED``, and ``UNMARKED``, each would have its own
primitive set.  Generally, any given dataset must have exactly one appropriate
primitive set per package, which is resolved through the ``parent`` member of
the AstroDataType. Leaf node primitive set assignments override parent
assignments.

Which primitive set is to be loaded for a given type is specified in index files.
Index files and primitive sets must appear in
``astrodata_Sample/RECIPES_Sample``, or any subdirectory of this directory.  Any
arrangement of files into subdirectories below this directory is acceptable.
However, by convention Gemini put all "primitive set" modules in the
``primitives`` subdirectory  and put only recipes in this top directory.  

The astrodata package essentially flattens these directories; moving files
around does not affect the configuration or require changing the content of any
files, with the exception that the primitive parameter file must appear in the
same location as the primitive set module itself.

Primitive Indices
#################

The astrodata package recursing a ``RECIPES_XYZ`` directory will look at each
filename, if it matches the primitive index naming convention, 
``primitivesIndex.<unique_name>.py``, it will try to load the contents of that
file and add it to the
internal primitive set index.  Below is an example of a primitive index file
which contributes to the central index::

    localPrimitiveIndex = {
        "OBSERVED":  ("primitives_OBSERVED.py", "OBSERVEDPrimitives"),
        "UNMARKED":  ("primitives_UNMARKED.py", "UNMARKEDPrimitives"),
        "MARKED"  :  ("primitives_MARKED.py", "MARKEDPrimitives"),
        }

The dictionary in the file must be named "localPrimitiveIndex". The key is the
type name and the value is a tuple containing the primitives' module basename
and  the name of the class inside the file, respectively, as strings.  These are
given as strings because they are only evaluated into Python objects if needed.

There can be multiple primitive indices. As mentioned each index file
merely updates a central index collected from all installed packages.
The index used in the end is the union of all indices.

Within the sample primitive set, ``primitives_OBSERVED.py``,
you will find something like the following::

    from astrodata.ReductionObjects import PrimitiveSet

        class OBSERVEDPrimitives(PrimitiveSet):
            astrotype = "OBSERVED"

            def init(self, rc):
                print "OBSERVEDPrimitives.init(rc)"
                return

            def typeSpecificPrimitive(self, rc):
                print "OBSERVEDPrimitives::typeSpecificPrimitive()"

            def mark(self, rc):
                for ad in rc.get_inputs_as_astrodata():
                    if ad.is_type("MARKED"):
                        print "OBSERVEDPrimitives::mark(%s) already marked" % ad.filename
                    else:
                        ad.phu_set_key_value("THEMARK", "TRUE")
                yield rc

            def unmark(self, rc):
                for ad in rc.get_inputs_as_astrodata():
                    if ad.is_type("UNMARKED"):
                        print "OBSERVEDPrimitives::unmark(%s) not marked" % ad.filename
                    else:
                        ad.phu_set_key_value("THEMARK", None)
                yield rc

Adding another primitive is merely a matter of adding another function to this
class.  No other index needs to change since it is the primitive set class
itself, not the primitives, that are registered in the index. However, note that
primitives are implemented with "generator" functions. This type of functions 
is a standard Python feature. For purposes of writing a primitive all you need
to understand about generators is that instead of a``return`` statement, you
will use ``yield``.  Like ``return`` statement the ``yield`` statement accepts a
value, and as with "returning a value" a generator "yields a value".
For primitives this value
must be the reduction context passed in to the primitive.  

A generator can have many yield statements.  The ``yield`` gives temporary
control to the infrastructure, and when the infrastructure is done processing
any outstanding duties, execution of the primitive resumes directly after the
``yield`` statement. To the primitive author it is as if the yield is a ``pass``
statement, except that the infrastructure may process requests made by the
primitive prior to the ``yield``, such as a calibration request.

Recipes
@@@@@@@

Recipes should appear in the ``RECIPES_<XYZ>`` subdirectory, and have the naming
convention ``recipe.<whatever>``. A simple recipe using the sample primitives is::

    showInputs(showTypes=True)
    mark
    typeSpecificPrimitive
    showInputs(showTypes=True)
    unmark
    typeSpecificPrimitive
    showInputs(showTypes=True)

With this file, named ``recipe.markUnmark``, in the ``RECIPIES_Sample``
directory in your test data directory you can execute this recipe with the 
``reduce`` command::

    reduce -r markUnmark test.fits
    
The ``showInputs`` primitive is a standard primitive, and the argument
``showTypes`` tells the primitive to display type information so we can see the
affect of the sample primitives. The ``typeSpecificPrimitive`` is a sample
primitive with different implementations for "MARKED" and "UNMARKED", which prints a message to demonstrate which implementation has been
executed.

