Creating Recipes and Primitive
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Primitives are basic transformations.  Since different dataset types will
sometimes require different concrete implementations of code to perform the
given step, the primitive names are shared system-wide, with 
type-specific implementations. 

A "recipes" is a text file containing one primitives (or other recipe) per line.
It is thus a sequential view of a reduction or data analysis process. It
contains no branching explicitly, but since primitives can be implemented
for particular dataset types, there is implicit branching based on dataset
type. The recipe is thus a high level simplified view, designed to fit a
scientific conception of the processing required in a particular case. 
Identification of steps in a sequence allows flexibility in rearranging the
sequence and replacing finer grained steps, and thus sharing as much common
processing code by making it easier to isolate steps which truly are unique
without pulling in operations which are not unique merely because they must
occur around a type-specific operation.

The AstroData class in general provides this kind of service accessing
meta-data, and primitives serve a very similar purpose, but instead of
abstracting properties, it is transformations that are abstracted.

Understanding Primitives
@@@@@@@@@@@@@@@@@@@@@@@@@

Primitives are bundled together in type-specific batches. Thus, for our Sample
types of ``OBSERVED``, ``MARKED``, and ``UNMARKED``, each would have it's own
primitive set.  Generally, any given dataset must have exactly one appropriate
primitive set per-package, which is resolved through the ``parent`` member of
the class, leaf node primitive set assignments override parent assignments.

Which primitive set will be loaded for a given type is specified in index files.
Index files and primitive sets must appear in
``astrodata_Sample/RECIPES_Sample``, or any subdirectory of this directory.  Any
distribution of files below this directory is fine, but it's worth mentioning
that by by convention we put
"primitive sets" (which are literally classes with sets of primitive functions
meant to apply to a common AstroDataType) in the ``primitives`` subdirectory 
and put only recipes in this top directory.  As the library of recipies grows,
recipes will move to a ``recipeies`` directory, with subdirectories organized by scientific
function.

The astrodata package essentially flattens these directories so moving files
around does not affect the configuration or require changing the content of any
files, with the exception that the primitive parameter file must appear in the
same location as the primitive set module itself.

Primitives Indeces
##################

The astrodata package recursing a ``RECIPES_XYZ`` directory will look at each
filename and if it matches the primitives index naming convention, 
``primitivesIndex.<unique_name>.py``, it will try to load this and add it to the
internal primitive set index.  Below is an example of a primitive index file
which contributes to the central index::

    localPrimitiveIndex = {
        "OBSERVED":  ("primitives_OBSERVED.py", "OBSERVEDPrimitives"),
        "UNMARKED":  ("primitives_UNMARKED.py", "UNMARKEDPrimitives"),
        "MARKED"  :  ("primitives_MARKED.py", "MARKEDPrimitives"),
        }

The dictionary in the file must be named "localPrimitiveIndex". The key is the
type name and the value is a tuple containing the primitives' module basename, and
the name of the class inside the file, as strings.  These are given as string
because they are only evaluated if needed.

Note: you can have multiple primitive indeces. As mentioned each index file
merely updates a central index collected from all installed packages.
The index used in the end is the union of all indices.

Within the sample primitive set, ``primitives_OBSERVED.py``,
you will find something like the following (notice
the Sample may have changed since construction of the document)::

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
class.  No other index needs to change as the class itself is registered in the
index. However, note that these are "generator" functions. These functions are a
standard Python feature. For purposes of writing a primitive all you need to
know is that instead of a``return`` statement, you will ``yield``.  Like 
``return`` statement the ``yeild`` can yeild a value.  For primitives this value
must be the reduction context passed in to the primitive.  A generator can have
many yield statements.  The ``yield`` gives control to the infrastructure, and
when the infrastructure is done processing any outstanding duties, the primitive
resumes directly after the ``yield`` statement. To the primitive author it is as
if the yield is a ``pass`` statement, except that the infrastructure may process
requests made by the primitive prior to the ``yield``, such as a calibration
request.

Recipies
@@@@@@@@@

Recipes should appear in the ``RECIPES_<XYZ>`` subdirectory, and have the naming
convention ``recipe.<whatever>``. A simple recipe using the sample primitives is::

    showInputs(showTypes=True)
    mark
    showInputs(showTypes=True)
    unmark
    showInputs(showTypes=True)

With this file names ``recipe.markUnmark`` in the ``RECIPEIS_Sample`` directory
in your test data directory you can execute this recipe with th e``reduce``
command::

    reduce -r markUnmark test.fits
    
The ``showInputs`` primitive is a standard primitive, and the argument
``showTypes`` tells the primitive to display type information so we can see the
affect of the sample primitives.

