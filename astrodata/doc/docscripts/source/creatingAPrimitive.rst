Creating Recipes and Primitive
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Primitives are basic transformations.  Since different dataset types will
sometimes require different concrete implementations of code to perform the
given step, the name are shared system-wide, and type-specific implementations
can be implemented.

Recipes are lists of primitives (or other recipes), using the shared names, but
the specific implementation of the primitive executed will be the one
appropriate to the given dataset, based on its AstroDataType.

Understanding Primitives
@@@@@@@@@@@@@@@@@@@@@@@@@

Primitives are bundled together in type-specific batches. Thus, for our Sample
types of ``OBSERVED``, ``MARKED``, and ``UNMARKED``, each would have it's own
primitive set.  Generally, any given dataset must have exactly one appropriate primitive
set per-package, which is resolved through the ``parent`` member of the class,
leaf node primitive set assignments override parent assignments.

Which primitive set will be loaded for a given type is specified in index files.
Index files and primitive sets must appear in
``astrodata_Sample/RECIPES_Sample``.  Any distribution of files below this
directory is fine, by convention we put primitive sets in the ``primitives``
subdirectory and put only recipes in this top directory.

Primitives Indes
##################

A primitives index is any file of the naming conventions
``primitivesIndex.<unique_name>.py``, which internally defines a dictionary as
follows::

    localPrimitiveIndex = {
        "OBSERVED":  ("primitives_OBSERVED.py", "OBSERVEDPrimitives"),
        "UNMARKED":  ("primitives_UNMARKED.py", "UNMARKEDPrimitives"),
        "MARKED"  :  ("primitives_MARKED.py", "MARKEDPrimitives"),
        }

The dictionary in the file must be named "localPrimitiveIndex". The key is the
type name, the key is a tuple containing the primitives' module basename, and
the name of the class inside the file.

Note: you can have multiple primitive indexes. The index used in the end is the
union of all indices.

Within ``primitives_OBSERVED.py`` you find something like the following (notice
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

