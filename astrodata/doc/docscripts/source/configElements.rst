Elements
&&&&&&&&&&&&&&&&&&&&&&&&&&

Instrument-mode-specific behaviors available through the AstroData class
are not implemented in the astrodata package itself, but are instead loaded from 
configuration packages. In the case of Gemini data the 
configuration package is a directory named ``astrodata_Gemini``.  This
configuration path is found by astrodata by the containing directory
appearing either on the PYTHONPATH, or 
on the RECIPEPATH, an astrodata environment variable.

The astrodata package searches for all directories named ``astrodata_<anything>``
in the RECIPEPATH environment variable.  Though the configurations contain
exectable python, it is not meant to be imported. However, for convienience
astrodata also searched the PYTHONPATH for astrodata configuration packages.

The General Configuration Creation Process
*******************************************

#. You will define a tree of AstroDataTypes identifying your data.
#. You will create "descriptor" functions which calculate a particular metadata
   value for nodes of the AstroDataType tree defined,
   such as ``gain`` or ``filter_name``.
#. You will write python member functions bundled into PrimitivesSet classes,
   which specifically understand your dataset.
#. You will assemble primitives into sequential lists, which we call a "recipe".

Initially you will develop classifications
for your data, and functions which will provide standard information, allowing
you to use AstroData, e.g. in processing scripts.  Then you will put your
processing scripts in the form of "primitives" and collect these in "recipes"
so they can be used for automated data reduction.


Configuration Elements Which Have To Be  Developed
***************************************************

1. **AstroData Types** identify classifications of MEF dataset to which other
   features can be assigned. Types have both requirements which must hold for
   an identified dataset, and also information about the place of the type in
   an overall type hierarchy (i.e. The GMOS type is the parent of GMOS_IMAGE).
   
2. **AstroData Descriptors** are functions which calculate a particular type
   of metadata which is expected to be available for all datasets throughout
   the type hierarchy. Examples from the Gemini configuration package are ``gain``
   and ``filtername``.  Different instruments (and thus for different AstroDataTypes)
   store information about the gain in unique headers, and may even require
   lookup tables not in the dataset to calculate.  Descriptors are type-appropriate
   functions   assigned at runtime to astrodata instance.
      
3. **Primitives** are dataset transformations meant to run in the recipe system.
   Primitives are implemented as python generator functions in sets of primitives
   that apply to a common AstroDataType.
   
4. **Recipes** are lists of primitives stored in plain text which can be executed
   by the AstroData Recipe System. While primitives work directly on the ``Reduction
   Context``, the context is implicit in recipes, so that recipes can be arguably
   "scientifically meaningful" with no "software artifacts".
