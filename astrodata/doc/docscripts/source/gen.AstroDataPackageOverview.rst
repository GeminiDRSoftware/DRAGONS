


The astrodata Package: Overview
-------------------------------

Strictly speaking what we call the astrodata package is just a single
package in the Gemini Python Package, which is itself a bundle of
packages. The astrodata package depends on three other packages, so
loosely we may also call "the astrodata package". In addition to this
subset the Gemini Package in toto includes other auxillary packages,
such as the IQTool, the original python NICI software, etc.

The "packages" of interest to the astrodata package proper functions
includes various packages.


+ astrodata
+ adutils
+ RECIPES_Gemini
+ ADCONFIG_Gemini



astrodata
~~~~~~~~~

This package contains the source code for the astrodata and recipe
systems, but no Gemini specific code. It provides the ability to
define a lexicon for your data, allowing the infrastructure enough
information to help executing dataset-type correct behavior, and
provide automation features.


adutils
~~~~~~~

This package contains modules and subpackages which are used by
various parts of the system, generally higher level scripts, such as
the color-test command line output filters in terminal.py. The
package, theoretically should not contain Gemini specific code, but
currently does. Such modules are subject to removal from adutils and
insertion into other packages (e.g. gdutils or into the astrodata
package proper).


ADCONFIG_Gemini
~~~~~~~~~~~~~~~

This package is not meant to be imported directly, but is in the
PYTHONPATH to allow astrodata to discover it. It defines the main
configurations for Gemini datasets, including:


+ AstroDataTypes: The dataset type definitions in the
  "classifications" subdirectories (split into typological types vs
  processing status types).
+ AstroData Descriptors: The descriptor implementations and
  assignments to dataset type.
+ AstroData Structures: The structure definitions.



RECIPES_Gemini
~~~~~~~~~~~~~~

This package is not meant to be imported directly, but is in the
PYTHONPATH to allow the RecipeManager module to discover it. It
defines the recipes, their datset-type assignments, and contains
implementations of the primitive sets and assignments to particular
dataset-types.

