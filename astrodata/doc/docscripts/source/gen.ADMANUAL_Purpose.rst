


Document Purpose
----------------

This document is the user manual for the "astrodata" software, which
is provided as part of the Gemini Data Processing Suite currently
being deployed at Gemini. This system includes the AstroData data
handling class for MEFs, as well as the the Astrodata "Recipe System"
which provides automation features. In all, the features described
relate to three installed packages, one which contains the base
Astrodata source code and two which contain configurations (including
code) used to handle data intelligently based on the type of dataset
it represents.

The term "astrodata" is used in different related sense, generally
distinguished in writing by the capitalization, but sometimes perhaps
depending on context. There is "AstroData" the class, which presents
itself as an I/O class (given a filename it loads the dataset and
returns an object representing it, the AstroData "instance"). There is
"astrodata" the package, which includes related classes but no Gemini
specific configurations, and there is simply "Astrodata" a loose term
for the whole package and possibly refering to astrodata plus its
Gemini configuration.


Intended Audience
-----------------

This document is intended for users of the astrodata package in
general, given any configuration of datatypes. Particular examples
refer to the astrodata_Gemini configuration, the only currently extant
configuration, and use Gemini definitions for dataset types, high
level meta-data names, and definitions of primitive dataset
transformations. In most cases, users should begin with the AstroData
User Tutorial, which can currently be found here
`http://ophiuchus.hi.gemini.edu/ADTRUNK/astrodata/doc/ADTutorial.pdf <
http://ophiuchus.hi.gemini.edu/ADTRUNK/astrodata/doc/ADTutorial.pdf>`_
_. It provides a quick hands-on introduction to the concepts involved
using AstroData to work with Gemini datasets. In contrast, this
document gives a more complete and detailed picture, emphasizing how
the system works.

