


Document Purpose
----------------

This document is the user manual for the "astrodata" software, which
is provided as part of the Gemini Data Processing Suite currently
being deployed at Gemini. This system includes the AstroData data
handling class for MEFs, as well as the the Astrodata "Recipe System"
which provides automation features. Each respectively abstracts the
datasets and the dataset transformation processes. In all, the
features described relate to two installed packages, one which
contains the base Astrodata source code and is a proper python package
(meant to be "imported") and another which is not meant to be
imported, but which lived on the PYTHONPATH in order to be discovered
by the astrodata package. This latter package is the astrodata
configuration in which types and type related behaviours are defined.

The term "astrodata" in this document can refer to slightly diffrent
aspects of the astrodata system. There is "AstroData" the class, which
is distinguishable by the camel caps capitalization. There is
"astrodata" the importable python package, which includes related
classes but no Gemini specific configurations, and there is simply
"Astrodata" a loose term for the whole package, possibly including the
Gemini astrodata configuration (the "astrodata_Gemini") package.


Intended Audience
-----------------

This document is intended for users of the astrodata package in
conjunction with the "astrodata_Gemini" configuration package. Some
descriptions are kept general, but all concrete examples rely on the
"astrodata_Gemini" configuration package which is the only currently
extant configuration. In most cases, new users should begin with the
AstroData User Tutorial , which can currently be found here
http://ophiuchus.hi.gemini.edu/ADTRUNK/astrodata/doc/ADTutorial.pdf.
It provides a quick hands-on introduction to the concepts involved
using AstroData to work with Gemini datasets. In contrast this
document provides a bigger picture of the components and philosophy of
Astrodata, and includes a reference manual for the classes of primary
interest to users of the system.

