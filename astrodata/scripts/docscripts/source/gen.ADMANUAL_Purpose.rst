


Document Purpose
----------------

This document is the reference manual for the "astrodata" software
provided as part of the python version of the Gemini Data Reduction
Package, currently being deployed at Gemini. This system includes the
AstroData data handling class for MEFs, as well as the the AstroData
"Recipe System" which provides automation features. In all, the
features described relate to four python packages, two which contain
code and two which contain configurations astrodata uses to handle
data intelligently.



Intended Audience
-----------------

This document is intended for users of the astrodata package, as well
as users of the Gemini Python Package which includes the Gemini-
specific configurations for astrodata which define the types, high
level metadata, dataset structure and primitive dataset
transformations. Readers should in most cases begin with AstroData
Tutorial, which can be found here
`http://nihal.hi.gemini.edu/LINK_TO_AD_TUTORIAL
<http://nihal.hi.gemini.edu/LINK_TO_AD_TUTORIAL>`__, for a quick
introduction to the concepts with an emphasis on how the system works
in python source code. This document gives a more complete and
detailed picture, emphasizing how the system works.

