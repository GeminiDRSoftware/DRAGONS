


Document Purpose
----------------

This document provides basic user information on the design and
primary interfaces of the astrodata software, a python package which
is part of the Gemini Data Processing Suite currently being deployed
at Gemini. The astrodata package includes the AstroData data handling
class for MEFs as well as the the Astrodata "Recipe System" which
provides automation features. AstroData abstracts datasets, and the
Recipe System abstracts transformation processes. The infrastructural
software that supports these features is located in an the "astrodata"
package itself, while type-specific information, including code, used
for recognizing, data mining, or processing specific types of datasets
is implemented in a configuration package astrodata discovers.
Gemini's configuration package is "astrodata_Gemini" and depends on a
library of code presented as a stand alone python package, "gempy".

The term "astrodata" in this document can refer to three somewhat
distinct aspects of the system. There is "AstroData" the class, which
is distinguishable by the camel caps capitalization and is the core
software element of the system. There is "astrodata" the importable
python package, which from the user's point of view imports the
configurations which are available in the environment, but which
strictly speaking is only the infrustructural code. And there is
simply "Astrodata" a loose term for the whole package, possibly
including the Gemini astrodata configuration (the "astrodata_Gemini")
package.


Intended Audience
-----------------

This document is intended for users of the astrodata package in
conjunction with the "astrodata_Gemini" configuration package. While
the astrodata system is often explained in the manual in non-type-
specific language, all concrete examples rely on the
"astrodata_Gemini" configuration package which is the only currently
extant configuration except for configuration packages used for
internal Gemini projects. In most cases, new users should get started
with the AstroData User Tutorial which allows one to start using the
class immediately. The tutorial can currently be found at
http://ophiuchus.hi.gemini.edu/ADTRUNK/astrodata/doc/ADTutorial.pdf.
It provides a quick hands-on introduction to the concepts involved
using AstroData to work with Gemini datasets. This document explains
show the components in the system work and includes an API reference
manual for classes of primary interest to users of the system.

