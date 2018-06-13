.. intro.rst

.. _intro:

*************************
Precedents and Motivation
*************************


The Gemini Observatory has produced a number of tools for data processing.
Historically this has translated into a number of IRAF\ [#IRAF]_ packages, and
eventually it was decided to adopt Python as a programming tool and a new
package was born: Gemini Python. Gemini Python provided tools to load and
manipulate Gemini-produced FITS\ [#FITS]_ files, along with a pipeline that
allowed the construction of reduction recipes. At the center of this package
was the AstroData subpackage, which supported the abstraction of the FITS
files.

Gemini Python reached version 1.0.1, released during November 2014. 2015
brought to Gemini the Science User Support department, which took on the
responsibility of maintaining the software reduction tools, and started
planning the new step. With time, thought, it was made evident that the design
of Gemini Python and, specially, of AstroData, made further development a
daunting task.

During 2016, an effort to clean up, debug, and improve Gemini Python kicked
off, and soon enough it was decided that it needed to be rehauled. Upon
inspection of the system it was decide to keep the best ideas, to be
implemented of a whole new design, starting from scratch. Thus,
DRAGONS\ [#DRAGONS]_ was born, with a new, simplified (and backwards *incompatible*)
AstroData v2.0 (which we will refer to simply as AstroData)

This manual documents both the high level design and some implementation
details of AstroData, along with an explanation of how to extend the package to
work for new environments.

.. rubric:: footnotes

.. [#IRAF] http://iraf.net
.. [#FITS] The `Flexible Image Transport System <http://https://fits.gsfc.nasa.gov/fits_standard.html>`_
.. [#DRAGONS] The `Data Reduction for Astronomy from Gemini Observatory North and South <https://github.com/GeminiDRSoftware/DRAGONS>`_ package
