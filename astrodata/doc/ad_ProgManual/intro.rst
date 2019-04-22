.. intro.rst

.. _intro:

*************************
Precedents and Motivation
*************************


The Gemini Observatory has produced a number of tools for data processing.
Historically this has translated into a number of IRAF\ [#IRAF]_ packages but
the lack of long-term support for IRAF, coupled with the well-known
difficulty in creating robust reduction pipelines within the IRAF
environment, led to a decision
to adopt Python as a programming tool and a new
package was born: Gemini Python. Gemini Python provided tools to load and
manipulate Gemini-produced multi-extension FITS\ [#FITS]_ (MEF) files,
along with a pipeline that
allowed the construction of reduction recipes. At the center of this package
was the AstroData subpackage, which supported the abstraction of the FITS
files.

Gemini Python reached version 1.0.1, released during November 2014. In 2015
the Science User Support Department (SUSD) was created at Gemini, which took on the
responsibility of maintaining the software reduction tools, and started
planning future steps. With improved oversight and time and thought, it became
evident that the design of Gemini Python and, specially, of AstroData, made
further development a daunting task.

In 2016 a decision was reached to overhaul Gemini Python. While the
principles behind AstroData were sound, the coding involved unnecessary
layers of abstraction and eschewed features of the Python language in favor
of its own implementation. Thus,
DRAGONS\ [#DRAGONS]_ was born, with a new, simplified (and backward *incompatible*)
AstroData v2.0 (which we will refer to simply as AstroData)

This manual documents both the high level design and some implementation
details of AstroData, together with an explanation of how to extend the
package to work for new environments.

.. rubric:: Footnotes

.. [#IRAF] http://iraf.net
.. [#FITS] The `Flexible Image Transport System <http://https://fits.gsfc.nasa.gov/fits_standard.html>`_
.. [#DRAGONS] The `Data Reduction for Astronomy from Gemini Observatory North and South <https://github.com/GeminiDRSoftware/DRAGONS>`_ package
