.. primitive_biasCorrect.rst

.. _primitive_biasCorrect:

***********
biasCorrect
***********
This primitive applies a bias correction to a set of one or more observed frames. The
image values of the specified bias frame(s) are subtracted from the observed frame(s),
in order to remove the unwanted bias signal arising due to the detector electronics.

*(The above text is subject to final approval).*

Implementations
***************

* :ref:`primitive_biasCorrect_core.ccd`

.. _primitive_biasCorrect_core.ccd:

Generic Implementation - core.primitive_ccd module
===================================================
.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Top description from docstring
..      Inputs and Outputs section of the docstring
..      Parameters section of the docstring
..
..    The "Inputs and Outputs" section and the "Parameters" section in the
..    docstring must be underlined with "---" the length of the title for
..    compatibility with this document.  (Actually, this document was adapted
..    to use "---" as the section indicators at this level to match what we
..    already use in the docstrings.)

.. include:: generated_doc/geminidr.core.primitives_ccd.CCD.biasCorrect_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.core.primitives_ccd.CCD.biasCorrect_param.rst

Algorithm
---------
This primitive subtracts the input bias frames from the specified input observation
frames. The variance and data quality information will be updated accordingly, if
they exist.

*(The above text is subject to final approval).*

Issues and Limitations
----------------------
The bias and obervation frame should match in terms of detector and array size used.

*(The above text is subject to final approval).*

----
