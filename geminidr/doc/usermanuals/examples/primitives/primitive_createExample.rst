.. primitive_createExample.rst

.. _primitive_createExample:

*************
createExample
*************
This primitive is a fake one and used to show the format of the primitive
rst files and the expected content.  This first paragraph (this paragraph) in
written by the author directly in this file and provides a general description
of the purpose and function of a primitive with this name.

Implementations
***************
(REMOVE this text in real doc: The Implementations section contains all the
implementations sharing the name of this primitive.)

* :ref:`primitive_createExample_core.blah`
* :ref:`primitive_createExample_instrument.instrument_blah`

.. _primitive_createExample_core.blah:

Generic Implementation - core.primitive_blah module
===================================================
(REMOVE this text in real doc: If there is an implementation in `core`,
that the one we want here.  If there isn't, just skip.)

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

.. include:: generated_doc/geminidr.core.primitives_ccd.CCD.subtractOverscan_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.core.primitives_ccd.CCD.subtractOverscan_param.rst

Algorithm
---------
This section is optional but recommended.  The author write in here any
relevant information about how the primitive accomplishes its task.  Free
form.  Use screenshots and diagram when useful.

Issues and Limitations
----------------------
This section is optional but recommended.  The author write in here any
relevant information about known issues and limitations.  Free
form.  Use screenshots and diagram when useful.

----

.. _primitive_createExample_instrument.instrument_blah:

<Instrument> <Mode> Implementation - instrument.primitives_instrument_blah module
=================================================================================
.. example:  GMOS Longslit Implementation - gmos.primitives_gmos module

(REMOVE this text in real doc: The description from the docstring for this
implementation will be shown here, the parameter defaults too.  HOWEVER,
if another implementation is re-used and only the parameter defaults are
different, it might smarter to just include the `-param.rst` file to avoid
unnecessary duplication and just refer to the appropriate implementation for
the rest of the info.)

.. include:: generated_doc/geminidr.gmos.primitives_gmos.GMOS.subtractOverscan_param.rst

Algorithm
---------
Optional, okay to refer to the section from another implementation to avoid
unnecessary duplication.

Issues and Limitations
----------------------
Optional, okay to refer to the section from another implementation to avoid
unnecessary duplication.
