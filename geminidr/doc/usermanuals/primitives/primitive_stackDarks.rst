.. primitive_stackDarks.rst

.. _primitive_createExample:

*************
stackDarks
*************
This primitive stacks a set of user-specified dark current frames. No
scaling or offsetting is applied.

This primitive makes use of the more-general stackFrames primitive.

Implementations
***************

* :ref:`primitive_stackDarks_core.stack`

.. _primitive_stackDarks.stack:

Generic Implementation - core.primitive_stack module
====================================================
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

.. include:: generated_doc/geminidr.core.primitives_stack.Stack.stackDarks_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.core.primitives_stack.Stack.stackDarks_param.rst

Algorithm
---------
This primitive makes use of the more-general stackFrames primitive.

For the stackDarks primitive, no scaling or offsetting is applied.
By default, bad pixel masking is applied, the arithmetic mean
is used for averaging, and outlying pixel values are rejected on the
basis of their variance values.

Issues and Limitations
----------------------
All input frames must contain the 'DARK' tag, and have the same exposure times.

If the input frames are GMOS 'Nod-and-shuffle' frames, then they must also all have
the same shuffle size.
