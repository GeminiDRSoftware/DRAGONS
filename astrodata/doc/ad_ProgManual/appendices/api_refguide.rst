.. api_refguide.rst

.. _api_refguide:

*******************
API Reference Guide
*******************

.. py:currentmodule:: astrodata.core

Abstract Classes
================

These classes are the top of their respective hierarchies, and need to be
fully implemented before being used. DRAGONS ships with implementations
covering the usage of Gemini-style FITS files.

``AstroData``
-------------

.. autoclass:: astrodata.core.AstroData
   :members:
   :special-members:
   :exclude-members: add, subtract, multiply, divide, __itruediv__

   .. method:: add(oper)

      Alias for ``__iadd__``

   .. method:: subtract(oper)

      Alias for ``__isub__``

   .. method:: multiply(oper)

      Alias for ``__imul__``

   .. method:: divide(oper)

      Alias for ``__itruediv__``

``DataProvider``
----------------

.. autoclass:: astrodata.core.DataProvider
   :members:
   :special-members:

``TagSet``
==========

.. autoclass:: astrodata.core.TagSet

``NDAstroData``
===============

.. autoclass:: astrodata.nddata.NDAstroData
   :members: window, set_section, data, variance

.. autoclass:: astrodata.nddata.NDWindowingAstroData

Decorators and other helper functions
=====================================

.. autofunction:: astrodata.core.astro_data_descriptor

.. autofunction:: astrodata.core.astro_data_tag

.. autofunction:: astrodata.core.returns_list
