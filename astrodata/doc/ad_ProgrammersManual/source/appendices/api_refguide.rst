.. api_refguide.rst

.. _api_refguide:

*******************
API Reference Guide
*******************

.. py:currentmodule:: astrodata.core

``AstroData``
=============

.. autoclass:: astrodata.core.AstroData
   :members:
   :special-members:

``DataProvider``
================

.. autoclass:: astrodata.core.DataProvider

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
