.. include overview
.. include interfaces

.. _intro:

************
Introduction
************

This document is the Recipe System Programmer's Manual, which covers version 2.0
(beta) of the DRAGONS Recipe System. This document presents detailed 
information and discussion about the programmatic interfaces on the system's 
underlying classes, ``RecipeMapper`` and ``PrimitiveMapper``. It
describes usage of the Recipe System's application programming interface (API).

Subsequent to this introduction, this document provides a high-level overview 
of the Recipe System (:ref:`Chapter 2, Overview <overview>`), which is then
followed by an introduction to the mapper classes in 
:ref:`Chapter 3, The Mappers <mapps>`. It then presents the interfaces
on the mapper classes and how to use them to retrieve the appropriate recipes 
and primitives.

It also provides instruction on how to write recipes the Recipe System
can recognise and use, how one can interactively run a recipe, step by step, 
perform discovery on the data processed, and pass adjusted parameters to the 
next step of a recipe.

Details and information about the ``astrodata`` package and/or the data processing 
involved in data reduction are beyond the scope of this document and will only be 
engaged when directly pertinent to the operations of the Recipe System.

.. _refdocs:

Reference Documents
===================

  - `RecipeSystem v2.0 Design Note`, Doc. ID: PIPE-DESIGN-104_RS2.0DesignNote,
    Anderson, K.R., Gemini Observatory, 2017, DPSGdocuments/.

  - `Reduce and Recipe System User Manual`, Doc. ID: PIPE-USER-109_RSUsersManual,
    Anderson, K.R., Gemini Observatory, 2017, 
    gemini_python/recipe_system/doc/rs_UsersManual/.

.. _related:

Related Documents
=================

  - `Astrodata cheat sheet`, Doc. ID: PIPE-USER-105_AstrodataCheatSheet,
    Cardenas, R., Gemini Observatory, 2017, astrodata/doc/ad_CheatSheet.

  - `Astrodata User’s Manual`, Doc. ID:  PIPE-USER-106_AstrodataUserManual,
    Labrie, K., Gemini Observatory, 2017, astrodata/doc/ad_UserManual/.

  - `Gemini Fits Storage System Overview` (Calibration Manager), 
    Doc. ID: `None`, Doc. file: `FitsStorageSystemOverview.docx`,
    P. Hirst, 2011, http://scisoft/svn/FitsStorage/trunk/docs/

Further Information
===================
As this document details programmatic use of the mapper classes, readers who wish
to read up on the Recipe System application, ``reduce``, should consult the 
DRAGONS document, :ref:`Reduce and Recipe System User Manual <refdocs>`, 
*Doc ID: PIPE-USER-109_RSUserManual*, which also describes usage of the 
``Reduce`` class API.

Users and developers wishing to see more information about the ``astrodata`` 
package, how to use the programmtic interfaces on such objects should consult the
documents :ref:`enumerated above <related>`.
