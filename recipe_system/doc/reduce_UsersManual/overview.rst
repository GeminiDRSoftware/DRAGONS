.. overview.rst
.. include glossary

.. _overview:

Overview
********

The Gemini Recipe System is a pure python package provided by the Gemini
Observatory's ``gemini_python`` data reduction package. The Recipe System is a
framework written to work with configurable data processing pipelines, i.e.,
"recipes," and which can accommodate processing pipelines for arbitrary
dataset types. The Recipe System is written to introspectively exploit features
and attributes of "instrument packages" and the defined recipes and primitives
that are to be found in is such packages. Gemini Observatory has developed a
suite of recipes and primitives that can be found in the gemini_python package,
``geminidr``.

In conjunction with the development of the Recipe System, Gemini Observatory has
also developed the new ``astrodata`` (v2.0), which works with instrument packages
defined in ``gemini_instruments``. This package provides the definitions for the
abstractions of Gemini Observatory astronomical observations. For further
information and discussion of ``astrodata`` and it's interface, see the
`Astrodata User's Manual`.

In Gemini Observatory's operational environment "on summit," ``reduce``, 
``astrodata``, and the ``gemini_instruments`` packages provide a currently 
defined, near-realtime, quality assurance pipeline, the so-called QAP. 
``reduce`` is used to launch this pipeline on newly acquired data and provide 
image quality metrics to observers, who then assess the metrics and apply 
observational decisions on telescope operations.

Users unfamiliar with terms and concepts heretofore presented should consult 
the glossary (Appendix A, :ref:`glossary`) for a definition of terms. For
greater detail and depth, users should consult the documentation cited in the
sections below.

Reference Documents
===================

  - `RecipeSystem v2.0 Design Note`, Doc. ID: PIPE-DESIGN-104_RS2.0DesignNote,
    Anderson, K.R., Gemini Observatory, 2017, DPSGdocuments/.

  - `Recipe System Programmer’s Manual`, Doc. ID: PIPE-USER-108_RSProgManual,
    Anderson, K.R., Gemini Observatory, 2017, 
    gemini_python/recipe_system/doc/rs_ProgManual/.

Related Documents
=================

  - `Astrodata cheat sheet`, Doc. ID: PIPE-USER-105_AstrodataCheatSheet,
    Cardenas, R., Gemini Observatory, 2017, astrodata/doc/ad_CheatSheet.

  - `Astrodata User’s Manual`, Doc. ID:  PIPE-USER-106_AstrodataUserManual,
    Labrie, K., Gemini Observatory, 2017, astrodata/doc/ad_UserManual/.

Schematic of the Recipe System
==============================
.. image:: RecipeSystem2.jpg
