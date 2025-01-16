.. intro.rst

.. include:: references.txt

.. _intro:

************
Introduction
************

This document is the Programmer Manual for DRAGONS Recipe System. It presents
detailed information and discussion about the programmatic interfaces on the
system's underlying classes, |RecipeMapper| and |PrimitiveMapper|. It
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

Details and information about Astrodata and/or the data processing involved in
data reduction are beyond the scope of this document and will only be
engaged when directly pertinent to the operations of the Recipe System.

.. _refdocs:

Reference Documents
===================

    - |RSUser|


.. _related:

Related Documents
=================

  - |astrodatauser|
  - |astrodataprog|


Further Information
===================
As this document details programmatic use of the mapper classes, readers who wish
to read up on the Recipe System application, |reduce|, should instead consult the
DRAGONS document, |RSUser|,
which also describes usage of the |Reduce| class API from a user's point of view
rather than a programmer's.

Users and developers wishing to see more information about the Astrodata
package, how to use the programmatic interfaces on such objects should consult the
documents :ref:`listed above <related>`.
