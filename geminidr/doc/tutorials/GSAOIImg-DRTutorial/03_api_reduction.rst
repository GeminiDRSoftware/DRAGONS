.. 03_api_reduction.rst

.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#caldb


.. _api_data_reduction:

Reduction using API
*******************

There may be cases where you might be interested in accessing the DRAGONS'
Application Program Interface (API) directly instead of using the command
line wrappers to reduce your data. In this case, you will need to access
DRAGONS' tools by importing the appropriate modules and packages.


Importing Libraries
-------------------

Here are all the packages and modules that you will have to import for running
this tutorial::

    >>> import glob
    >>> import os
    >>>
    >>> from gempy.adlibrary import dataselect
    >>> from recipe_system import cal_service
    >>> from recipe_system.reduction.coreReduce import Reduce


The first two packages, :mod:`glob` and :mod:`os`, are Python built-in packages.
Here, :mod:`os` will be used to perform operations with the files names and
:mod:`glob` will be used to return a list with the input file names.

.. todo: check references

Then, we are importing the :mod:`~gempy.adlibrary.dataselect` from the
:mod:`gempy.adlibrary`. It will be used to select the data in the same way we
did as in :ref:`create_file_lists` section. The
:mod:`~recipe_system.cal_service` package will be our interface with the
local calibration database. Finally, the
:class:`~recipe_system.reduction.coreReduce.Reduce` class will be
used to actually run the data reduction pipeline.


The Calibration Service
-----------------------

Before we start, let's be sure we have properly setup our database. First
create the `rsys.cfg` file as described in
`the caldb documentation in the Recipe System User's Manual <caldb>`_. Then,
you can use the following commands to configure the local database and
initialize it::

    >>> calibration_service = cal_service.CalibrationService()
    >>> calibration_service.config()
    >>> calibration_service.init(wipe=True)
    >>>
    >>> cal_service.set_calservice()

Process DARK files
------------------



Create BPM files
----------------


Process FLAT files
------------------


Process Science files
---------------------


Stack Science reduced images
----------------------------

