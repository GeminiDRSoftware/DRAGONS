.. cal_service.rst

.. _cal_service:

**********************************
Setting up the Calibration Service
**********************************

DRAGONS comes with a local calibration manager that uses the same calibration
association rules as the Gemini Observatory Archive. This allows the command
line "|reduce|" and the API ``Reduce`` instance to make requests to a local
light-weight database for matching **processed** calibrations when needed to
reduce a dataset.

Below we show how to configure the database and then how to initialize and use
it from the command line and from the API.

If you need more details, check the "|caldb|" documentation in the Recipe
System User Manual.

.. _cal_service_config:

The Configuration File
======================

The database is configure in the DRAGONS configuration file under the
``[calibs]`` section.

In ``~/.dragons/``, create or edit the configuration file ``dragonsrc`` as
follow:

.. code-block:: none

    [calibs]
    databases = ${path_to_my_data}/gnirsls_tutorial/playground/cal_manager.db get store

The ``[calibs]`` section tells the system where to put the calibration database
and how to name it.  Here we use ``cal_manager.db`` to match what was used in
the pre-v3.1 version of DRAGONS, but you can now set the name of the database
to what suits your needs and preferences.

That database will keep track of the processed calibrations that we are going to
send to it.  With the "get" and "store" options, the database will be used
by DRAGONS to automatically *get* matching calibrations and to automatically
*store* master calibrations that you produce.  If you remove the "store" option
you will have to ``caldb add`` (command line) or ``caldb.add_cal()`` (API)
your calibration product yourself.

.. note:: The tilde (``~``) in the path above refers to your home directory.
   Also, mind the dot in ``.dragons``.

.. _cal_service_cmdline:

Usage from the Command Line
===========================

To initialize (create) the database run:

.. code-block:: bash

    caldb init

That's it. It is ready to use.  You can check the configuration and confirm the
settings with ``caldb config``.   On the command line, if the database already
exists it will refuse to act and will let you know.  If you do want to delete
the existing database and start fresh, use the "wipe" option: ``caldb init -w``.

You can manually add processed calibrations with ``caldb add <filename>``, list
the database content with ``caldb list``, and ``caldb remove <filename>`` to
remove a file from the database (it will **not** remove the file on disk.)


.. _cal_service_api:

Usage from the API
==================

From the API, the calibration database is initialized as follows:

.. code-block:: python

    from recipe_system import cal_service

    caldb = cal_service.set_local_database()
    caldb.init()


The calibration service is now ready to use.

To delete the existing database and start fresh, use the "wipe"
option: ``caldb.init(wipe=True)``.

You can manually add processed calibrations with ``caldb.add_cal(<filename>)``,
list the database content with ``caldb.list_files()``, and
``caldb.remove_cal(<filename>)`` to remove a file from the database (it will
**not** remove the file on disk.)
