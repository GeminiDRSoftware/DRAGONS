
dataselect
----------

This command finds files that matches certain criteria defined by tags and
expression involving descriptors. You can access the basic documentation by
typing:

::

    $ dataselect --help

``dataselect`` accepts list of input files separated by space or  wildcards.

Here are some usage examples:

::

    $ dataselect raw/*.fits --tags FLAT,LAMPON

This command selects all the FITS files inside the ``raw`` directory whose tags
match ``FLAT`` and ``LAMPON``.

::

    $ dataselect *.fits --xtags CAL

This commands prints all the files in the current directory that do not have
the ``CAL`` tag (calibration files).

::

    $ dataselect --expr 'observation_class=="science"' raw/*.fits

This command selects all the files whose descriptors match the "science" value.
The ``--expr`` can be used with more than one criteria:

::

   $ dataselect --expr '(observation_class=="science" and exposure_time==60.)' *.fits

