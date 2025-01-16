.. dataselect.rst

.. _dataselect:

dataselect
==========
The tool ``dataselect`` will help with the bookkeeping and with creating lists
of input files to feed to the Recipe System.  The tool has a command line
script and an API. This tool finds files that match certain criteria defined
with AstroData Tags and expressions involving AstroData Descriptors.

You can access the basic documentation from the command line by typing:

::

    $ dataselect --help

    usage: dataselect [-h] [--tags TAGS] [--xtags XTAGS] [--expr EXPRESSION] [--strict]
                      [--output OUTPUT] [--adpkg ADPKG] [--verbose] [--debug]
                      inputs [inputs ...]

    Find files that matches certain criteria defined by tags and expression involving
    descriptors.

    positional arguments:
      inputs                Input FITS file

    optional arguments:
      -h, --help            show this help message and exit
      --tags TAGS, -t TAGS  Comma-separated list of required tags.
      --xtags XTAGS         Comma-separated list of tags to exclude
      --expr EXPRESSION     Expression to apply to descriptors (and tags)
      --strict              Toggle on strict expression matching for exposure_time (not
                            just close) and for filter_name (match component number).
      --output OUTPUT, -o OUTPUT
                            Name of the output file
      --adpkg ADPKG         Name of the astrodata instrument package to useif not
                            gemini_instruments
      --verbose, -v         Toggle verbose mode when using -o
      --debug               Toggle debug mode

``dataselect`` Command Line Tool
--------------------------------

``dataselect`` accepts list of input files separated by space, and wildcards.
Below are some usage examples.

1. This command selects all the FITS files inside the ``raw`` directory with a
   tag that matches ``DARK``.

   ::

    $ dataselect raw/*.fits --tags DARK

2. To select darks of a specific exposure time:

   ::

    $ dataselect raw/*.fits --tags DARK --expr='exposure_time==20'

3. To send that list to a file that can be used later:

   ::

    $ dataselect raw/*.fits --tags DARK --expr='exposure_time==20' -o dark20s.lis


4. This commands prints all the files in the current directory that *do not*
   have the ``CAL`` tag (calibration files).

   ::

    $ dataselect raw/*.fits --xtags CAL

5. The ``xtags`` can be used with ``tags``.  To select images that are not
   flats:

   ::

    $ dataselect raw/*.fits --tags IMAGE --xtags FLAT

6. This command selects all the files with a specific target name:

   ::

    $ dataselect --expr 'object=="FS 17"' raw/*.fits

7. This command selects all the files with an "observation_class" descriptor
   that matches the "science" value and a specific exposure time:

   ::

    $ dataselect --expr '(observation_class=="science" and exposure_time==60.)' raw/*.fits



``dataselect`` API
------------------

The same selections presented in the command line section above can be done
from the ``dataselect`` API.  Here is the API versions of the examples
presented in the previous sections.

The list of files on disk must first be obtained with Python's ``glob`` module.

::

    >>> import glob
    >>> all_files = glob.glob('raw/*.fits')

The ``dataselect`` module is located in ``gempy.adlibrary`` and must first be
imported::

    >>> from gempy.adlibrary import dataselect

1. This command selects all the FITS files inside the ``raw`` directory with a
   tag that matches ``DARK``.

   ::

    >>> all_darks = dataselect.select_data(all_files, ['DARK'])


2. To select darks of a specific exposure time:

   ::

    >>> expression = 'exposure_time==20'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> darks20 = dataselect.select_data(all_files, ['DARK'], [], parsed_expr)


3. To send that list to a file that can be used later:

   ::

    >>> expression = 'exposure_time==20'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> darks20 = dataselect.select_data(all_files, ['DARK'], [], parsed_expr)
    >>> with open('dark20s.lis', 'w') as f:
    ...     for filename in dark20:
    ...         f.write(filename + '\n')
    ...
    >>>

   Note that the need to send a list of a file on disk will probably not be
   very common when using the API as ``Reduce`` will take the Python list
   directly.

4. This commands prints all the files in the current directory that *do not*
   have the ``CAL`` tag (calibration files).

   ::

    >>> non_cals = dataselect.select_data(all_files, [], ['CAL'])


5. The ``xtags`` can be used with ``tags``.  To select images that are not
   flats:

   ::

    >>> has_tags = ['IMAGE']
    >>> has_not_tags = ['FLAT']
    >>> non_flat_images = dataselect.select_data(all_files, has_tags, has_not_tags)


6. This command selects all the files with a specific target name:

   ::

    >>> expression = 'object="FS 17"'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> stds = dataselect.select_data(all_files, expression=parsed_expr)


7. This command selects all the files with an "observation_class" descriptor that
   matches the "science" value and a specific exposure time:

   ::

    >>> expression = '(observation_class=="science" and exposure_time==60.)'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> sci60 = dataselect.select_data(all_files, expression=parsed_expr)



The ``strict`` Flag
-------------------

The ``strict`` flag applies to the descriptors ``central_wavelength``,
``detector_name``, ``disperser``, ``exposure_time()``, ``filter_name()``.
To keep the user interface more friendly, in the expressions, the exposure
time and central wavelength are matched on a "close enough" principle and
the filter name, disperser and detector name are matched on the
"pretty name" principle.

For example, if the exposure time in the header is 10.001 second, from a user's
perspective, asking to match "10" seconds is a lot nicer, ``exposure_time==10``.
Similarly, asking for the "H"-band filter is more natural than asking for the
"H_G0203" filter.

However, there might be cases where the exposure time or the filter name must
be matched *exactly*.  In such case, the ``strict`` flag should be activated.
For example::

    $ dataselect raw/*.fits --strict --expr='exposure_time==0.95'

And::

    >>> expression = 'exposure_time==0.95'
    >>> parsed_expr = dataselect.expr_parser(expression, strict=True)
    >>> filelist = dataselect.select_data(all_files, expression=parsed_expr)
