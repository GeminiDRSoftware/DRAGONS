.. supptools:
.. include userenv
.. include interfaces

Supplemental tools
==================

The astrodata package provides a number of command line driven tools, which 
users may find helpful in executing reduce on their data. 

With the installation and configuration of ``astrodata`` and ``reduce`` comes
some supplemental tools to help users discover information, not only about their
own data, but about the Recipe System, such as available recipes, primitives, 
and defined AstroDataTypes.

If the user environment has been configured correctly these applications 
will work directly.

.. _typewalk:

typewalk
--------

``typewalk`` examines files in a directory or directory tree and reports the types 
and status values through the AstroDataType classification scheme. Running ``typewalk`` 
on a directory containing some Gemini datasets will demonstrate what users can expect 
to see. If a user has downloaded gemini_python X1 package with the 'test_data', the 
user can move to this directory and run ``typewalk`` on that extensive set of
Gemini datasets.

By default, ``typewalk`` will recurse all subdirectories under the current
directory. Users may specify an explicit directory with the **-d** or 
**--dir** option; the behavior remains recursive.

``typewalk`` provides the following options [**-h, --help**]::

  -h, --help            show this help message and exit
  -b BATCHNUM, --batch BATCHNUM
                        In shallow walk mode, number of files to process at a
                        time in the current directory. Controls behavior in
                        large data directories. Default = 100.
  --calibrations        Show local calibrations (NOT IMPLEMENTED).
  -c, --color           Colorize display
  -d TWDIR, --dir TWDIR
                        Walk this directory and report types. default is cwd.
  -f FILEMASK, --filemask FILEMASK
                        Show files matching regex <FILEMASK>. Default is all
                        .fits and .FITS files.
  -i, --info            Show file meta information.
  --keys KEY [KEY ...]  Print keyword values for reported files.Eg., --keys
                        TELESCOP OBJECT
  -n, --norecurse       Do not recurse subdirectories.
  --or                  Use OR logic on 'types' criteria. If not specified,
                        matching logic is AND (See --types). Eg., --or --types
                        GEMINI_SOUTH GMOS_IMAGE will report datasets that are
                        either GEMINI_SOUTH *OR* GMOS_IMAGE.
  -o OUTFILE, --out OUTFILE
                        Write reported files to this file. Effective only with
                        --types option.
  --raise               Raise descriptor exceptions.
  --types TYPES [TYPES ...]
                        Find datasets that match only these type criteria.
                        Eg., --types GEMINI_SOUTH GMOS_IMAGE will report
                        datasets that are both GEMINI_SOUTH *and* GMOS_IMAGE.
  --status              Report data processing status only.
  --typology            Report data typologies only.
  --xtypes XTYPES [[XTYPES ...]
                        Exclude <xtypes> from reporting.

Files are selected and reported through a regular expression mask which, 
by default, finds all ".fits" and ".FITS" files. Users can change this mask 
with the **-f, --filemask** option.

As the **--types** option indicates, ``typewalk`` can find and report data that 
match specific type criteria. For example, a user might want to find all GMOS 
image flats under a certain directory. ``typewalk`` will locate and report all 
datasets that would match the AstroDataType, GMOS_IMAGE_FLAT.

A user may request that a file be written containing all datasets 
matching AstroDataType qualifiers passed by the **--types** option. An output 
file is specified through the **-o, --out** option. Output files are formatted 
so they may be passed `directly to the reduce command line` via that applications 
'at-file' (@file) facility. See :ref:`atfile` or the reduce help for more on 
'at-files'.

Users may select type matching logic with the **--or** switch. By default,
qualifying logic is AND, i.e. the logic specifies that `all` types must be
present (x AND y); **--or** specifies that ANY types, enumerated with 
**--types**, may be present (x OR y). **--or** is only effective when the 
**--types** option is specified with more than one type.

For example, find all GMOS images from Cerro Pachon in the top level
directory and write out the matching files, then run reduce on them
(**-n** is 'norecurse')::

  $ typewalk -n --types GEMINI_SOUTH GMOS_IMAGE --out gmos_images_south
  $ reduce @gmos_images_south

Find all F2_SPECT and GMOS_SPECT datasets in a directory tree::

 $ typewalk --or --types GMOS_SPECT F2_SPECT

This will also report match results to stdout, colourized if requested (**-c**).

Users may find the **--xtypes** flag useful, as it provides a facility for
filtering results further by allowing certain types to be excluded from the
report. 

For example, find GMOS_IMAGE types, but exclude ACQUISITION images from reporting::

  $ typewalk --types GMOS_IMAGE --xtypes ACQUISITION

  directory: ../test_data/output
     S20131010S0105.fits ............... (GEMINI) (GEMINI_SOUTH) (GMOS) (GMOS_IMAGE) 
     (GMOS_RAW) (GMOS_S) (IMAGE) (RAW) (SIDEREAL) (UNPREPARED)

     S20131010S0105_forFringe.fits ..... (GEMINI) (GEMINI_SOUTH) (GMOS) (GMOS_IMAGE) 
     (GMOS_S) (IMAGE) (NEEDSFLUXCAL) (OVERSCAN_SUBTRACTED) (OVERSCAN_TRIMMED) 
     (PREPARED) (SIDEREAL)

     S20131010S0105_forStack.fits ...... (GEMINI) (GEMINI_SOUTH) (GMOS) (GMOS_IMAGE) 
     (GMOS_S) (IMAGE) (NEEDSFLUXCAL) (OVERSCAN_SUBTRACTED) (OVERSCAN_TRIMMED) 
     (PREPARED) (SIDEREAL)

Exclude GMOS_IMAGE ACQUISITION images and GMOS_IMAGE datasets that have been 'prepared'::

  $ typewalk --types GMOS_IMAGE --xtypes ACQUISITION PREPARED

  directory: ../test_data/output
     S20131010S0105.fits ............... (GEMINI) (GEMINI_SOUTH) (GMOS) (GMOS_IMAGE) 
     (GMOS_RAW) (GMOS_S) (IMAGE) (RAW) (SIDEREAL) (UNPREPARED)

With **--types** and **--xtypes**, users may really tune their searches for very
specific datasets.
