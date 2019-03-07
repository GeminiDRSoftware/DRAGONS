.. _typewalk:

typewalk
--------
The ``typewalk`` application examines files in a directory or directory tree
and reports the data classifications through the ``astrodata`` tag sets. By
default, typewalk will recurse all subdirectories under the current
directory. Users may specify an explicit directory with the ``-d,--dir``
option.

``typewalk`` supports the following options::

  -h, --help            show this help message and exit
  -b BATCHNUM, --batch BATCHNUM
                        In shallow walk mode, number of files to process at a
                        time in the current directory. Controls behavior in
                        large data directories. Default = 100.
  -d TWDIR, --dir TWDIR
                        Walk this directory and report tags. default is cwd.
  -f FILEMASK, --filemask FILEMASK
                        Show files matching regex <FILEMASK>. Default is all
                        .fits and .FITS files.
  -n, --norecurse       Do not recurse subdirectories.
  --or                  Use OR logic on 'tags' criteria. If not specified,
                        matching logic is AND (See --tags). Eg., --or --tags
                        SOUTH GMOS IMAGE will report datasets that are one of
                        SOUTH *OR* GMOS *OR* IMAGE.
  -o OUTFILE, --out OUTFILE
                        Write reported files to this file. Effective only with
                        --tags option.
  --tags TAGS [TAGS ...]
                        Find datasets that match only these tag criteria. Eg.,
                        --tags SOUTH GMOS IMAGE will report datasets that are
                        all tagged SOUTH *and* GMOS *and* IMAGE.
  --xtags XTAGS [XTAGS ...]
                        Exclude <xtags> from reporting.

Files are selected and reported through a regular expression mask which,
by default, finds all ".fits" and ".FITS" files. Users can change this mask
with the **-f, --filemask** option.

As the **--tags** option indicates, ``typewalk`` can find and report data
that match specific tag criteria. For example, a user might want to find
all GMOS image flats under a certain directory. ``typewalk`` will locate and
report all datasets that would match the AstroData tags,
``set(['GMOS', 'IMAGE', 'FLAT'])``.

A user may request that an output file be written containing all datasets
matching AstroData tag qualifiers passed by the **--tags** option. An output
file is specified through the **-o, --out** option. Output files are
formatted so they may be passed `directly to the reduce command line` via
that applications 'at-file' (@file) facility. See :ref:`atfile` or the reduce
help for more on 'at-files'.

Users may select tag matching logic with the **--or** switch. By default,
qualifying logic is AND, i.e. the logic specifies that `all` tags must be
present (x AND y); **--or** specifies that ANY tags, enumerated with
**--tags**, may be present (x OR y). **--or** is only effective when the
**--tags** option is specified with more than one tag.

For example, find all GMOS images from Cerro Pachon in the top level
directory and write out the matching files, then run reduce on them
(**-n** is 'norecurse')::

  $ typewalk -n --tags SOUTH GMOS IMAGE --out gmos_images_south
  $ reduce @gmos_images_south

Find all F2 SPECT datasets in a directory tree::

 $ typewalk --tags SPECT F2

This will also report match results to stdout.

Users may find the **--xtags** flag useful, as it provides a facility for
filtering results further by allowing certain tags to be excluded from the
report.

For example, find GMOS, IMAGE tag sets, but exclude ACQUISITION images from
reporting::

  $ typewalk --tags GMOS IMAGE --xtags ACQUISITION

  directory: ../test_data/output
     S20131010S0105.fits .............. (GEMINI) (SOUTH) (GMOS) (IMAGE) (RAW)
     (SIDEREAL) (UNPREPARED)

     S20131010S0105_forFringe.fits .... (GEMINI) (SOUTH) (GMOS)
     (IMAGE) (NEEDSFLUXCAL) (OVERSCAN_SUBTRACTED) (OVERSCAN_TRIMMED)
     (PREPARED) (PROCESSED_SCIENCE) (SIDEREAL)

     S20131010S0105_forStack.fits ...... (GEMINI) (SOUTH) (GMOS) (IMAGE)
     (NEEDSFLUXCAL) (OVERSCAN_SUBTRACTED) (OVERSCAN_TRIMMED)
     (PREPARED) (SIDEREAL)

Exclude GMOS ACQUISITION images and GMOS IMAGE datasets that have been
'prepared'::

  $ typewalk --tags GMOS IMAGE --xtags ACQUISITION PREPARED

  directory: ../test_data/output
     S20131010S0105.fits .............. (GEMINI) (SOUTH) (GMOS) (IMAGE) (RAW)
     (SIDEREAL) (UNPREPARED)

With **--tags** and **--xtags**, users may really tune their searches for
very specific datasets.
