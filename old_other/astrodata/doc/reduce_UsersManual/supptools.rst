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

listprimitives
--------------

The application ``listprimitives`` is available as a command line executable.
This tool displays available primitives for all AstroDataTypes, their parameters, 
and defaults. These are the parameters discussed in Sec. :ref:`userpars` that 
can be changed by the user with the **-p, --param** option on reduce. under the 
AstroDataTypes. The help describes more options::

  $ listprimitives -h
  
  Usage: listprimitives [options]
  
  Gemini Observatory Primitive Inspection Tool, v1.0 2011
  
  Options:
  -h, --help            show this help message and exit
  -c, --use-color       apply color output scheme
  -i, --info            show more information
  -p, --parameters      show parameters
  -r, --recipes         list top recipes
  -s, --primitive-set   show primitive sets (Astrodata types)
  -v, --verbose         set verbose mode
  --view-recipe=VIEW_RECIPE
                        display the recipe

listprimitives information
++++++++++++++++++++++++++

The following section presents examples of the kind of information that 
``listprimitives`` may provide. 

Show available recipes::

 $ listprimitives -r

 ===============================================================================

 RECIPES_Gemini
 -------------------------------------------------------------------------------
    1. basicQA
    2. checkQA
    3. makeProcessedArc.GMOS_SPECT
    4. makeProcessedBias
    5. makeProcessedDark
    6. makeProcessedFlat
    7. makeProcessedFlat.GMOS_IMAGE
    8. makeProcessedFlat.GMOS_SPECT
    9. makeProcessedFlat.NIRI_IMAGE
    10. makeProcessedFringe
    11. qaReduce.GMOS_IMAGE
    12. qaReduce.GMOS_SPECT
    13. qaReduce.NIRI_IMAGE
    14. qaReduceAndStack.GMOS_IMAGE
    15. qaStack.GMOS_IMAGE
    16. reduce.F2_IMAGE
    17. reduce.GMOS_IMAGE

 Subrecipes
 -------------------------------------------------------------------------------
    1. biasCorrect
    2. correctWCSToReferenceCatalog
    3. darkCorrect
    4. flatCorrect
    5. lampOnLampOff
    6. makeSky
    7. overscanCorrect
    8. prepare
    9. skyCorrect
    10. standardizeHeaders
    11. thermalEmissionCorrect

 ===============================================================================

Users can also display the contents of a particular recipe or subrecipe.
This will present the sequence of primitives that will be called by the
Recipe System when the particular recipe is either specified through the 
``reduce`` command line by the user, or selected internally by the Recipe System 
itself.

For example, a user may like to see the primitive stack called by the default
'QA' recipe for GMOS_IMAGE data. As seen in the above example, these 'qa' recipes 
are defined for several AstroDataTypes. 

Show the primitive stack for the 'qa' GMOS_IMAGE type::

 $ listprimitives --view-recipe qaReduce.GMOS_IMAGE

 ===============================================================================
 RECIPE: qaReduce.GMOS_IMAGE
 ===============================================================================
 # This recipe performs the standardization and corrections needed to convert
 # the raw input science images into a single stacked science image

 prepare
 addDQ
 addVAR(read_noise=True)
 detectSources
 measureIQ(display=True)
 measureBG
 measureCCAndAstrometry
 overscanCorrect
 biasCorrect
 ADUToElectrons
 addVAR(poisson_noise=True)
 flatCorrect
 mosaicDetectors
 makeFringe
 fringeCorrect
 detectSources
 measureIQ(display=True)
 measureBG
 measureCCAndAstrometry
 addToList(purpose=forStack)


 ===============================================================================

``listprimitives`` is in need of refinement and work continues on 
building a tool that will present primitives and parameters in a more focused
way, i.e., report only those primitives and parameters relevant to a given
dataset. As it currently stands, users can request that ``listprimitives`` 
display primitive parameters (as may be passed to ``reduce`` through the
**-p** or **--param** option, Sec. :ref:`userpars`), but this results in a
list of all AstroDataTypes, their primitives and associated parameters.
Admittedly, this list is rather ungainly, but users may see, for example, that 
the primitive ``detectSources`` has several user-tunable parameters::

 detectSources
     suffix: '_sourcesDetected'
     centroid_function: 'moffat'
     threshold: 3.0
     sigma: None
     fwhm: None
     method: 'sextractor'
     max_sources: 50

See the discussion in Sec. :ref:`userpars` on command line override of
primitive parameters, and where overriding the 'threshold' parameter is dicussed
specifically.

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
