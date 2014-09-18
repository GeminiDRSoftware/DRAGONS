.. include userenv
.. include interfaces

Supplemental tools
==================

The astrodata package provides a number of command line driven tools, two of 
which users may find helpful in executing reduce on their data. These tools can
present primitive names available, their parameters and defaults, as well
as perform data and type discovery in a directory tree.

If the user environment has been correctly configured (Sec. :ref:`config`), 
these applications will work directly. 

listprimitives
++++++++++++++

In a correct environment, the ``listprimitives.py`` module (linked as
``listprimitives``) will be available as a command line executable.
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
  -e, --engineering     show engineering recipes
  -i, --info            show more information
  -p, --parameters      show parameters
  -r, --recipes         list top recipes
  -s, --primitive-set   show primitive sets (Astrodata types)
  -v, --verbose         set verbose mode
  --view-recipe=VIEW_RECIPE
                        display the recipe

Admittedly, this tool is in need of refinement and the GDPSG is working on 
building a tool that will present primitives and parameters in a more focused
way. I.e. report only those primitives and parameters relevant to a given
dataset. 

.. _typewalk:

typewalk
++++++++

The application ``typewalk`` can help users find data that match type
criteria. As with ``listprimitives``, if users have reduce available on the 
command line, ``typewalk`` will be available, too. See the help, **-h, --help**, 
for all available options on ``typewalk``.

More generally, ``typewalk`` examines files in a directory or directory tree and 
reports the types and status values through the AstroDataType classification 
scheme. Files are selected and reported through a regular expression mask 
which, by default, finds all ".fits" and ".FITS" files. Users can change 
this mask with the **-f, --filemask** option.

By default, ``typewalk`` will recurse all subdirectories under the current
directory. Users may specify an explicit directory with the **-d, --dir** option.

A user may request that an output file is written when AstroDataType
qualifiers are passed by the **--types** option. An output file is specified
through the **-o, --out** option. Output files are formatted so they may
be passed `directly to the reduce command line` via that applications
'at-file' (@file) facility. See :ref:`atfile` or the reduce help for more on 
'at-files'.

Users may select type matching logic with the **--or** switch. By default,
qualifying logic is AND. I.e. the logic specifies that `all` types must be
present (x AND y); **--or** specifies that ANY types, enumerated with 
**--types**, may be present (x OR y). **--or** is only effective when 
**--types** is used.

For example, find all gmos images from Cerro Pachon in the top level
directory and write out the matching files, then run reduce on them
(**-n** is 'norecurse')::

  $ typewalk -n --types GEMINI_SOUTH GMOS_IMAGE --out gmos_images_south
  $ reduce @gmos_images_south

This will also report match results to stdout, colourized if requested (**-c**).
