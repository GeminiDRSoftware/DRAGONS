.. supptools:

Supplemental tools
******************

.. _auto_mos:

automosaic
==========

The gemini_python ``mosaic`` package (``gempy.mosaic``) provides the executable
script, ``automosaic``, which provides a quick and convenient way to create mosaics 
of multi-chip devices. ``automosaic`` provides two (2) options that users can use 
to somewhat "tune" the mosaic output as they like. Help is available on the 
executable with the usual --help flag. A manual page (manpage) is also provided.

Usage::

  automosaic [-h] [-i] [-t] [-v] [infiles [infiles ...]]

  Auto mosaic builder.

  positional arguments:
  infiles        infile1 [infile2 ...]

  optional arguments:
  -h, --help     show this help message and exit
  -i, --image    Tranform image (SCI) data only.
  -t, --tile     Tile data only.
  -v, --version  show program's version number and exit

Examples
--------

1) Run mosaic on the full dataset::

     $ automosaic S20161025S0111_varAdded.fits

2) Request tiling on the dataset::

     $ automosaic -t S20161025S0111_varAdded.fits

3) Request tiling only on the science (SCI) data arrays::

     $ automosaic -t -i S20161025S0111_varAdded.fits

Users should see output that appears like::

  AutoMosaic, 2.0.0 (beta)
	Working on S20161025S0111_varAdded.fits
	AstroData object built
	Working on type: GMOS IMAGE
	Constructing MosaicAD instance ...
	Making mosaic, converting data ...
  No OBJMASK on S20161025S0111_varAdded.fits 
	Writing file ...
	Mosaic fits image written: S20161025S0111_varAdded_mosaic.fits
