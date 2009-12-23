.. _ncmkflats:

ncmkflats. Create calibration files
===================================

**ncmkflats(inputs idir='' odir='' sigma=6 clobber=True suffix='default' logfile='')**

    ncmkflats takes the *inputs* files containing FLATS frames with shutter open and close and creates the following calibration files:

 ::

  flats_red_<suffix>.fits     # Dark subtracted and median divided (red frames)
  flats_blue_<suffix>.fits    # Dark subtracted and median divided (blue frames)
  sky_red_<suffix>.fits       # Denoised median of skycube_red
  sky_blue_<suffix>.fits      # Denoised median of skycube_red 
  skycube_red_<suffix>.fits   # Stack of denoised red frames
  skycube_blue_<suffix>.fits  # Stack of denoised bluer frames

**Parameters**

* *inputs*
   A input list of FITS files to process. This list can a Unix wildcard pathname, e.g. * .fits, root23[2-9].fits, root??.fits or a @ list, e.g. @file.lis, where 'file.lis' is a text file with a list of FITS files, one file per line or a plain list of FITS filenames separated by commas.

* *idir*
   Default is current directory. Directory pathname where the input files reside.

* *odir*
   Default is current directory. Directory pathname to put the output FITS files. 

* *sigma*
   Default is 6. Set to Nan all pixel above this value from the median. 

* *clobber*
   Default value is False. Will overwrite the output filename if value True.

* *suffix*
   Defaul value is the first rootname of the input list. Append the value to the file rootname. This should be the same as the one from the ncscience script.

* *logfile*
   Default value is True. Not Yet Implemented.

**Examples**

1. ncmkflats \*.fits --odir='/data' (Unix command)

  All the FITS files in the current directory are read but only those with type FLAT are processed. Write the output FITS files in the directory '/data'. 

2. ncmkflats @ncFlats.lis idir='/data/20090312/' odir='/data/flats/' suffix='test_run' (Pyraf mode)

  The input flats FITS files are in the list file 'ncFlats.lis' as one 
  filename per line. You can put the full pathname of each file in which 
  case do not specified 'idir'. If only filenames are given, then 
  the script will open the FITS files in 'idir'. The output files
  are written to 'odir' pathname. The output file will have
  the suffix 'test_run' as in *flats_red_test_run.fits*. Remember that 
  in Unix mode you can get the list of this scripts' parameters by typing
  'ncmkflats -h'. 


