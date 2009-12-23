ncprepare. Find masks center
============================

**ncprepare(inputs oprefix='n' idir='' odir='' clobber=False fl_var=False fl_dq=False)**

    ncprepare is a Python script that takes raw NICI data with 2 FITS extensions and calculates the center of each mask -interactively if necessary, adding these values to the header. This is a require step before running **ncscience**.

**Parameters**

* *inputs*
          A input list of FITS files to process. This list can a Unix wildcard pathname, e.g. * .fits, root23[2-9].fits, root??.fits or a @ list, e.g. @file.lis, where 'file.lis' is a text file with a list of FITS files, one file per line or a plain list of FITS filenames separated by commas. 

* *oprefix*
          Default value is ' n'. Is the prefix used for the output filenames. 

* *idir*
          Default is current directory. Directory pathname where the input files reside. 

* *odir*
          Default is current directory. Directory pathname to put the output FITS files. 

* *clobber*
          Default value is False. Set to True to overwrite.

* *fl_var* (Not Yet implemented)
          Default value is False. If True it will append an extension to the output file with the variance image. 

* *fl_dq* (Not Yet implemented)
          Default value is False. If True it will append an extension to the output file with the a mask image. 

**Mask Centroid notes**

    Mask centroid is done automatically and the 2 FITS header of the output FITS file will have XCEN and YCEN keyword with its coordinates. If the finding algorithm fails then ncprepare will go into "interactive" mode using DS9 to display the frame.

1. Mark the center with left button, then use q to continue.
2. The frame is displayed again but at higher resolution. Mark again and press q to continue.

**Examples**

1. ncprepare '\*.fits' odir='/data'

   Prepare all the FITS files in the current directory, find the mask center and update the headers. Write the output files in '/data'
2. ncprepare @niciFiles.lis idir='/data/20090312/' odir='/data/reduced' clobber=yes (Pyraf mode) 
3. ncprepare @niciFiles.lis --idir='/data/20090312/' --odir='/data/reduced' --clobber (Unix mode) 

   The input FITS files are in the list file 'niciFiles.lis' as one 
   filename per line. You can put the full pathname of each file in 
   which case do not specified 'idir'. If only filenames are given, 
   then the script will open the FITS files in 'idir'. The output 
   files are written to 'odir' pathname. Remember that in Unix mode 
   you can get the list of this script by typing 'ncprepare -h'. 

