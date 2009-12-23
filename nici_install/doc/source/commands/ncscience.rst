
ncscience. Analysis of science data 
===================================

**ncscience(inputs idir='' odir='' fdir='' fsuffix='default' central=False suffix='default' clobber=False dobadpix=True pad=False logfile='')**

**Parameters**

* *inputs*
   The list of files used in **ncprepare**.  This list can a Unix wildcard pathname, e.g. * .fits, root23[2-9].fits, root??.fits or a @ list, e.g. @file.lis, where 'file.lis' is a text file with a list of FITS files, one file per line or a plain list of FITS filenames separated by commas. 

* *idir*
   Default is current directory. Directory pathname where the input files reside. 

* *odir*
   Default is current directory. Directory pathname to put the output FITS files. 

* *fdir*
   Directory name where the flats are. The files are: flats_red_<fsuffix>.fits, flats_blue_<fsuffix>.fits, dark_red_<fsuffix>.fits and dark_blue_<fsuffix>.fits. 

* *fsuffix*
   Suffix used by the Calibration files (ncmkflats). If default it will used the
   *suffix* value.

* *central*
   Default False. Use the whole frame size 1024x1024. If set to True it uses the central area (512x512). 

* *suffix*
   Dataset name. If 'default' it will take the rootname of the first element in the input list.

* *clobber*
   Default value is False. Set to True to overwrite output files when they exist.

* *dobadpix*
   Correct bad pixels the best we can.

* *pad*
   Use an extended area for interpolation

* *logfile*  (Not Yet Implemented)


**Description**

      Ncscience is a collection of python scripts to analyze the science files given in the parameter inputs and produces the following output files in the following order.

::

      cube_[red,blue].fits
          Flat fielded cube. Input for next steps. 
      medcrunch
          Median reduce of the input cube. 
      sumcrunch
          Sum reduce of the input cube. 
      cube_rotate
          Rotated cube using the parallactic angles. 
      medcrunch_rotate
          Median reduced of the cube_rotate. 
      sumcrunch_rotate
          Sum reduced of the cube_rotate. 
      cube_medfilter
          Median filtering of cube slices. This image is the initial cube 
          minus the median-smoothed image. This is sort-of-an-unsharp-mask 
          but we use a median instead of boxcar smooth. 
      medfilter_medcrunch_rotate
          Rotated median reduced of cube_medfilter. 
      cube_shift_medfilter
          Scales the two channels to a common 'speckle' size. This is 
          done using the ratio of the central wavelengths of 
          the filter bandpasses. 
      cube_sdi
          Differential imaging result. 
      sdi_medcrunch
          Median reduced of cube_sdi. 
      cube_loci_sdi
          LOCI subtraction of cube_sdi. 
      loci_sdi_medcrunch
          Median reduced of the cube_loci_sdi. 
      loci_sdi_sumcrunch
          Sum reduced of the cube_loci_sdi. 
      loci_medfilter_medcrunch
          Median reduced of cube_loci_medfilter. 
      loci_medfilter_sumcrunch
          Sum reduced of cube_loci_medfilter. 
      cube_loci_medfilter
          Median filtering cube_medfilter. 
      cube_asdi
          'Super' Loci subtraction using the red channel as a cube
           and the blue channel as an additional sample of images. 
      asdi_medcrunch
          Rotation and median reduced of the cube_asdi. 
      asdi_counter_medcrunch
          Counter rotation and median reduced of the cube_asdi. 


**Examples**

1. ncscience nS20090312S00[1-3][0-9].fits --fdir='/home/nc/data/flats' --odir='/data' --suffix='NiciTest '    (Unix mode)

   Reduce all the matching FITS files . The the flats file located in the given directory and the output files will contain the string 'NiciTest '. 

2. ncscience @ncScience.lis fdir='home/nc/data/flats' idir='/data/20090312/' odir='/data/reduced' (Pyraf mode)

