ncqlook. Quick look and data quality assesment. 
===============================================

**ncqlook(inputs idir='' odir='' log=True lists=True saturate=None display=True crop=False)**

The ncqlook script produces a quick look analysis of the nici raw files specified in the 'inputs' parameter. It will produce as output a cube of FITS files (512x256xnumber_of_files) in the working directory plus several output files if the 'lists' parameter is kept True. See Parameters for more detail. While is running, each pair of frames are displayed on a ds9 panel.

**Parameters**

* **inputs**
          If left blank then last night NICI raw files resident in the Gemini South repository /net/petrohue/dataflow will be processed. If you want to display data from the repository from a different date, then the format is of the form YYYYMMDD. You can also give a list of files or a unix wild card. See examples. 

* **idir**
          The input directory where the input fits files are located. If left blank, inputs should included directory pathname. 

* **odir**
          The output directory where all the listing and fits files cube will be written. If left blank, they will written in the working directory. 

* **log**
          The default value is True. Will create a log with filename, min-max rms, median value for extension 1 and 2 and values for keywords OBJECT,OBSCLASS,OBSTYPE,MODE,ITIME,NCOADD. 

* **lists**
          The default value is True. Will create several output files useful 

* **saturate**
          Saturation limit. Default value is 5000. To change give a value.

* **display**
          Default value True for displaying current frames and False otherwise. 

* **crop**
          Default value False. If True it will crop a 256x256 area around the mask center.
          If no mask is present or if value is the default, the frame is just rebinned.

**Output files**

* *root_cube.fits*
     FITS file cube.
* *root.log*
    For each frame it contains min-max, and median listing. The values ADI,SDI and 
    ASDI are computed from keywords CRMODE and DICHROIC. The last 4 fields in the 
    log are Exposure time, Ncoads, Core2Halo ratio for red and blue frames.
* *root.1_flats*
    Contains calibration files for the ADI mode 
* *root.2_flats*
   Contains calibration files for the ASDI and SDI mode. 
* *root.(adi,sdi,asdi)*
    Contains science object listings. NOTE that these files can have listings of more than one object. You will need to edit these files and create one list per object if you want to use them in ncprepare and ncscience the log file has the necessary information for this. 

**Examples** 
 

1. ncqlook 

    Will do a quick analysis of all the NICI FITS files residing in /net/petrohue/dataflow for the date of last night, displaying each pair of frames on a ds9 frame while a listing of the log file runs on your screen. 

2. ncqlook 20090313 --odir='/tmp' --saturate=5000 --nodisplay

    (Unix command mode)

    List all the NICI fits files from /net/petrohue/dataflow/S20090313S*.fits The output listing will be written in the '/tmp' directory. No display is produced, so ds9 need not be running.

    The output files are:

        * 200903013_cube.fits
        * 200903013.log
        * 200903013.1_flats
        * 200903013.2_flats
        * 200903013.adi
        * 200903013.sdi
        * 200903013.asdi 

3. ncqlook(20090313,odir='/tmp',saturate=5000,display=False)

    This is the syntax for the command in the PYTHON shell. 

4. ncqlook "/data/nici/200903/S2009*.fits" --odir='/tmp' --crop

    Check all the fits files in the given directory writing the listing
    and cube in the '/tmp' directory. '--crop' is the flag to tell
    ncqlook to crop the data aroubd the mask center.

