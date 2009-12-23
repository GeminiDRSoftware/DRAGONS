***********************************
Running under Unix, Pyraf or Python
***********************************

.. _section-running:


**UNIX**

::

        #!/usr/bin/env python

	#-- Define the location of the NICI module.
        #   See 'installation` for details.

	    setenv nicipath /tmp/lib/python/nici    
	    chmod +x $nicipath/nc*.py              

        # Define some handy aliases
	    alias ncqlook    $nicipath/ncqlook.py
	    alias ncprepare  $nicipath/ncprepare.py
	    alias ncmkflats  $nicipath/ncmkflats.py
	    alias ncscience  $nicipath/ncscience.py

	#-- nc commands help
	#  Each of the command have a one liner help 
        #  on the arguments.
        # 
        #  MAKE SURE ds9 is running

	    ncqlook --help

	#-- Quick look
        #   Make a directory for output data

	   mkdir /tmp/results

           ncqlook '@/tmp/lib/python/test/in.lis' --idir='/net/petrohue/dataflow' --odir='/tmp/results'

	#-- Prepare science files
        #  Make a directory to hold the prepared science data

	   mkdir /tmp/results/science        

	   ncprepare @/tmp/lib/python/test/object.lis --idir=/net/petrohue/dataflow --odir=/tmp/results/science --clobber

	#-- Generate calibration files
        #  Make a directory to hold the calibration files

	   mkdir /tmp/results/flats

	   ncmkflats  @/tmp/lib/python/test/flats.lis --idir=/net/petrohue/dataflow --odir=/tmp/results/flats --clobber

	#-- Finally run the science
	# Use as input list the output files from ncprepare

	   ncscience '/tmp/results/science/*.fits' --odir=/tmp/results --fdir=/tmp/results/flats --suffix='S20090410' --clobber



**PYRAF**

::

        
	# Before running Pyraf make sure you can load the
        # NICI module from a python shell. 
        # To achieve this the PYTHONPATH should point to
        # the installed NICI directory.

        # Start pyraf
        pyraf

	#  set nicipath=<path_to_nici_scripts>/    #  Notice the ending '/'
	set nicipath=/tmp/lib/python/nici/      # An example

	# Define the pyraf task
	task nici = nicipath$nici.cl

	# Load the nici scripts
	nici

	# Now you have the nici scripts available in your Pyraf session.
	# You can get the same result is you write ithe 3 previous commands
        # in your loginuser.cl.

	mkdir /tmp/results         # your output directory

	# Run the quick look tool
        ncqlook('@/tmp/lib/python/test/in.lis',idir='/net/petrohue/dataflow',odir='/tmp/results')

        # Generate Flats calibrations files:
        # flat_blue.fits, flat_red.fits,
        # sky_blue.fits and sky_red.fits.
        #  Make a directory to hold the calibration files

	mkdir /tmp/results/flats

	ncmkflats  @/tmp/lib/python/test/flats.lis idir=/net/petrohue/dataflow odir=/tmp/results/flats clobber=yes

        # Directory to hold the science files

	mkdir /tmp/results/science               

	# Now find the mask centers and update the headers

	ncprepare @/tmp/lib/python/test/object.lis idir=/net/petrohue/dataflow odir=/tmp/results/science clobber=yes


	# Finally run the science reduction
	lpar ncscience              # Pyraf command to see the parameters value

	ncscience '/tmp/results/science/*.fits' odir=/tmp/results fdir=/tmp/results/flats suffix='S20090410' clobber=yes


**PYTHON**

::

	#- Getting data files
	mkdir /tmp/results       # Raw data directory

	#- python setup
	ipython             # start python
	import nici as nc   # Load the nici package

	#- nc commands help, example
	help nc.ncqlook

	#- Quick look

	nc.ncqlook('@/tmp/lib/python/test/in.lis',idir='/net/petrohue/dataflow',odir='/tmp/results')

	#- Prepare science files

	mkdir /tmp/results/science

	nc.ncprepare('@/tmp/lib/python/test/object.lis',
	idir='/net/petrohue/dataflow',odir='/tmp/results/science',clobber=True)

	#- Generate calibration files
	#- Create the file list

	mkdir /tmp/results/flats

	nc.ncmkflats('@/tmp/lib/python/test/flats.lis',idir='/net/petrohue/dataflow',odir='/tmp/results/flats',clobber=True)

	#- Finally run the science

	nc.ncscience('/tmp/results/science/*.fits',odir='/tmp/results',\
	   fdir='/tmp/results/flats',suffix='S20090410',clobber=True)



