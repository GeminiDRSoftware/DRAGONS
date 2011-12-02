Creating A Config Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Preparation
&&&&&&&&&&&&

For this work it is required that the Gemini Python AstroData package already
be installed and
functional. In general the Gemini AstroData configuration package(s) will 
already
be installed somewhere on the PYTHONPATH, and Gemini Python scripts such 
as ``reduce`` and
``typewalk`` will be on the system path.  Installations from SVN will 
require that the ``astrodata/scripts`` package directory in the PATH.

Clone the Sample Package
&&&&&&&&&&&&&&&&&&&&&&&&&

The easiest and recommended way to start a new configuration package is by
copying the astrodata_Sample package.  Prior to this it is adviseable to have a
working Recipe System (which relies on a working AstroData system).

The sample configuration package is located in
``astrodata/sample/astrodata_Sample``, copy this to a development workspace
as in the following example, where ``<ad_install_dir>`` should be the
directory in which the ``astrodata`` package is installed::
    
   mkdir workspace
   cd workspace

   cp -r <ad_install_dir>/astrodata/samples/astrodata_Sample .
   
Note that, presumably, ``<ad_install_dir>`` should already be on the PYTHONPATH.

The name of the destination can, of course, be other than ``astrodata_Sample``,
and it can be changed later as well. 
Every configuration package, wherever on the
path, must have a unique name.

You must also ensure the the new directory *containing*
``astrodata_Sample`` is in either ADCONFIGPATH or RECIPEPATH, or alternately
for convienience (i.e. when installing packages via setup.py) in the PYTHONPATH.
If you are following the above steps, you are in the directory to which
``astrodata_Sample`` was copied.

    export RECIPEPATH=$(pwd):$RECIPEPATH
    
You can now test that ``astrodata_Sample`` is being 
discovered by the astrodata package
by running a tool from the ``astrodata/scripts`` directory, but which should
have been installed to system bin directories by the ``setup.py`` process.

I will presume you are working in a test data directory, with a subdirectory
named source_data into which you have copied at least one fits file. 
For these examples, we assume you have copied ``test.fits`` 
(and/or your actual test data) here::

    cd ~
    mkdir test_data
    cd test_data
    mkdir source_data
    cp <somepath>/test.fits source_data

We'll assume you are working in this directory for the rest of the example.
To see if the types from ``astrodata_Sample`` are discovered, type::

    typewalk -c
    
This will generate output like the following::


    directory: . (/home/dpd/test_data)
     test.fits ......................... (CAL) (GEMINI) (GEMINI_NORTH) (GMOS) 
     ................................... (GMOS_CAL) (GMOS_IMAGE) 
     ................................... (GMOS_IMAGE_FLAT) (GMOS_N) (GMOS_RAW) 
     ................................... (IMAGE) (MARKED) (OBSERVED) (RAW) 
     ................................... (UNPREPARED) 
     
A line should show up for test.fits and any other fits files in the current
directory **or** any subdirectory listing the AstroData types which apply to the
dataset. The list will contain some Gemini types, such as RAW and
UNPREPARED, and if the data in question is Gemini data, types associated with
the instrument-mode and processing status.

However, it should also include two types from the sample configuration,
``UNMARKED`` (or possibly ``MARKED`` if the
dataset has been manipulated by the Sample package previously), and ``OBSERVED``.

