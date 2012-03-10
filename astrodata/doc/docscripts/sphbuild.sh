#!/bin/bash

# NOTE: DEPENDENCIES
#
# Note, a python interpreter has to be used that can load astrodata AND sphinx
# on my development system I have done this by updating standard python
# installations with sphinx because this allows package management in Ubuntu for lots of 
# dependencies. Astrodata requires only pyfits.  So I have created a python2.6
# package system, because the shared "astro" versions are all python2.5 and
# caused segfaults.


# export PYTHONPATH=/home/callen/ad26/trunk:/home/callen/pymodules/lib/python2.6/site-packages
echo "building html, see html.build.log"
sphinx-build -b html source build &>html.build.log

echo "building latex, see latex.build.log"

sphinx-build -b latex source build/_latex_build &>latex.build.log

echo "making pdf in build/_latex_build, see makepdf.build.log"
cd build/_latex_build 

# make the astrodatadocumentation
rm astrodatadocumentation.pdf
make astrodatadocumentation.pdf &> makepdf.build.log
cp astrodatadocumentation.pdf ../AstroDataDeveloperManual.pdf

# make AD type reference
# slow... commented out
rm ADtypereference.pdf
make ADtypereference.pdf &> makepdf.build.log
