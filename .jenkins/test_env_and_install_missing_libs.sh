#!/usr/bin/env bash

pip install stsci.numdisplay

which pip

which python

python --version

python -c "import future"
python -c "import astropy; print(astropy.__version__)

cd gempy/library
cythonize -a -i cyclip.pyx
cd -

python setup.py build
python setup.py install

