#!/usr/bin/env bash

source activate ${BUILD_TAG}

python setup.py build

python setup.py install

cd gempy/library

cythonize -i cyclip.pyx

cd -
