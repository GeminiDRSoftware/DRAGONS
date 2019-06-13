#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

python setup.py build

python setup.py install

cd gempy/library

cythonize -i cyclip.pyx

cd -
