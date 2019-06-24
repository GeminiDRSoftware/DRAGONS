#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

# test python build
python setup.py build

# compile cython libraries
python setup.py build_ext --inplace

# install dragons in current env
python setup.py install
