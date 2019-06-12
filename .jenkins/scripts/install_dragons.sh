#!/usr/bin/env bash

conda activate ${CONDA_ENV_NAME}

python setup.py build

python setup.py install
