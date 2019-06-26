#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

python setup.py sdist --formats=gztar

