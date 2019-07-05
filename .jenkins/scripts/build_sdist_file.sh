#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

python setup.py sdist --formats=gztar

DRAGONS_VERSION=$(python setup.py --version)
GIT_SHA=$(git log --pretty=format:'%h' -n 1)

mv dist/dragons-${DRAGONS_VERSION}.tar.gz \
    dist/dragons-${DRAGONS_VERSION}_${BUILD_NUMBER}_${GIT_SHA}.tar.gz
