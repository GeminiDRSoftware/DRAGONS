#!/usr/bin/env bash

DRAGONS_VERSION=$(python setup.py --version)
GIT_SHA=$(git log --pretty=format:'%h' -n 1)

source activate ${CONDA_ENV_NAME}

python setup.py sdist --formats=gztar --dist-dir ${DRAGONS_DIST}

# Allows other users to clean up dist directory
chmod -R 777 ${DRAGONS_DIST}

mv ${DRAGONS_DIST}/dragons-${DRAGONS_VERSION}.tar.gz \
    ${DRAGONS_DIST}/dragons-${DRAGONS_VERSION}_${BUILD_NUMBER}_${GIT_SHA}.tar.gz
