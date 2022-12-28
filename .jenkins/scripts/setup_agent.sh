#!/bin/bash
# -*- coding: utf-8 -*-

set -eux

# Cleanup
git clean -fxd
mkdir plots reports
if [[ -n "${DRAGONS_TEST_OUTPUTS-}" ]]; then
    if [[ -d "${DRAGONS_TEST_OUTPUTS}" ]]; then
        echo "Cleaning previous test results in ${DRAGONS_TEST_OUTPUTS}"
        rm -r ${DRAGONS_TEST_OUTPUTS}
    else
        echo "Skip delete unexisting ${DRAGONS_TEST_OUTPUTS}"
    fi
fi

source .jenkins/scripts/download_and_install_anaconda.sh

conda install --yes pip wheel
pip install "tox>=3.8.1" tox-conda
