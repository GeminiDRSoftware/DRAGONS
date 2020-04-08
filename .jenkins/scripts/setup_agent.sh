#!/bin/bash
# -*- coding: utf-8 -*-

set -eux

# Cleanup
git clean -fxd
mkdir plots reports
if [[ -n "${DRAGONS_TEST_OUTPUTS}" ]]; then
    echo "Cleaning previous test results in ${DRAGONS_TEST_OUTPUTS}"
    rm -r ${DRAGONS_TEST_OUTPUTS}
fi

source .jenkins/scripts/download_and_install_anaconda.sh

pip install tox tox-conda
