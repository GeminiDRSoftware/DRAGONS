#!/bin/bash
# -*- coding: utf-8 -*-

set -eux

# Cleanup
pwd
git clean -fxd
mkdir plots reports
if [[ -n "${DRAGONS_TEST_OUT-}" ]]; then
    if [[ -d "${DRAGONS_TEST_OUT}" ]]; then
        echo "Cleaning previous test results in ${DRAGONS_TEST_OUT}"
        rm -r ${DRAGONS_TEST_OUT}
    else
        echo "Skip deletion of inexistent ${DRAGONS_TEST_OUT}"
    fi
else
    echo "DRAGONS_TEST_OUT is not set, so not deleting it"
fi
