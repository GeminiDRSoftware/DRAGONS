#!/usr/bin/env bash

conda env create --quiet --file .jenkins/conda_test_environment.yml \
    -n ${BUILD_TAG}
