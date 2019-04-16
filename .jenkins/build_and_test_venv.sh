#!/usr/bin/env bash
conda env create --quiet --file .jenkins/conda_venv.yml -n ${BUILD_TAG}
conda activate ${BUILD_TAG}