#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

cd .jenkins/local_calibration_manager/

pip install --quiet GeminiCalMgr-0.9.11-py3-none-any.whl

cd -
