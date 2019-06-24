#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
parsedVersion=$(echo "${version//./}")
echo "Installing missing packages for Python ${parsedVersion}"


cd .jenkins/local_calibration_manager/

if [[ "$parsedVersion" -lt "300" && "$parsedVersion" -gt "270" ]]; then
    pip install --quiet -r requirements_py2.txt
    pip install --quiet GeminiCalMgr-0.9.11-py2-none-any.whl
else
    pip install --quiet -r requirements_py3.txt
    pip install --quiet GeminiCalMgr-0.9.11-py3-none-any.whl
fi

cd -
