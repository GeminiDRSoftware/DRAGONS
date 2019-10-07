#!/usr/bin/env bash
#
# Uses `conda  info --envs` to list the existing virtual environments and
# checks if the `$CONDA_ENV_NAME` string is inside the output. If so, it means
# that the `$CONDA_ENV_NAME` virtual environment exists and its creating step
# can be skipped

#echo " Removing ${CONDA_ENV_NAME} ---"
#conda env remove -n "${CONDA_ENV_NAME}" || echo 0


echo " Installing ${CONDA_ENV_NAME} ---"
if conda info --envs | grep -q $CONDA_ENV_NAME; then
    echo " Skipping creating of existing conda environment: ${CONDA_ENV_NAME}";
else
    conda env create --quiet --file ${CONDA_ENV_FILE} -n "${CONDA_ENV_NAME}";
fi
