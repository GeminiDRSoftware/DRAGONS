#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

pylint --exit-zero --jobs=4 --rcfile=gempy/support_files/pylintrc \
    astrodata \
    gemini_instruments \
    gempy \
    geminidr \
    recipe_system > ./reports/pylint.log