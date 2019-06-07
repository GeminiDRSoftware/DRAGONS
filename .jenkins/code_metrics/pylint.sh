#!/usr/bin/env bash

source activate ${BUILD_TAG}

pylint --exit-zero --jobs=4 --rcfile=gempy/support_files/pylintrc \
    astrodata \
    gemini_instruments \
    gempy \
    geminidr \
    recipe_system > ./reports/pylint.log