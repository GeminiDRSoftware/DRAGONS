#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

pydocstyle --add-ignore D400,D401,D205,D105,D105 \
    astrodata \
    gemini_instruments \
    gempy \
    geminidr \
    recipe_system > 'reports/pydocstyle.log' || exit 0
