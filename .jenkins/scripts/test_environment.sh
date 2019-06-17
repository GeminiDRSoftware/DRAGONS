#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

python -c "import astropy"

python -c "import stsci.numdisplay"

python -c "import astrodata"

python -c "import gemini_instruments"

python -c "import gemini_calmgr"

# TODO: This is not actually working
# python -c "from gempy.library import cyclip"
