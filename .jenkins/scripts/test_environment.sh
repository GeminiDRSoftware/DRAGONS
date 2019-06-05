#!/usr/bin/env bash

source activate ${BUILD_TAG}

python -c "import astropy"

python -c "import numdisplay"

python -c "import astrodata"

python -c "import gemini_instruments"

python -c "import gemini_calmgr"

python -c "from gempy.library import cyclip"
