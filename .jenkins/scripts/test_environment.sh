#!/usr/bin/env bash

source activate ${CONDA_ENV_NAME}

python -c "import astropy"
echo "Test import astropy passed"

python -c "import stsci.numdisplay"
echo "Test import stsci.numdisplay passed"

python -c "import astrodata"
echo "Test import astrodata passed"

python -c "import gemini_instruments"
echo "Test import gemini_instruments passed"

python -c "import gemini_calmgr"
echo "Test import gemini_calmgr passed"

python -c "from gempy.library import cyclip"

chmod a+rw $DRAGONS_TEST_OUTPUTS
chmod a+rw $DRAGONS_TEST_REFS