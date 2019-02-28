#!/usr/bin/env bash

which pip

which python

python --version

python -c "import future"
python -c "import astropy; print('AstroPy v'.format(astropy.__version__))"

cd gempy/library
cythonize -a -i cyclip.pyx
cd -

cd .jenkins
pip install --quiet GeminiCalMgr-0.9.11-py3-none-any.whl
cd -

mkdir -p ${HOME}/.geminidr
cp recipe_system/cal_service/tests/rsys.cfg ${HOME}/.geminidr
cd -

