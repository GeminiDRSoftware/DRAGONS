#!/usr/bin/env bash

echo "Checking (Ana/Mini)Conda installation at ${JENKINS_CONDA_HOME}"

mkdir -p "${TMPDIR}"  # otherwise micromamba (miniforge install) crashes

## Remove anaconda to replace with miniconda
if [ -d "${JENKINS_CONDA_HOME}/bin/anaconda" ]; then
    rm -Rf ${JENKINS_CONDA_HOME}
fi

LINUX_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
MACOS_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"

if [[ "$(uname)" == "Darwin" ]]; then
    CONDA_URL="${MACOS_URL}"
elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
    CONDA_URL="${LINUX_URL}"
fi

env

if ! [[ "$(command -v conda)" ]]; then
    echo "\n\n   Conda is not installed - Downloading and installing\n\n"
    curl -L "${CONDA_URL}" --output Miniforge3-latest.sh --silent

    chmod a+x Miniforge3-latest.sh
    ./Miniforge3-latest.sh -u -b -p ${JENKINS_CONDA_HOME}
else
  echo "\n\n   Conda is already installed --- Skipping step.\n\n"
fi

echo ${PATH}
which conda
conda clean -i -t -y  # in case of corrupt package cache from previous run
# These 2 channels need removing if testing old branches has reinstated them:
conda config --remove channels http://ssb.stsci.edu/astroconda || :
conda config --remove channels http://astroconda.gemini.edu/public/noarch || :
conda config --remove channels http://jastro.org/astroconda/public || :
conda config --add channels conda-forge
conda config --add channels http://astroconda.gemini.edu/public
conda config --set channel_priority true
conda config --set restore_free_channel false
conda update --quiet conda
conda env list
