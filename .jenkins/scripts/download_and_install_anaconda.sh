#!/usr/bin/env bash

echo "Checking (Ana/Mini)Conda installation at ${CONDA_HOME}"

## Remove anaconda to replace with miniconda
if [ -d "$CONDA_HOME/bin/anaconda" ]; then
    rm -Rf $CONDA_HOME
fi

LINUX_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MACOS_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"

if [[ "$(uname)" == "Darwin" ]]; then
    CONDA_URL="${MACOS_URL}"
elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
    CONDA_URL="${LINUX_URL}"
fi

if ! [[ "$(command -v conda)" ]]; then
    echo "\n\n   Conda is not installed - Downloading and installing\n\n"
    curl "${CONDA_URL}" --output Miniconda3-latest.sh --silent

    chmod a+x Miniconda3-latest.sh
    ./Miniconda3-latest.sh -u -b -p $CONDA_HOME
else
  echo "\n\n   Conda is already installed --- Skipping step.\n\n"
fi

conda update --quiet conda
conda config --add channels http://ssb.stsci.edu/astroconda
conda config --add channels http://astroconda.gemini.edu/public/noarch
conda config --set channel_priority false
conda config --set restore_free_channel true
conda env list
