#!/usr/bin/env bash

# Remove anaconda for re-installation
conda install anaconda-clean --yes
anaconda-clean --yes


LINUX_URL="https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh"
MACOS_URL="https://repo.anaconda.com/archive/Anaconda3-2019.03-MacOSX-x86_64.sh"


if [[ "$(uname)" == "Darwin" ]]; then
    ANACONDA_URL="${MACOS_URL}"
elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
    ANACONDA_URL="${LINUX_URL}"
fi


if ! [[ "$(command -v conda)" ]]; then

    echo "Conda is not installed - Downloading and installing"
    curl "${ANACONDA_URL}" --output anaconda.sh --silent

    chmod a+x anaconda.sh
    ./anaconda.sh -u -b -p $JENKINS_HOME/anaconda3/

else

  echo "Anaconda is already installed --- Skipping step."

fi

conda update --quiet conda
conda config --add channels http://ssb.stsci.edu/astroconda
conda config --add channels http://astroconda.gemini.edu/public/noarch
conda env list