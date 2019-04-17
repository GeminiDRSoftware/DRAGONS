#!/usr/bin/env bash

if ! [ "$(command -v conda)" ]; then
    echo "Conda is not installed - Downloading and installing"

    curl https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh \\
    --output anaconda.sh --silent

    chmod a+x anaconda.sh
    ./anaconda.sh -u -b -p $JENKINS_HOME/anaconda3/

    conda config --add channels http://ssb.stsci.edu/astroconda
    conda update --quiet conda
else
    echo "Anaconda is already installed --- Skipping step."
fi

echo ". /data/jenkins/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
echo "conda activate" >> ~/.bashrc
