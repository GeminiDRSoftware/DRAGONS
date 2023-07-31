#!/bin/bash
# -*- coding: utf-8 -*-

set -eux

pwd
git clean -fxd

source .jenkins/scripts/download_and_install_anaconda.sh

conda install --yes pip wheel
pip install "tox>=3.8.1" tox-conda
