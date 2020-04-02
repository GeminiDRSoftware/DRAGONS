#!/bin/bash
# -*- coding: utf-8 -*-

set -eux

git clean -fxd
mkdir plots reports
source .jenkins/scripts/download_and_install_anaconda.sh
pip install tox tox-conda
